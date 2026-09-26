from __future__ import annotations

import httpx
import pytest

from app.services.company_exposure.network import (
    FetchRequest,
    PublicDocumentTransport,
    host_allowed,
    sanitize_url,
    sniff_media_type,
)

PUBLIC_IP = "93.184.216.34"


class NetworkSpy:
    def __init__(self, resolution=None):
        self.resolution = resolution or {"www.sec.gov": [PUBLIC_IP]}
        self.connected_hosts: list[str] = []
        self.headers: list[dict] = []
        self._responses: list = []

    def respond(self, status=200, body=b"<html>ok</html>", headers=None):
        self._responses.append(
            httpx.Response(status, headers=headers or {}, stream=httpx.ByteStream(body))
        )

    def respond_redirect(self, location):
        self._responses.append(httpx.Response(302, headers={"location": location}))

    def resolver(self, host):
        return self.resolution.get(host, [])

    def handler(self, request):
        self.connected_hosts.append(request.headers["host"])
        self.headers.append(dict(request.headers))
        assert request.url.host in {ip for ips in self.resolution.values() for ip in ips}
        return self._responses.pop(0)


@pytest.fixture
def network_spy():
    return NetworkSpy()


@pytest.fixture
def public_transport(network_spy):
    return PublicDocumentTransport(
        transport=httpx.MockTransport(network_spy.handler), resolver=network_spy.resolver
    )


@pytest.fixture
def public_request():
    request = FetchRequest(
        url="https://www.sec.gov/Archives/doc.htm",
        allowed_hosts=("www.sec.gov",),
        max_bytes=1000,
    )
    return request


@pytest.mark.case("R14")
@pytest.mark.exposure_layer("unit")
def test_private_redirect_is_rejected_before_connection(
    public_transport, public_request, network_spy
):
    network_spy.respond_redirect("http://169.254.169.254/latest/meta-data/")
    result = public_transport.fetch_once(public_request)
    assert result.failure_code == "blocked_destination"
    assert network_spy.connected_hosts == ["www.sec.gov"]


@pytest.mark.case("R14")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    ("resolution", "expected"),
    [
        (["127.0.0.1"], "blocked_destination"),
        (["10.0.0.5"], "blocked_destination"),
        ([PUBLIC_IP, "192.168.1.1"], "blocked_destination"),  # mixed answers
        (["::ffff:127.0.0.1"], "blocked_destination"),  # IPv4-mapped loopback
        ([], "blocked_destination"),
    ],
)
def test_non_global_resolution_is_refused(resolution, expected):
    spy = NetworkSpy({"www.sec.gov": resolution})
    transport = PublicDocumentTransport(
        transport=httpx.MockTransport(spy.handler), resolver=spy.resolver
    )
    result = transport.fetch_once(
        FetchRequest(
            url="https://www.sec.gov/x", allowed_hosts=("www.sec.gov",), max_bytes=10
        )
    )
    assert result.failure_code == expected
    assert spy.connected_hosts == []


@pytest.mark.case("R14")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:pw@www.sec.gov/x", "url_credentials_forbidden"),
        ("https://evil.example/x", "origin_not_permitted"),
        ("https://2130706433/x", "origin_not_permitted"),
        ("https://www.sec.gov:8443/x", "blocked_destination"),
        ("ftp://www.sec.gov/x", "unsupported_url"),
    ],
)
def test_url_level_attacks_are_refused(public_transport, network_spy, url, expected):
    result = public_transport.fetch_once(
        FetchRequest(url=url, allowed_hosts=("www.sec.gov",), max_bytes=10)
    )
    assert result.failure_code == expected
    assert network_spy.connected_hosts == []


@pytest.mark.case("R14")
@pytest.mark.exposure_layer("unit")
def test_dishonest_content_length_is_bounded_while_streaming(
    public_transport, public_request, network_spy
):
    network_spy.respond(body=b"x" * 5000, headers={"content-length": "10"})
    result = public_transport.fetch_once(public_request)
    assert result.failure_code in {"response_size_limit", "fetch_failed"}
    assert result.body == b""


def test_encoded_responses_are_refused(public_transport, public_request, network_spy):
    network_spy.respond(headers={"content-encoding": "gzip"})
    assert (
        public_transport.fetch_once(public_request).failure_code
        == "response_encoding_unsupported"
    )


@pytest.mark.case("R14")
@pytest.mark.exposure_layer("unit")
def test_credentials_are_never_forwarded_to_a_redirect_host():
    spy = NetworkSpy({"api.edinet-fsa.go.jp": [PUBLIC_IP], "cdn.example.jp": ["93.184.216.35"]})
    spy.respond_redirect("https://cdn.example.jp/file.pdf")
    spy.respond(body=b"%PDF-1.7")
    transport = PublicDocumentTransport(
        transport=httpx.MockTransport(spy.handler), resolver=spy.resolver
    )
    result = transport.fetch_once(
        FetchRequest(
            url="https://api.edinet-fsa.go.jp/api/v2/documents/1",
            allowed_hosts=("api.edinet-fsa.go.jp", "cdn.example.jp"),
            max_bytes=100,
            credential_host="api.edinet-fsa.go.jp",
            credential_headers={"Ocp-Apim-Subscription-Key": "secret"},
        )
    )
    assert result.ok
    assert "ocp-apim-subscription-key" in spy.headers[0]
    assert "ocp-apim-subscription-key" not in spy.headers[1]
    assert "secret" not in result.final_url


def test_each_hop_calls_before_request(public_transport, public_request, network_spy):
    network_spy.resolution["www.sec.gov"] = [PUBLIC_IP]
    network_spy.respond_redirect("https://www.sec.gov/Archives/final.htm")
    network_spy.respond()
    hops = []
    result = public_transport.fetch_once(public_request, before_request=hops.append)
    assert result.ok and len(hops) == 2


@pytest.mark.case("I06")
@pytest.mark.exposure_layer("unit")
def test_http_errors_are_typed_outcomes(public_transport, public_request, network_spy):
    network_spy.respond(status=404)
    assert public_transport.fetch_once(public_request).failure_code == "http_status_404"
    network_spy.respond(status=429, headers={"retry-after": "5"})
    limited = public_transport.fetch_once(public_request)
    assert (limited.failure_code, limited.retry_after) == ("rate_limited", "5")


def test_helpers():
    assert sanitize_url("https://u:p@host.example/a?token=1#f") == "https://host.example/a"
    assert host_allowed("ir.example.com", ("*.example.com",))
    assert not host_allowed("example.com.evil.net", ("*.example.com",))
    assert sniff_media_type(b"%PDF-1.7 ...", "") == ("application/pdf", None)
    assert sniff_media_type(b"MZ\x90\x00", "application/pdf") == (
        None,
        "media_executable_refused",
    )
    assert sniff_media_type(b"PK\x03\x04", "")[1] == "media_archive_refused"
    assert sniff_media_type(b"<!DOCTYPE html><html>", "")[0] == "text/html"
