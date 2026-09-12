"""Bounded public HTTP reads with address pinning and per-hop checks."""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from .records import http_url


@dataclass(frozen=True)
class PublicResponse:
    body: bytes
    final_url: str
    content_type: str


def _resolve(host):
    return list(
        dict.fromkeys(
            row[4][0] for row in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        )
    )


def _destination(url, resolver):
    http_url(url)
    parts = urlsplit(url)
    if parts.port not in (None, 80, 443):
        raise ValueError("public_port_required")
    addresses = resolver(parts.hostname)
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError("public_address_required")
    return addresses[0]


def fetch_public(
    url: str, *, max_bytes=5_000_000, max_redirects=5, transport=None, resolver=_resolve
) -> PublicResponse:
    if max_bytes <= 0 or not 0 <= max_redirects <= 10:
        raise ValueError("invalid_fetch_limits")
    try:
        with httpx.Client(
            transport=transport,
            trust_env=False,
            timeout=httpx.Timeout(20),
            follow_redirects=False,
        ) as client:
            for _ in range(max_redirects + 1):
                address = _destination(url, resolver)
                original = httpx.URL(url)
                pinned = original.copy_with(host=address)
                # Connect to the checked address, while retaining TLS verification/SNI
                # and the publisher's Host header. A second DNS lookup cannot rebind it.
                client.cookies.clear()  # Public reads never carry cookies across pinned origins.
                with client.stream(
                    "GET",
                    pinned,
                    headers={
                        "Host": original.netloc.decode(),
                        "User-Agent": "ThemeEvidenceReview/1.0",
                        "Accept-Encoding": "identity",
                    },
                    extensions={"sni_hostname": original.host},
                ) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("redirect_without_destination")
                        url = urljoin(url, location)
                        continue
                    if response.status_code != 200:
                        raise ValueError("http_status_" + str(response.status_code))
                    if (
                        response.headers.get("content-encoding", "identity")
                        .lower()
                        .strip()
                        != "identity"
                    ):
                        raise ValueError("response_encoding_unsupported")
                    if int(response.headers.get("content-length", "0")) > max_bytes:
                        raise ValueError("response_size_limit")
                    body = bytearray()
                    for chunk in response.iter_bytes(chunk_size=65536):
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise ValueError("response_size_limit")
                    return PublicResponse(
                        bytes(body), url, response.headers.get("content-type", "")
                    )
            raise ValueError("redirect_limit")
    except (httpx.HTTPError, OSError) as exc:
        raise ValueError("public_fetch_failed") from exc
