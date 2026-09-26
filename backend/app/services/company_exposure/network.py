"""Bounded public document transport for exposure research.

Extends the pattern of ``theme_evaluation.public_fetch.fetch_public`` (whose
public contract stays unchanged): every hop's hostname must be on the
target's approved origin list, all resolved addresses must be global, and
the connection is pinned to the checked address while TLS still validates
the original hostname (SNI). A second DNS answer therefore cannot rebind the
connection.

Additional rules for research acquisition:

* URLs with embedded credentials are refused; credentials configured for a
  fixed official API are sent only to that exact HTTPS host and never on a
  redirect to anywhere else.
* ``Accept-Encoding: identity`` only, so no decompression bombs; a response
  that is encoded anyway is refused.
* Byte limits are enforced while streaming (a dishonest ``Content-Length``
  cannot bypass them).
* Outcomes are typed values, not exceptions: 404, 429, blocked destination
  and size limits become coverage facts, never negative exposure evidence.
* Every hop (including redirects) calls ``before_request`` so the caller can
  acquire shared provider pacing for each HTTP attempt.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

USER_AGENT_DEFAULT = "StockScreenerExposureResearch/1.0"
_REDIRECTS = (301, 302, 303, 307, 308)


def default_resolver(host: str) -> list[str]:
    return list(
        dict.fromkeys(
            row[4][0] for row in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        )
    )


@dataclass(frozen=True, slots=True)
class FetchRequest:
    url: str
    allowed_hosts: tuple[str, ...]
    max_bytes: int
    user_agent: str = USER_AGENT_DEFAULT
    accept: str = "*/*"
    timeout_seconds: float = 30.0
    max_redirects: int = 5
    if_none_match: str | None = None
    if_modified_since: str | None = None
    credential_host: str | None = None
    credential_headers: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FetchResponse:
    ok: bool
    status: int | None
    final_url: str
    body: bytes = b""
    content_type: str = ""
    failure_code: str | None = None
    not_modified: bool = False
    etag: str | None = None
    last_modified: str | None = None
    retry_after: str | None = None
    hops: tuple[str, ...] = ()


def sanitize_url(url: str) -> str:
    """Drop userinfo, query and fragment: citation URLs never carry secrets."""

    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = host if parts.port is None else f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def host_allowed(host: str, allowed_hosts: Iterable[str]) -> bool:
    host = host.lower().rstrip(".")
    for allowed in allowed_hosts:
        allowed = allowed.lower().rstrip(".")
        if allowed.startswith("*."):
            suffix = allowed[1:]
            if host.endswith(suffix) and host != suffix[1:]:
                return True
        elif host == allowed:
            return True
    return False


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


class PublicDocumentTransport:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        resolver: Callable[[str], list[str]] = default_resolver,
    ):
        self._transport = transport
        self._resolver = resolver

    def _check(self, url: str, request: FetchRequest) -> tuple[str | None, str | None]:
        """Return ``(pinned_address, failure_code)`` for one hop."""

        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None, "unsupported_url"
        if parts.username is not None or parts.password is not None:
            return None, "url_credentials_forbidden"
        if parts.port not in (None, 80, 443):
            return None, "blocked_destination"
        try:
            literal = ipaddress.ip_address(parts.hostname)
        except ValueError:
            literal = None
        if literal is not None and not _is_public(str(literal)):
            return None, "blocked_destination"
        if not host_allowed(parts.hostname, request.allowed_hosts):
            return None, "origin_not_permitted"
        try:
            addresses = self._resolver(parts.hostname)
        except (OSError, UnicodeError):
            return None, "dns_failed"
        try:
            if not addresses or not all(_is_public(a) for a in addresses):
                return None, "blocked_destination"
        except ValueError:
            return None, "blocked_destination"
        return addresses[0], None

    def fetch_once(
        self,
        request: FetchRequest,
        *,
        before_request: Callable[[str], None] | None = None,
    ) -> FetchResponse:
        if request.max_bytes <= 0 or not 0 <= request.max_redirects <= 10:
            raise ValueError("invalid_fetch_limits")
        url = request.url
        hops: list[str] = []
        try:
            with httpx.Client(
                transport=self._transport,
                trust_env=False,
                timeout=httpx.Timeout(request.timeout_seconds, connect=10.0),
                follow_redirects=False,
            ) as client:
                for _ in range(request.max_redirects + 1):
                    address, failure = self._check(url, request)
                    hops.append(sanitize_url(url))
                    if failure is not None:
                        return FetchResponse(
                            False, None, sanitize_url(url), failure_code=failure,
                            hops=tuple(hops),
                        )
                    if before_request is not None:
                        before_request(url)
                    original = httpx.URL(url)
                    pinned = original.copy_with(host=address)
                    headers = {
                        "Host": original.netloc.decode(),
                        "User-Agent": request.user_agent,
                        "Accept": request.accept,
                        "Accept-Encoding": "identity",
                    }
                    if request.if_none_match:
                        headers["If-None-Match"] = request.if_none_match
                    if request.if_modified_since:
                        headers["If-Modified-Since"] = request.if_modified_since
                    if (
                        request.credential_headers
                        and original.scheme == "https"
                        and original.host == request.credential_host
                    ):
                        headers.update(request.credential_headers)
                    client.cookies.clear()
                    with client.stream(
                        "GET",
                        pinned,
                        headers=headers,
                        extensions={"sni_hostname": original.host},
                    ) as response:
                        status = response.status_code
                        if status in _REDIRECTS:
                            location = response.headers.get("location")
                            if not location:
                                return FetchResponse(
                                    False, status, sanitize_url(url),
                                    failure_code="redirect_without_destination",
                                    hops=tuple(hops),
                                )
                            url = urljoin(url, location)
                            continue
                        if status == 304:
                            return FetchResponse(
                                True, status, sanitize_url(url), not_modified=True,
                                etag=response.headers.get("etag"),
                                last_modified=response.headers.get("last-modified"),
                                hops=tuple(hops),
                            )
                        if status != 200:
                            return FetchResponse(
                                False, status, sanitize_url(url),
                                failure_code=(
                                    "rate_limited" if status == 429 else f"http_status_{status}"
                                ),
                                retry_after=response.headers.get("retry-after"),
                                hops=tuple(hops),
                            )
                        encoding = (
                            response.headers.get("content-encoding", "identity")
                            .lower()
                            .strip()
                        )
                        if encoding != "identity":
                            return FetchResponse(
                                False, status, sanitize_url(url),
                                failure_code="response_encoding_unsupported",
                                hops=tuple(hops),
                            )
                        declared = response.headers.get("content-length")
                        if declared is not None:
                            try:
                                if int(declared) > request.max_bytes:
                                    return FetchResponse(
                                        False, status, sanitize_url(url),
                                        failure_code="response_size_limit",
                                        hops=tuple(hops),
                                    )
                            except ValueError:
                                return FetchResponse(
                                    False, status, sanitize_url(url),
                                    failure_code="response_invalid", hops=tuple(hops),
                                )
                        body = bytearray()
                        for chunk in response.iter_bytes(chunk_size=65536):
                            body.extend(chunk)
                            if len(body) > request.max_bytes:
                                return FetchResponse(
                                    False, status, sanitize_url(url),
                                    failure_code="response_size_limit",
                                    hops=tuple(hops),
                                )
                        return FetchResponse(
                            True, status, sanitize_url(url), bytes(body),
                            content_type=response.headers.get("content-type", ""),
                            etag=response.headers.get("etag"),
                            last_modified=response.headers.get("last-modified"),
                            hops=tuple(hops),
                        )
                return FetchResponse(
                    False, None, sanitize_url(url), failure_code="redirect_limit",
                    hops=tuple(hops),
                )
        except httpx.TimeoutException:
            return FetchResponse(
                False, None, sanitize_url(url), failure_code="fetch_timeout",
                hops=tuple(hops),
            )
        except (httpx.HTTPError, OSError):
            return FetchResponse(
                False, None, sanitize_url(url), failure_code="fetch_failed",
                hops=tuple(hops),
            )


# Leading bytes of formats research may retain. Anything executable or
# archived is refused rather than stored.
_SIGNATURES = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)
_REFUSED = (
    (b"MZ", "executable"),
    (b"\x7fELF", "executable"),
    (b"PK\x03\x04", "archive"),
    (b"\x1f\x8b", "archive"),
    (b"Rar!", "archive"),
    (b"7z\xbc\xaf", "archive"),
)


def sniff_media_type(body: bytes, declared: str) -> tuple[str | None, str | None]:
    """Return ``(media_type, refusal_reason)`` from content, not headers."""

    head = body[:512]
    for signature, reason in _REFUSED:
        if head.startswith(signature):
            return None, f"media_{reason}_refused"
    for signature, media in _SIGNATURES:
        if head.startswith(signature):
            return media, None
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if text.startswith((b"<!doctype html", b"<html", b"<head", b"<body")) or (
        b"<html" in text
    ):
        return "text/html", None
    if text.startswith((b"{", b"[")):
        return "application/json", None
    if text.startswith((b"<?xml", b"<")):
        return "application/xml" if "xml" in declared else "text/html", None
    try:
        body[:4096].decode("utf-8")
    except UnicodeDecodeError:
        return None, "media_unrecognized"
    return "text/plain", None
