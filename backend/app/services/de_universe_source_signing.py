"""Request signing helpers for the Boerse Frankfurt public equity-search API.

The boerse-frankfurt.de single-page app derives an ``X-Security`` header for
every API call from the request URL, an ISO-8601 ``Client-Date``, a per-request
trace id, and a public salt baked into its JavaScript bundle. The endpoint
rejects requests that omit the headers or fail signature verification.

The signing algorithm is not part of an official Deutsche Boerse contract — it
is reverse-engineered from the public web client and may rotate without
notice. When that happens the live fetch will start returning HTTP 401/403,
and ``OfficialMarketUniverseSourceService.fetch_de_snapshot`` is responsible
for catching the failure and falling back to the bundled DE seed CSV. The
helpers in this module are kept pure and isolated so the algorithm can be
swapped without touching the fetcher or its tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import uuid

# Public salt observed in the boerse-frankfurt.de JS bundle as of 2026-05.
# Treated as a versioned constant: when the live API starts rejecting signed
# requests, refresh this value (or accept the CSV-fallback degradation until
# the salt is updated).
_BOERSE_FRANKFURT_PUBLIC_SALT = "w4icutssNbaHJ4ufrFKM7AINGeNSe9aR"

_CLIENT_ORIGIN = "https://www.boerse-frankfurt.de"


def client_date_header(now: datetime | None = None) -> str:
    """Return an ISO-8601 millisecond timestamp for the ``Client-Date`` header.

    Boerse Frankfurt's web client formats the timestamp with a ``+0000``
    timezone offset (no colon), which is what the signature input string
    expects byte-for-byte.
    """
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    iso = moment.isoformat(timespec="milliseconds")
    return iso.replace("+00:00", "+0000")


def client_trace_id() -> str:
    """Return a fresh UUIDv4 hex string for the ``X-Client-TraceId`` header."""
    return uuid.uuid4().hex


def compute_x_security(
    *,
    url: str,
    client_date: str,
    trace_id: str,
    salt: str = _BOERSE_FRANKFURT_PUBLIC_SALT,
) -> str:
    """Compute the SHA-256 ``X-Security`` header expected by the API."""
    payload = f"{trace_id}{url}{client_date}{salt}".encode("utf-8")
    return sha256(payload).hexdigest()


def build_signed_headers(
    url: str,
    *,
    now: datetime | None = None,
    trace_id: str | None = None,
    salt: str = _BOERSE_FRANKFURT_PUBLIC_SALT,
) -> dict[str, str]:
    """Build the full set of headers required for a Boerse Frankfurt request."""
    client_date = client_date_header(now=now)
    resolved_trace_id = trace_id or client_trace_id()
    return {
        "Origin": _CLIENT_ORIGIN,
        "Referer": f"{_CLIENT_ORIGIN}/",
        "Accept": "application/json",
        "Client-Date": client_date,
        "X-Client-TraceId": resolved_trace_id,
        "X-Security": compute_x_security(
            url=url,
            client_date=client_date,
            trace_id=resolved_trace_id,
            salt=salt,
        ),
    }
