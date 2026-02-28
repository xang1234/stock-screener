"""Browser contracts."""

from .policy import (
    DEFAULT_ALLOWED_OVERLAY_SELECTORS,
    DEFAULT_BLOCKED_RESOURCE_TYPES,
    OverlayDismissResult,
    ResourceRoutingPolicy,
    configure_resource_routing,
    detect_unsupported_overlay_state,
    dismiss_allowed_overlays,
    install_resource_routing,
    make_resource_route_handler,
)
from .session import (
    BrowserPage,
    BrowserSessionManager,
    BrowserSessionOptions,
    PlaywrightBrowserSession,
)

__all__ = [
    "DEFAULT_ALLOWED_OVERLAY_SELECTORS",
    "DEFAULT_BLOCKED_RESOURCE_TYPES",
    "BrowserPage",
    "BrowserSessionManager",
    "BrowserSessionOptions",
    "OverlayDismissResult",
    "PlaywrightBrowserSession",
    "ResourceRoutingPolicy",
    "configure_resource_routing",
    "detect_unsupported_overlay_state",
    "dismiss_allowed_overlays",
    "install_resource_routing",
    "make_resource_route_handler",
]
