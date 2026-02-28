"""Error taxonomy for stable module boundaries."""


class XUIReaderError(Exception):
    """Base exception for xui-reader."""


class ConfigError(XUIReaderError):
    """Raised when configuration is invalid or missing."""


class ProfileError(XUIReaderError):
    """Raised when profile lifecycle operations are unsafe or invalid."""


class AuthError(XUIReaderError):
    """Raised for authentication and storage-state lifecycle failures."""


class BrowserError(XUIReaderError):
    """Raised for browser/session management failures."""


class CollectError(XUIReaderError):
    """Raised for collection lifecycle failures."""


class ExtractError(XUIReaderError):
    """Raised when extraction cannot parse expected structures."""


class StoreError(XUIReaderError):
    """Raised for storage/read checkpoint failures."""


class RenderError(XUIReaderError):
    """Raised when rendering output fails."""


class SchedulerError(XUIReaderError):
    """Raised for watch loop and budget coordination failures."""


class DiagnosticsError(XUIReaderError):
    """Raised for doctor/diagnostics path failures."""
