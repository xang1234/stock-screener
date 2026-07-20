class NoPublishedStaticMarketArtifact(RuntimeError):
    """Static export cannot find a published market artifact source."""

    def __init__(self, message: str, *, markets: tuple[str, ...] = ()) -> None:
        self.markets = tuple(markets)
        super().__init__(message)


class StaticSiteSectionUnavailableError(RuntimeError):
    """An optional static-site section cannot be exported for the target date."""

    def __init__(self, *, section: str, reason: str) -> None:
        self.section = section
        self.reason = reason
        super().__init__(reason)
