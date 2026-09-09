"""SQL adapter for the social-source validation use case."""

from app.services.social_source_admin_service import SocialSourceAdminService


class SqlSourceTestRegistry:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def claim(self, source_id, actor):
        with self.session_factory() as db:
            return SocialSourceAdminService(db).claim_test(source_id, actor)

    def complete(self, request, outcome, actor):
        with self.session_factory() as db:
            return SocialSourceAdminService(db).record_test_result(
                request.source_id,
                request.provider,
                outcome,
                actor,
                request_id=request.request_id,
                expected_version=request.version,
            )


__all__ = ["SqlSourceTestRegistry"]
