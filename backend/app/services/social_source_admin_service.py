"""Short, serialized source-registry transactions. No provider I/O belongs here."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
import re
from uuid import uuid4

from sqlalchemy import select, update

from app.domain.social_signals.records import SocialSourceAuditView, SocialSourceView, SourceTestOutcome
from app.infra.db.models.social_signals import SocialSourceAuditEvent, SocialSourceConfiguration, SocialSourceRegistry
from app.models.theme import ContentSource


SEED_SOCIAL_SOURCES = (
    ("1522014550211457024", "Minervini Research List", "https://x.com/i/lists/1522014550211457024"),
    ("1986290701492232693", "Asia-Pacific Growth List", "https://x.com/i/lists/1986290701492232693"),
)


class SocialSourceStateError(ValueError):
    pass


class SocialSourceVersionError(SocialSourceStateError):
    pass


@dataclass(frozen=True, slots=True)
class SocialRuntimeState:
    """Version is the shared administrative policy generation, including sources."""
    mode: str
    provider: str
    version: int


@dataclass(frozen=True, slots=True)
class SocialSourceTestRequest:
    request_id: str
    source_id: str
    list_id: str
    provider: str
    version: int
    registry_version: int


def parse_social_list_ref(value: str) -> str:
    match = re.fullmatch(r"(?:https://x\.com/i/lists/)?([0-9]{1,32})", str(value))
    if not match:
        raise SocialSourceStateError("invalid_list_ref")
    return str(int(match.group(1)))


def _name(value):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 100:
        raise SocialSourceStateError("invalid_name")
    return value.strip()


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


class SocialSourceAdminService:
    """Owns commit/rollback. Call with a Session outside a caller transaction.

    PostgreSQL locks the migration-seeded row; SQLite's no-op UPDATE acquires
    the database write lock before any validation reads. No process-local lock
    is involved. Startup readers never initialize or apply environment values.
    """

    def __init__(self, db):
        self.db = db

    @contextmanager
    def _transaction(self, *, lock=False, initialize=False):
        if self.db.in_transaction():
            raise SocialSourceStateError("caller_transaction_active")
        with self.db.begin():
            if lock:
                self._policy_changed = False
                dialect = self.db.get_bind().dialect.name
                if initialize:
                    if dialect == "sqlite":
                        from sqlalchemy.dialects.sqlite import insert
                    elif dialect == "postgresql":
                        from sqlalchemy.dialects.postgresql import insert
                    else:
                        raise SocialSourceStateError("unsupported_database")
                    self.db.execute(insert(SocialSourceRegistry).values(id=1).on_conflict_do_nothing(index_elements=["id"]))
                if dialect == "sqlite":
                    self.db.execute(update(SocialSourceRegistry).where(SocialSourceRegistry.id == 1).values(version=SocialSourceRegistry.version))
                elif dialect != "postgresql":
                    raise SocialSourceStateError("unsupported_database")
                registry = self.db.execute(select(SocialSourceRegistry).where(SocialSourceRegistry.id == 1).with_for_update().execution_options(populate_existing=True)).scalar_one_or_none()
                if registry is None:
                    raise SocialSourceStateError("registry_not_initialized")
                yield registry
                if self._policy_changed:
                    registry.version += 1
                    registry.updated_at = datetime.now(timezone.utc)
            else:
                yield

    def _source(self, source_id):
        row = self.db.get(SocialSourceConfiguration, int(source_id), populate_existing=True)
        if row is None:
            raise SocialSourceStateError("source_not_found")
        return row

    @staticmethod
    def _version(row, expected_version):
        if row.version != expected_version:
            raise SocialSourceVersionError("version_conflict")

    def _view(self, row):
        source = self.db.get(ContentSource, row.content_source_id)
        outcome = None
        if row.test_status not in {None, "queued", "running"}:
            outcome = SourceTestOutcome(row.tested_provider, row.test_status, row.test_sample_count, _utc(row.tested_at))
        progress = row.test_status if row.test_status in {"queued", "running"} else None
        return SocialSourceView(str(source.id), source.name, source.url, row.x_list_id, row.lifecycle_state, row.provenance, outcome, _utc(row.last_successful_collection_at), row.version, _utc(row.created_at), _utc(row.updated_at), progress)

    def _metadata(self, row):
        view = self._view(row)
        return {"source_id": view.source_id, "name": view.name, "list_id": view.list_id,
                "lifecycle": view.lifecycle, "provenance": view.provenance,
                "version": str(view.version), "tested_provider": row.tested_provider,
                "test_status": row.test_status, "test_request_id": row.test_request_id,
                "test_sample_count": str(row.test_sample_count) if row.test_sample_count is not None else None,
                "tested_at": _utc(row.tested_at).isoformat() if row.tested_at is not None else None}

    def _audit(self, action, actor, after, before=None, source_id=None):
        if not isinstance(actor, str) or not actor.strip():
            raise SocialSourceStateError("invalid_actor")
        self.db.add(SocialSourceAuditEvent(scope="source" if source_id is not None else "runtime", registry_id=1, content_source_id=source_id, action=action, actor=actor, before_json=before, after_json=after, created_at=datetime.now(timezone.utc)))
        if source_id is not None:
            self._policy_changed = True

    def _changed(self, row, action, actor, before):
        row.version += 1
        row.updated_at = datetime.now(timezone.utc)
        self.db.flush()
        self._audit(action, actor, self._metadata(row), before, row.content_source_id)
        return self._view(row)

    def read_runtime(self):
        with self._transaction():
            row = self.db.get(SocialSourceRegistry, 1, populate_existing=True)
            return SocialRuntimeState(row.mode, row.provider, row.version) if row else SocialRuntimeState("off", "disabled", 0)

    def apply_runtime(self, mode, provider, expected_version, actor):
        if mode not in {"off", "validation", "live"} or provider not in {"disabled", "official", "xui"}:
            raise SocialSourceStateError("invalid_runtime")
        with self._transaction(lock=True) as registry:
            self._version(registry, expected_version)
            if (registry.mode, registry.provider) == (mode, provider):
                return SocialRuntimeState(mode, provider, registry.version)
            before = {"mode": registry.mode, "provider": registry.provider, "version": str(registry.version)}
            registry.mode, registry.provider = mode, provider
            registry.version += 1
            registry.updated_at = datetime.now(timezone.utc)
            self._audit("runtime_changed", actor, {"mode": mode, "provider": provider, "version": str(registry.version)}, before)
            return SocialRuntimeState(mode, provider, registry.version)

    def apply_deployment_settings(self, settings, expected_version, actor):
        return self.apply_runtime(settings.social_signals_mode, settings.social_ingest_provider, expected_version, actor)

    def reserve_official_capacity(self, day, requested_posts, daily_limit):
        """Atomically reserve the conservative maximum size of one official read."""
        if (not isinstance(day, date) or isinstance(day, datetime)
                or not isinstance(requested_posts, int) or isinstance(requested_posts, bool)
                or not isinstance(daily_limit, int) or isinstance(daily_limit, bool)
                or requested_posts <= 0 or daily_limit <= 0):
            raise SocialSourceStateError("invalid_official_capacity_request")
        with self._transaction(lock=True) as registry:
            if registry.official_budget_day is None or day > registry.official_budget_day:
                registry.official_budget_day = day
                registry.official_reserved_posts = 0
            elif day < registry.official_budget_day:
                return 0
            remaining = max(0, daily_limit - registry.official_reserved_posts)
            granted = min(requested_posts, remaining)
            registry.official_reserved_posts += granted
            return granted

    def _create(self, name, list_id, actor, *, seed=False):
        if self.db.query(SocialSourceConfiguration).filter_by(x_list_id=list_id).first():
            raise SocialSourceStateError("duplicate_list_id")
        url = f"https://x.com/i/lists/{list_id}"
        source = self.db.query(ContentSource).filter_by(source_type="twitter", url=url).first()
        if source is None:
            source = ContentSource(name=name, source_type="twitter", url=url)
            self.db.add(source)
        source.is_active = seed
        source.name = name
        source.pipelines = ["technical"]
        source.fetch_interval_minutes = 360
        self.db.flush()
        now = datetime.now(timezone.utc)
        row = SocialSourceConfiguration(content_source_id=source.id, x_list_id=list_id, lifecycle_state="enabled" if seed else "pending", provenance="system_seed" if seed else "admin", version=1, created_at=now, updated_at=now)
        self.db.add(row)
        self.db.flush()
        self._audit("created", actor, self._metadata(row), source_id=source.id)
        return self._view(row)

    def ensure_seed_sources(self):
        with self._transaction(lock=True, initialize=True):
            result = []
            for list_id, name, _ in SEED_SOCIAL_SOURCES:
                row = self.db.query(SocialSourceConfiguration).filter_by(x_list_id=list_id).first()
                result.append(self._view(row) if row else self._create(name, list_id, "system", seed=True))
            return tuple(result)

    def create_source(self, name, list_ref, actor):
        name, list_id = _name(name), parse_social_list_ref(list_ref)
        with self._transaction(lock=True):
            return self._create(name, list_id, actor)

    def rename_source(self, source_id, name, expected_version, actor):
        name = _name(name)
        with self._transaction(lock=True):
            row = self._source(source_id)
            self._version(row, expected_version)
            if row.lifecycle_state == "archived":
                raise SocialSourceStateError("source_archived")
            if self.db.get(ContentSource, row.content_source_id).name == name:
                return self._view(row)
            before = self._metadata(row)
            self.db.get(ContentSource, row.content_source_id).name = name
            return self._changed(row, "renamed", actor, before)

    def request_test(self, source_id, expected_version, actor):
        with self._transaction(lock=True) as registry:
            row = self._source(source_id)
            self._version(row, expected_version)
            if row.lifecycle_state == "archived":
                raise SocialSourceStateError("source_archived")
            if row.lifecycle_state not in {"pending", "disabled"}:
                raise SocialSourceStateError("source_test_not_required")
            if registry.provider == "disabled":
                raise SocialSourceStateError("provider_disabled")
            before = self._metadata(row)
            row.test_request_id = str(uuid4())
            row.test_request_version = row.version + 1
            row.test_registry_version = registry.version + 1
            row.tested_provider = registry.provider
            row.test_status = "queued"
            row.tested_at = row.test_sample_count = None
            self._changed(row, "test_requested", actor, before)
            return SocialSourceTestRequest(row.test_request_id, str(row.content_source_id), row.x_list_id, registry.provider, row.version, registry.version + 1)

    def claim_test(self, source_id, actor):
        """Move one exact queued diagnostic to running without external I/O."""
        if not isinstance(actor, str) or not actor.strip():
            raise SocialSourceStateError("invalid_actor")
        with self._transaction(lock=True) as registry:
            row = self._source(source_id)
            if (row.test_status != "queued" or not row.test_request_id
                    or row.lifecycle_state not in {"pending", "disabled"}
                    or row.test_request_version != row.version
                    or row.test_registry_version != registry.version
                    or row.tested_provider != registry.provider
                    or registry.mode == "off" or registry.provider == "disabled"):
                raise SocialSourceStateError("stale_test_request")
            row.test_status = "running"
            return SocialSourceTestRequest(
                row.test_request_id, str(row.content_source_id), row.x_list_id,
                registry.provider, row.version, registry.version,
            )

    def record_test_result(self, source_id, provider, outcome, actor, *, request_id, expected_version):
        with self._transaction(lock=True) as registry:
            row = self._source(source_id)
            self._version(row, expected_version)
            if (row.lifecycle_state == "archived" or row.test_request_id != request_id
                    or row.test_request_version != expected_version or row.test_status not in {"queued", "running"}
                    or row.test_registry_version != registry.version or provider != registry.provider
                    or row.tested_provider != provider or outcome.provider != provider):
                raise SocialSourceStateError("stale_test_result")
            before = self._metadata(row)
            row.test_status, row.test_sample_count, row.tested_at = outcome.status, outcome.sample_count, outcome.tested_at
            return self._changed(row, "test_completed", actor, before)

    def transition_source(self, source_id, target, expected_version, actor):
        if target not in {"enabled", "disabled", "archived"}:
            raise SocialSourceStateError("invalid_target_state")
        with self._transaction(lock=True) as registry:
            row = self._source(source_id)
            self._version(row, expected_version)
            if row.lifecycle_state == target:
                return self._view(row)
            if row.lifecycle_state == "archived":
                raise SocialSourceStateError("source_archived")
            if row.lifecycle_state == "enabled" and target != "enabled":
                count = self.db.query(SocialSourceConfiguration).filter_by(lifecycle_state="enabled").count()
                if count <= 2:
                    raise SocialSourceStateError("minimum_two_enabled")
            if target == "enabled" and row.provenance != "system_seed":
                if row.test_status != "passed" or row.tested_provider != registry.provider:
                    raise SocialSourceStateError("current_provider_test_required")
            before = self._metadata(row)
            row.lifecycle_state = target
            row.archived_at = datetime.now(timezone.utc) if target == "archived" else None
            self.db.get(ContentSource, row.content_source_id).is_active = target == "enabled"
            return self._changed(row, target, actor, before)

    def list_sources(self, include_archived=False):
        with self._transaction():
            query = self.db.query(SocialSourceConfiguration)
            if not include_archived:
                query = query.filter(SocialSourceConfiguration.lifecycle_state != "archived")
            return tuple(self._view(row) for row in query.order_by(SocialSourceConfiguration.content_source_id))

    def audit_events(self, source_id):
        with self._transaction():
            rows = self.db.query(SocialSourceAuditEvent).filter_by(content_source_id=int(source_id)).order_by(SocialSourceAuditEvent.id)
            return tuple(SocialSourceAuditView(row.action, row.actor, _utc(row.created_at), tuple((row.before_json or {}).items()), tuple(row.after_json.items())) for row in rows)
