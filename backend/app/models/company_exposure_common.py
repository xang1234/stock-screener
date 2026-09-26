"""Shared persistence helpers for company-exposure research models.

Evidence, claim, assessment and decision history is append-only. Two layers
enforce that, mirroring the economic-taxonomy runtime:

* an ORM ``before_flush`` guard rejects updates/deletes of registered
  append-only rows (and of sealed rows, for sealable models); and
* PostgreSQL ``BEFORE UPDATE OR DELETE`` triggers reject raw SQL mutation.

Operational tables (leases, queue status caches) are deliberately not
registered here.
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import DDL, Column, DateTime, Uuid, event, func, inspect
from sqlalchemy.orm import Session

from app.models.economic_taxonomy_runtime_common import ImmutableRuntimePayload

APPEND_ONLY_EXPOSURE_MODELS: list[type] = []
SEALABLE_EXPOSURE_MODELS: list[type] = []
# (child model, foreign-key attribute, sealable parent model)
SEALED_CHILD_RELATIONS: list[tuple[type, str, type]] = []

TRIGGER_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION company_exposure_reject_mutation()
RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'company_exposure_payload_immutable';
END;
$$ LANGUAGE plpgsql;
"""

SEAL_ONCE_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION company_exposure_seal_once()
RETURNS trigger AS $$
BEGIN
  IF TG_OP = 'DELETE' OR OLD.status = 'sealed' THEN
    RAISE EXCEPTION 'company_exposure_sealed_payload_immutable';
  END IF;
  IF NEW.status <> 'sealed'
     OR NEW.semantic_hash IS NULL
     OR NEW.sealed_at IS NULL
     OR (to_jsonb(NEW) - ARRAY['status','semantic_hash','sealed_at']::text[])
        IS DISTINCT FROM
        (to_jsonb(OLD) - ARRAY['status','semantic_hash','sealed_at']::text[]) THEN
    RAISE EXCEPTION 'company_exposure_payload_immutable';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


GUARD_SEALED_PARENT_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION company_exposure_guard_sealed_parent()
RETURNS trigger AS $$
DECLARE
  parent_id uuid;
  parent_sealed boolean;
BEGIN
  parent_id := (to_jsonb(NEW) ->> TG_ARGV[1])::uuid;
  -- FOR SHARE conflicts with the sealing UPDATE, so a child cannot slip in
  -- while its parent is being sealed.
  EXECUTE format(
    'SELECT EXISTS (SELECT 1 FROM %I WHERE id = $1 AND status = ''sealed'' FOR SHARE)',
    TG_ARGV[0]
  ) INTO parent_sealed USING parent_id;
  IF parent_sealed THEN
    RAISE EXCEPTION 'company_exposure_sealed_payload_immutable';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


def sealed_parent_trigger_sql(child_table: str, parent_table: str, column: str) -> str:
    return (
        f"CREATE TRIGGER trg_{child_table}_parent_open "
        f"BEFORE INSERT ON {child_table} "
        "FOR EACH ROW EXECUTE FUNCTION "
        f"company_exposure_guard_sealed_parent('{parent_table}', '{column}');"
    )


def append_only_trigger_sql(table_name: str) -> str:
    return (
        f"CREATE TRIGGER trg_{table_name}_append_only "
        f"BEFORE UPDATE OR DELETE ON {table_name} "
        "FOR EACH ROW EXECUTE FUNCTION company_exposure_reject_mutation();"
    )


def seal_once_trigger_sql(table_name: str) -> str:
    return (
        f"CREATE TRIGGER trg_{table_name}_seal_once "
        f"BEFORE UPDATE OR DELETE ON {table_name} "
        "FOR EACH ROW EXECUTE FUNCTION company_exposure_seal_once();"
    )


def uuid_pk():
    return Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)


def created_at():
    return Column(DateTime(timezone=True), nullable=False, server_default=func.now())


def append_only(model: type) -> type:
    """Class decorator registering ORM and PostgreSQL append-only protection."""

    APPEND_ONLY_EXPOSURE_MODELS.append(model)
    event.listen(
        model.__table__,
        "after_create",
        DDL(TRIGGER_FUNCTION_SQL + append_only_trigger_sql(model.__tablename__)).execute_if(
            dialect="postgresql"
        ),
    )
    return model


def sealable(model: type) -> type:
    """Rows with ``status`` unsealed→sealed; sealed rows are immutable."""

    SEALABLE_EXPOSURE_MODELS.append(model)
    event.listen(
        model.__table__,
        "after_create",
        DDL(SEAL_ONCE_FUNCTION_SQL + seal_once_trigger_sql(model.__tablename__)).execute_if(
            dialect="postgresql"
        ),
    )
    return model


def sealed_child(column: str, parent_model: type):
    """Reject inserting children once their sealable parent is sealed."""

    def register(model: type) -> type:
        SEALED_CHILD_RELATIONS.append((model, column, parent_model))
        event.listen(
            model.__table__,
            "after_create",
            DDL(
                # DDL() applies %-formatting; escape the plpgsql format() %I.
                GUARD_SEALED_PARENT_FUNCTION_SQL.replace("%", "%%")
                + sealed_parent_trigger_sql(
                    model.__tablename__, parent_model.__tablename__, column
                )
            ).execute_if(dialect="postgresql"),
        )
        return model

    return register


def _protect_sealable(row) -> None:
    state = inspect(row)
    prior = state.attrs.status.history.deleted
    prior_status = prior[0] if prior else row.status
    if prior_status == "sealed":
        raise ImmutableRuntimePayload("company_exposure_sealed_payload_immutable")
    if row.status == "unsealed":
        return
    changed = {
        attr.key
        for attr in state.mapper.column_attrs
        if state.attrs[attr.key].history.has_changes()
    }
    if row.status != "sealed" or not changed.issubset(
        {"status", "semantic_hash", "sealed_at"}
    ):
        raise ImmutableRuntimePayload("company_exposure_payload_immutable")
    if not row.semantic_hash or row.sealed_at is None:
        raise ImmutableRuntimePayload("company_exposure_sealed_payload_incomplete")


@event.listens_for(Session, "before_flush")
def _protect_company_exposure_history(session, _flush_context, _instances):
    append_only_types = tuple(APPEND_ONLY_EXPOSURE_MODELS)
    sealable_types = tuple(SEALABLE_EXPOSURE_MODELS)
    for row in session.deleted:
        if isinstance(row, append_only_types + sealable_types):
            raise ImmutableRuntimePayload("company_exposure_payload_immutable")
    for row in session.dirty:
        if isinstance(row, append_only_types) and session.is_modified(row):
            raise ImmutableRuntimePayload("company_exposure_payload_immutable")
        if isinstance(row, sealable_types) and session.is_modified(row):
            _protect_sealable(row)
    for row in session.new:
        for child_type, column, parent_type in SEALED_CHILD_RELATIONS:
            if not isinstance(row, child_type):
                continue
            parent_id = getattr(row, column)
            if parent_id is None:
                continue
            with session.no_autoflush:
                parent = session.get(parent_type, parent_id)
            if parent is not None and _persisted_status(parent) == "sealed":
                raise ImmutableRuntimePayload(
                    "company_exposure_sealed_payload_immutable"
                )


def _persisted_status(row) -> str:
    """Status as last flushed, ignoring an unflushed seal in this batch."""

    state = inspect(row)
    if state.pending:
        return row.status
    prior = state.attrs.status.history.deleted
    return prior[0] if prior else row.status


__all__ = (
    "APPEND_ONLY_EXPOSURE_MODELS",
    "GUARD_SEALED_PARENT_FUNCTION_SQL",
    "SEALABLE_EXPOSURE_MODELS",
    "SEALED_CHILD_RELATIONS",
    "SEAL_ONCE_FUNCTION_SQL",
    "TRIGGER_FUNCTION_SQL",
    "ImmutableRuntimePayload",
    "append_only",
    "append_only_trigger_sql",
    "created_at",
    "seal_once_trigger_sql",
    "sealable",
    "sealed_child",
    "sealed_parent_trigger_sql",
    "uuid_pk",
)
