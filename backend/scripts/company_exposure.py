"""Operator CLI for company-exposure research (US verify-only shadow slice).

Read-only by default; every mutation needs an explicit flag. Nothing here
enables research, paid search or publication: research mode, routes and the
local allocation come from the environment (see
docs/runbooks/company-exposure-map.md). Secret values are never printed.

    python scripts/company_exposure.py status
    python scripts/company_exposure.py job JOB_ID
    python scripts/company_exposure.py resolve-issuer --security-id 42 --cik 1234567 [--apply]
    python scripts/company_exposure.py resume JOB_ID [--apply]
    python scripts/company_exposure.py process [--max-steps 3]
    python scripts/company_exposure.py inspect-holds
    python scripts/company_exposure.py refresh-holds [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

EXIT_OK, EXIT_BLOCKED, EXIT_USAGE = 0, 2, 64


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _session_factory():
    from app.database import SessionLocal

    return SessionLocal


def status(session, config) -> dict:
    from sqlalchemy import func, select, text
    from sqlalchemy.exc import SQLAlchemyError

    from app.models.company_exposure import ResearchEvent, ResearchWorkItem
    from app.services.company_exposure.activation import evaluate_all

    latest = (
        select(ResearchEvent.request_id, func.max(ResearchEvent.sequence).label("seq"))
        .group_by(ResearchEvent.request_id)
        .subquery()
    )
    states = Counter(
        session.execute(
            select(ResearchEvent.state).join(
                latest,
                (ResearchEvent.request_id == latest.c.request_id)
                & (ResearchEvent.sequence == latest.c.seq),
            )
        ).scalars()
    )
    work = Counter(session.execute(select(ResearchWorkItem.status)).scalars())
    try:
        head = (
            session.execute(text("SELECT version_num FROM alembic_version"))
            .scalars()
            .all()
        )
    except SQLAlchemyError:  # table absent in a scratch database
        session.rollback()
        head = []
    return {
        "configuration": config.public_status(),
        "document_store": config.document_store,
        "activation": [
            {"stage": d.stage, "allowed": d.allowed, "reasons": list(d.reasons)}
            for d in evaluate_all(config, check_storage=True)
        ],
        "jobs_by_state": dict(sorted(states.items())),
        "work_items_by_status": dict(sorted(work.items())),
        "migration_heads": head,
    }


def resolve_issuer(
    session, *, security_id: int, cik: str, apply: bool, admin_subject: str
) -> dict:
    from app.domain.economic_taxonomy.contracts import AdminPrincipal
    from app.services.company_exposure.issuer_identity import (
        IssuerIdentityAdapter,
        LinkProposal,
    )

    identity = IssuerIdentityAdapter(session)
    current = identity.resolve_security(security_id)
    if current.resolved:
        return {"state": "already_linked", "issuer_id": str(current.issuer_id)}
    if not apply:
        return {"state": "dry_run", "security_id": security_id, "cik": cik.zfill(10)}
    if not admin_subject:
        return {"state": "blocked", "reason": "admin_principal_unbound"}
    proposal = identity.propose_link(
        LinkProposal(
            security_id=security_id,
            issuer_id=None,
            identifiers=(("US", "cik", cik),),
            evidence={"reference": "operator-reviewed CIK via CLI"},
            requested_by=admin_subject,
            reason="operator resolved a review_required issuer match",
        )
    )
    if proposal.link_revision_id is None:
        return {"state": proposal.state, "reason": proposal.reason}
    principal = AdminPrincipal(
        subject=admin_subject,
        auth_method="operator_cli",
        roles=frozenset({"taxonomy:review"}),
    )
    ref = identity.apply_link(
        proposal.link_revision_id, principal, proposal.proposal_hash
    )
    session.commit()
    return {
        "state": ref.state,
        "issuer_id": str(ref.issuer_id),
        "link_revision_id": str(ref.link_revision_id),
    }


def resume(session, *, job_id: UUID, apply: bool) -> dict:
    from app.infra.db.repositories.company_exposure_work_repo import (
        CompanyExposureWorkRepository,
    )
    from app.services.company_exposure.reads import ResearchJobReader

    job = ResearchJobReader(session).read(job_id)
    if job is None:
        return {"state": "not_found"}
    if not apply:
        return {
            "state": "dry_run",
            "current_state": job["state"],
            "condition": job["condition"],
        }
    CompanyExposureWorkRepository(session).resume(job_id)
    session.commit()
    return {"state": "queued", "previous_state": job["state"]}


def inspect_holds(session) -> list[dict]:
    from sqlalchemy import select

    from app.models.company_exposure import ExposureUseHoldRevision

    latest: dict[str, ExposureUseHoldRevision] = {}
    for row in session.execute(select(ExposureUseHoldRevision)).scalars():
        if (
            row.stream_key not in latest
            or row.revision_number > latest[row.stream_key].revision_number
        ):
            latest[row.stream_key] = row
    return [
        {
            "subject_kind": row.subject_kind,
            "subject_id": row.subject_id,
            "hold_kind": row.hold_kind,
            "reason": row.reason,
            "since": row.created_at,
        }
        for row in sorted(latest.values(), key=lambda r: r.stream_key)
        if row.action == "apply"
    ]


def refresh_holds(session, *, apply: bool) -> dict:
    from app.services.company_exposure.freshness import refresh_due_holds

    report = refresh_due_holds(session)
    if apply:
        session.commit()
    else:
        session.rollback()
    return {
        "applied": apply,
        "new_holds": len(report.new_holds),
        "evaluated_at": report.evaluated_at,
    }


def main(
    argv: list[str] | None = None,
    *,
    session_factory: Callable | None = None,
    config_loader: Callable | None = None,
    worker: Callable | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Company exposure operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    job = sub.add_parser("job")
    job.add_argument("job_id", type=UUID)
    resolve = sub.add_parser("resolve-issuer")
    resolve.add_argument("--security-id", type=int, required=True)
    resolve.add_argument("--cik", required=True)
    resolve.add_argument("--apply", action="store_true")
    res = sub.add_parser("resume")
    res.add_argument("job_id", type=UUID)
    res.add_argument("--apply", action="store_true")
    proc = sub.add_parser("process")
    proc.add_argument("--max-steps", type=int, default=3)
    sub.add_parser("inspect-holds")
    refresh = sub.add_parser("refresh-holds")
    refresh.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "resolve-issuer" and not (
        args.cik.isdigit() and len(args.cik) <= 10
    ):
        print("--cik must be 1-10 digits", file=sys.stderr)
        return EXIT_USAGE
    if args.command == "process" and not 1 <= args.max_steps <= 20:
        print("--max-steps must be between 1 and 20", file=sys.stderr)
        return EXIT_USAGE

    if config_loader is None:
        from app.services.company_exposure.config import load_config as config_loader
    config = config_loader()

    if args.command == "process":
        if worker is None:
            from app.tasks.company_exposure_tasks import process_exposure_work

            worker = process_exposure_work.run
        outcome = worker(max_steps=args.max_steps)
        _print(outcome)
        return (
            EXIT_OK
            if outcome.get("status") != "skipped" or outcome.get("reason") == "no_work"
            else EXIT_BLOCKED
        )

    session = (session_factory or _session_factory())()
    try:
        if args.command == "status":
            payload = status(session, config)
            _print(payload)
            return EXIT_OK
        if args.command == "job":
            from app.services.company_exposure.reads import ResearchJobReader

            payload = ResearchJobReader(session).read(args.job_id)
            _print(payload or {"state": "not_found"})
            return EXIT_OK if payload else EXIT_BLOCKED
        if args.command == "resolve-issuer":
            from app.config import settings

            payload = resolve_issuer(
                session,
                security_id=args.security_id,
                cik=args.cik,
                apply=args.apply,
                admin_subject=(
                    getattr(settings, "admin_principal_id", "") or ""
                ).strip(),
            )
            _print(payload)
            return EXIT_BLOCKED if payload["state"] == "blocked" else EXIT_OK
        if args.command == "resume":
            payload = resume(session, job_id=args.job_id, apply=args.apply)
            _print(payload)
            return EXIT_BLOCKED if payload["state"] == "not_found" else EXIT_OK
        if args.command == "inspect-holds":
            _print(inspect_holds(session))
            return EXIT_OK
        if args.command == "refresh-holds":
            _print(refresh_holds(session, apply=args.apply))
            return EXIT_OK
    finally:
        session.close()
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
