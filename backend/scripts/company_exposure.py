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


def resume(session, config, *, job_id: UUID, apply: bool) -> dict:
    from app.services.company_exposure.reads import ResearchJobReader
    from app.services.company_exposure.research_requests import ResearchRequests

    job = ResearchJobReader(session).read(job_id)
    if job is None:
        return {"state": "not_found"}
    if not apply:
        return {
            "state": "dry_run",
            "current_state": job["state"],
            "condition": job["condition"],
        }
    ResearchRequests(session, config).resume(job_id)
    session.commit()
    return {"state": "queued", "previous_state": job["state"]}


def inspect_holds(session) -> list[dict]:
    from app.services.company_exposure.holds import HoldRegistry

    return [
        {
            "subject_kind": row.subject_kind,
            "subject_id": row.subject_id,
            "hold_kind": row.hold_kind,
            "reason": row.reason,
            "since": row.created_at,
        }
        for row in HoldRegistry(session).all_active()
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


def _admin_subject() -> str:
    from app.config import settings

    return (settings.admin_principal_id or "").strip()


def _cmd_status(session, config, args):
    return status(session, config), EXIT_OK


def _cmd_job(session, config, args):
    from app.services.company_exposure.reads import ResearchJobReader

    payload = ResearchJobReader(session).read(args.job_id)
    return (payload, EXIT_OK) if payload else ({"state": "not_found"}, EXIT_BLOCKED)


def _cmd_resolve_issuer(session, config, args):
    payload = resolve_issuer(
        session,
        security_id=args.security_id,
        cik=args.cik,
        apply=args.apply,
        admin_subject=_admin_subject(),
    )
    return payload, EXIT_BLOCKED if payload["state"] == "blocked" else EXIT_OK


def _cmd_resume(session, config, args):
    payload = resume(session, config, job_id=args.job_id, apply=args.apply)
    return payload, EXIT_BLOCKED if payload["state"] == "not_found" else EXIT_OK


def _cmd_inspect_holds(session, config, args):
    return inspect_holds(session), EXIT_OK


def _cmd_refresh_holds(session, config, args):
    return refresh_holds(session, apply=args.apply), EXIT_OK


def _cik(value: str) -> str:
    if not (value.isdigit() and len(value) <= 10):
        raise argparse.ArgumentTypeError("must be 1-10 digits")
    return value


def _steps(value: str) -> int:
    steps = int(value)
    if not 1 <= steps <= 20:
        raise argparse.ArgumentTypeError("must be between 1 and 20")
    return steps


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Company exposure operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(handler=_cmd_status)
    job = sub.add_parser("job")
    job.add_argument("job_id", type=UUID)
    job.set_defaults(handler=_cmd_job)
    resolve = sub.add_parser("resolve-issuer")
    resolve.add_argument("--security-id", type=int, required=True)
    resolve.add_argument("--cik", type=_cik, required=True)
    resolve.add_argument("--apply", action="store_true")
    resolve.set_defaults(handler=_cmd_resolve_issuer)
    res = sub.add_parser("resume")
    res.add_argument("job_id", type=UUID)
    res.add_argument("--apply", action="store_true")
    res.set_defaults(handler=_cmd_resume)
    proc = sub.add_parser("process")
    proc.add_argument("--max-steps", type=_steps, default=3)
    proc.set_defaults(handler=None)
    sub.add_parser("inspect-holds").set_defaults(handler=_cmd_inspect_holds)
    refresh = sub.add_parser("refresh-holds")
    refresh.add_argument("--apply", action="store_true")
    refresh.set_defaults(handler=_cmd_refresh_holds)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    session_factory: Callable | None = None,
    config_loader: Callable | None = None,
    worker: Callable | None = None,
) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK

    if args.command == "process":
        # The worker task loads its own configuration and session.
        if worker is None:
            from app.tasks.company_exposure_tasks import process_exposure_work

            worker = process_exposure_work.run
        outcome = worker(max_steps=args.max_steps)
        _print(outcome)
        blocked = (
            outcome.get("status") == "skipped" and outcome.get("reason") != "no_work"
        )
        return EXIT_BLOCKED if blocked else EXIT_OK

    if config_loader is None:
        from app.services.company_exposure.config import load_config as config_loader
    session = (session_factory or _session_factory())()
    try:
        payload, code = args.handler(session, config_loader(), args)
    finally:
        session.close()
    _print(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
