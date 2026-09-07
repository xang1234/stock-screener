"""Administrator-attested issuer mappings; never inferred or machine verified.

HTTP callers must establish admin authority before constructing an authorized
writer. Reads join the caller transaction; replacements own a short transaction.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import re

from sqlalchemy import select

from app.domain.social_signals.records import validate_utc_timestamp
from app.infra.db.models.social_signals import SocialSourceRegistry
from app.models.app_settings import AppSetting
from app.services.social_source_admin_service import SocialSourceAdminService

KEY = "social_company_identities"
POLICY = "admin-attested-company-v1"


@dataclass(frozen=True, slots=True)
class CompanyIdentityEntry:
    symbol: str
    company_id: str
    verification_reference: str
    verified_at: str


@dataclass(frozen=True, slots=True)
class CompanyIdentityConfiguration:
    registry_version: int
    version: int
    entries: tuple[CompanyIdentityEntry, ...]
    policy_version: str = POLICY

    @property
    def verified_company_ids(self):
        return {entry.symbol: entry.company_id for entry in self.entries}


def _entries(values):
    if not isinstance(values, list) or len(values) > 50000:
        raise ValueError("invalid_company_identities")
    result = []
    symbols = set()
    for value in values:
        if not isinstance(value, dict) or set(value) != {"symbol", "company_id", "verification_reference", "verified_at"}:
            raise ValueError("invalid_company_identity")
        if any(not isinstance(v, str) or not v.strip() or len(v) > 500 for v in value.values()):
            raise ValueError("invalid_company_identity")
        symbol = value["symbol"]
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,19}", symbol) or symbol in symbols:
            raise ValueError("invalid_company_symbol")
        validate_utc_timestamp(datetime.fromisoformat(value["verified_at"]), "verified_at")
        symbols.add(symbol)
        result.append(CompanyIdentityEntry(**value))
    return tuple(sorted(result, key=lambda item: item.symbol))


class SocialCompanyIdentityService:
    def __init__(self, db, *, admin_authorized=False):
        self.db = db
        self.admin_authorized = admin_authorized

    def read(self):
        with self.db.no_autoflush:
            registry = self.db.get(SocialSourceRegistry, 1, populate_existing=True)
            setting = self.db.scalar(select(AppSetting).where(AppSetting.key == KEY).execution_options(populate_existing=True))
            data = json.loads(setting.value) if setting else {"version": 0, "policy_version": POLICY, "entries": []}
            if (set(data) != {"version", "policy_version", "entries"}
                    or data["policy_version"] != POLICY or type(data["version"]) is not int or data["version"] < 0):
                raise ValueError("invalid_company_identity_configuration")
            return CompanyIdentityConfiguration(registry.version if registry else 0, data["version"], _entries(data["entries"]))

    def replace(self, entries, *, expected_version, actor):
        if self.admin_authorized is not True:
            raise PermissionError("admin_required")
        validated = _entries(entries)
        admin = SocialSourceAdminService(self.db)
        with admin._transaction(lock=True) as registry:
            admin._version(registry, expected_version)
            before = self.read()
            data = {"version": before.version + 1, "policy_version": POLICY, "entries": [asdict(item) for item in validated]}
            setting = self.db.scalar(select(AppSetting).where(AppSetting.key == KEY))
            if setting is None:
                setting = AppSetting(key=KEY, category="social")
                self.db.add(setting)
            setting.value = json.dumps(data)
            registry.version += 1
            admin._audit("runtime_changed", actor,
                         {"configuration": KEY, "version": data["version"], "entry_count": len(validated), "policy_version": POLICY,
                          "verification_references": sorted({item.verification_reference for item in validated})},
                         {"version": before.version, "entry_count": len(before.entries)})
            return CompanyIdentityConfiguration(registry.version, data["version"], validated)
