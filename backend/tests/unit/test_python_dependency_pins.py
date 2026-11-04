from pathlib import Path
import re

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REQUIREMENT_FILES = (
    Path(__file__).parents[2] / "requirements-runtime.txt",
    Path(__file__).parents[2] / "requirements-server.txt",
)


def _exact_pin(path: Path, package_name: str) -> Version:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-r"):
            continue
        requirement = Requirement(line)
        if requirement.name.lower() != package_name.lower():
            continue
        pins = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator == "=="
        ]
        if len(pins) == 1:
            return Version(pins[0])
    raise AssertionError(f"{path} does not exactly pin {package_name}")


def test_numpy_pin_satisfies_provider_calendar_policy():
    for requirement_file in REQUIREMENT_FILES:
        assert _exact_pin(requirement_file, "exchange-calendars") == Version("4.11.1")
        assert _exact_pin(requirement_file, "numpy") >= Version("1.26.4")


def test_runtime_lock_matches_runtime_requirements():
    backend = Path(__file__).parents[2]
    locked_versions = {}
    locked_direct = set()
    lock_text = (backend / "requirements-runtime.lock").read_text(encoding="utf-8")
    for block in re.split(r"^(?=[A-Za-z0-9])", lock_text, flags=re.MULTILINE):
        match = re.match(r"([A-Za-z0-9_.-]+)==([^\s;]+)", block)
        if match:
            name = canonicalize_name(match[1])
            locked_versions.setdefault(name, []).append(match[2])
            if "-r backend/requirements-runtime.txt" in block:
                locked_direct.add(name)

    requirements = {}
    for raw_line in (backend / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            requirements[canonicalize_name(requirement.name)] = requirement

    assert locked_direct == set(requirements), "run make lock-runtime"
    for name, requirement in requirements.items():
        for version in locked_versions[name]:
            assert requirement.specifier.contains(version), (
                f"{requirement} does not allow locked {version}; run make lock-runtime"
            )
