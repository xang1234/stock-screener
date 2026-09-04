from pathlib import Path

import pytest

from app.services.atomic_directory_publisher import AtomicDirectoryPublisher


def test_atomic_publisher_preserves_destination_when_population_fails(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "value.txt").write_text("previous", encoding="utf-8")

    def fail(stage: Path) -> None:
        (stage / "value.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("population failed")

    with pytest.raises(RuntimeError, match="population failed"):
        AtomicDirectoryPublisher().publish(destination, fail)

    assert (destination / "value.txt").read_text(encoding="utf-8") == "previous"


def test_atomic_publisher_can_seed_stage_from_existing_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "kept.txt").write_text("kept", encoding="utf-8")

    AtomicDirectoryPublisher().publish(
        destination,
        lambda stage: (stage / "added.txt").write_text("added", encoding="utf-8"),
        clean=False,
    )

    assert (destination / "kept.txt").read_text(encoding="utf-8") == "kept"
    assert (destination / "added.txt").read_text(encoding="utf-8") == "added"
