"""Reusable atomic publication for generated directory trees."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

ResultT = TypeVar("ResultT")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


class AtomicDirectoryPublisher:
    """Populate, validate, and atomically replace one destination directory."""

    def publish(
        self,
        destination: Path,
        populate: Callable[[Path], ResultT],
        *,
        validate: Callable[[Path], object] | None = None,
        clean: bool = True,
    ) -> ResultT:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.stage-",
                dir=str(target.parent),
            )
        )
        backup = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.previous-",
                dir=str(target.parent),
            )
        )
        backup.rmdir()
        incumbent_moved = False
        published = False
        try:
            if not clean and (target.exists() or target.is_symlink()):
                shutil.copytree(target, stage, dirs_exist_ok=True)
            result = populate(stage)
            if validate is not None:
                validate(stage)

            if target.exists() or target.is_symlink():
                target.rename(backup)
                incumbent_moved = True
            try:
                stage.rename(target)
                published = True
            except Exception:
                if incumbent_moved and backup.exists() and not target.exists():
                    backup.rename(target)
                    incumbent_moved = False
                raise

            if incumbent_moved and backup.exists():
                _remove_path(backup)
                incumbent_moved = False
            return result
        finally:
            if stage.exists() or stage.is_symlink():
                _remove_path(stage)
            # Never delete the only surviving copy of the incumbent if rollback
            # itself failed. A successful publish can safely discard it.
            if backup.exists() and (published or target.exists()):
                _remove_path(backup)


__all__ = ["AtomicDirectoryPublisher"]
