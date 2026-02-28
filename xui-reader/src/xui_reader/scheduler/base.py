"""Scheduler interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WatchBudget:
    max_cycles: int = 0
    max_runtime_seconds: int = 0


class Scheduler(Protocol):
    def run_once(self) -> int:
        """Run one read cycle and return processed source count."""

    def run_watch(self, budget: WatchBudget) -> int:
        """Run watch mode under a budget and return cycles completed."""
