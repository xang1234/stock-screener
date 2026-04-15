"""Governance services package (asia.11.x).

Public surface:
- ``launch_gates``: gate definitions, runner, and signed-artifact emission
  for the ASIA v2 launch-gate charter (G1-G9).
"""

from .launch_gates import (  # noqa: F401
    GateResult,
    GateStatus,
    LaunchGateReport,
    run_all_gates,
)
