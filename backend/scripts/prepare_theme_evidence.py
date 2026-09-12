#!/usr/bin/env python3
"""Prepare and review benchmark evidence independently from production ingestion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.theme_evaluation.preparation_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
