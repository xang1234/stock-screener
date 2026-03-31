# Release-Readiness Checklist

Pre-merge and pre-release verification for the Setup Engine.

---

## 1. Automated Gates (CI)

All gates run automatically on PR via `.github/workflows/ci.yml`.
Locally: `make gates` or `make all`.

- [ ] **Gate 1 — Detector Correctness**
  - Interface contracts honored (`test_detector_interface_contract`)
  - Subtask detectors produce expected results (`test_detector_subtasks_c3a_c4a_c6a`)
  - Fixture-driven detector validation (`test_detector_fixtures_se_g1`)
  - Setup engine contract (`test_setup_engine_contract`)
  - Report schema validates (`test_setup_engine_report_schema`)
  - Screener integration (`test_setup_engine_screener`)
  - Parameter handling (`test_setup_engine_parameters`)
  - Aggregator pipeline determinism (`test_aggregator_execution_pipeline`)

- [ ] **Gate 2 — Temporal Integrity**
  - No future-data leakage (`test_temporal_integrity_no_lookahead`)
  - Data sufficiency policies enforced (`test_setup_engine_data_policy`)
  - Score traces auditable (`test_setup_engine_score_trace`)
  - Readiness checks pass (`test_setup_engine_readiness`)

- [ ] **Gate 3 — Integration Coverage**
  - Round-trip persistence (`test_setup_engine_persistence`)
  - Feature flags gate correctly (`test_setup_engine_feature_flag`)
  - Query pipeline round-trip (`test_setup_engine_query_integration`)
  - Backfill script correctness (`test_backfill_setup_engine`)
  - Legacy/feature-store parity (`test_scan_parity`, `test_scan_path_parity`)

- [ ] **Gate 4 — Performance Baselines** *(advisory)*
  - Detector budget: < 150 ms each
  - Aggregator budget: < 500 ms
  - Scanner budget: < 1000 ms
  - Note: timing tests may vary on CI; review for structural regressions

- [ ] **Gate 5 — Golden Regression**
  - Detector snapshots match (`test_golden_detectors`)
  - Aggregator snapshots match (`test_golden_aggregator`)
  - Scanner snapshots match (`test_golden_scanner`)

- [ ] **Gate Check** — all SE test files assigned to a gate (`make gate-check`)

---

## 2. Manual Checks (Pre-Merge)

Reviewer due diligence beyond automated gates:

- [ ] If golden snapshots were regenerated (`make golden-update`), review the diff in
      `backend/tests/unit/golden/snapshots/` — confirm changes are intentional
- [ ] If Gate 4 shows advisory failure, review timing numbers —
      a consistent 2x+ regression warrants investigation
- [ ] If new detector added: verify it appears in gate-check glob patterns
      and is assigned to Gate 1
- [ ] If new fields added to report schema: verify `test_setup_engine_report_schema`
      covers them
- [ ] If schema version bumped: verify golden snapshots updated accordingly

---

## 3. Pre-Release (Deployment)

- [ ] Docker build succeeds: `docker-compose build`
- [ ] Database migrations applied (if any): check `backend/migrations/`
- [ ] Feature flag `SETUP_ENGINE_ENABLED` tested in both states (on/off)
- [ ] Spot-check: run a scan with Setup Engine enabled, verify results appear
      in the UI and match expected detector outputs
- [ ] Frontend lint and tests pass: `make frontend`
- [ ] Documentation review: verify `docs/INSTALL_*.md` match any compose or build changes
