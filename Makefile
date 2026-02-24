# ═══════════════════════════════════════════════════════════════════
# Setup Engine Quality Gates
#
# Quality gates ordered by diagnostic severity — if Gate 1 fails,
# later gates are less meaningful.
#
#   make help          Show all targets
#   make gates         Run all 5 gates
#   make all           Full CI (gate-check + identity + backend + frontend)
# ═══════════════════════════════════════════════════════════════════

.PHONY: help gate-identity gate-1 gate-2 gate-3 gate-4 gate-5 gates gate-check \
        frontend-lint frontend-test frontend golden-update all

# ── Tooling ─────────────────────────────────────────────────────────

# Use venv python if available (local dev), fall back to system python (CI)
PYTHON = $(shell [ -x backend/venv/bin/python ] && echo ./venv/bin/python || echo python)
PYTEST = cd backend && $(PYTHON) -m pytest
NVM_ACTIVATE = export NVM_DIR="$$HOME/.nvm" && [ -s "$$NVM_DIR/nvm.sh" ] && . "$$NVM_DIR/nvm.sh" || true

# ── Gate 1: Detector Correctness ────────────────────────────────────
# Detectors produce correct outputs, contracts are honored, schemas validate.

GATE_1 = \
  tests/unit/test_detector_interface_contract.py \
  tests/unit/test_detector_subtasks_c3a_c4a_c6a.py \
  tests/unit/test_detector_fixtures_se_g1.py \
  tests/unit/test_setup_engine_contract.py \
  tests/unit/test_setup_engine_report_schema.py \
  tests/unit/test_setup_engine_screener.py \
  tests/unit/test_setup_engine_parameters.py \
  tests/unit/test_aggregator_execution_pipeline.py

# ── Identity Invariants (Fail Fast) ──────────────────────────────────
# Canonical-key uniqueness, alias-key integrity, and identity contracts.

GATE_IDENTITY = \
  tests/unit/test_theme_identity_invariants_ci.py

# ── Gate 2: Temporal Integrity ──────────────────────────────────────
# No future-data leakage, data policies enforce sufficiency.

GATE_2 = \
  tests/unit/test_temporal_integrity_no_lookahead.py \
  tests/unit/test_setup_engine_data_policy.py \
  tests/unit/test_setup_engine_score_trace.py \
  tests/unit/test_setup_engine_readiness.py

# ── Gate 3: Integration Coverage ────────────────────────────────────
# Round-trip persistence, feature flags, query pipeline, path parity.

GATE_3 = \
  tests/unit/test_setup_engine_persistence.py \
  tests/unit/test_setup_engine_feature_flag.py \
  tests/integration/test_setup_engine_query_integration.py \
  tests/unit/test_backfill_setup_engine.py \
  tests/parity/test_scan_parity.py \
  tests/unit/test_scan_path_parity.py

# ── Gate 4: Performance Baselines ───────────────────────────────────
# Runtime budget regression (advisory — won't block CI).

GATE_4 = \
  tests/performance/test_setup_engine_performance.py

# ── Gate 5: Golden Regression ───────────────────────────────────────
# Snapshot-pinned detector, aggregator, and scanner outputs.

GATE_5 = \
  tests/unit/golden/test_golden_detectors.py \
  tests/unit/golden/test_golden_aggregator.py \
  tests/unit/golden/test_golden_scanner.py

# All gate files (used by gate-check)
ALL_GATE_FILES = $(GATE_1) $(GATE_2) $(GATE_3) $(GATE_4) $(GATE_5)


# ═══════════════════════════════════════════════════════════════════
# Targets
# ═══════════════════════════════════════════════════════════════════

help: ## Show available targets
	@echo "Setup Engine Quality Gates"
	@echo "══════════════════════════════════════════════════════════"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Backend Gates ───────────────────────────────────────────────────

gate-1: ## Detector correctness
	$(PYTEST) $(GATE_1) -v --tb=short

gate-identity: ## Theme identity invariants
	$(PYTEST) $(GATE_IDENTITY) -v --tb=short

gate-2: ## Temporal integrity
	$(PYTEST) $(GATE_2) -v --tb=short

gate-3: ## Integration coverage
	$(PYTEST) $(GATE_3) -v --tb=short

gate-4: ## Performance baselines (advisory)
	$(PYTEST) $(GATE_4) -v --tb=short

gate-5: ## Golden regression
	$(PYTEST) $(GATE_5) -v --tb=short

gates: ## Run all 5 gates sequentially (gate-4 advisory)
	$(MAKE) gate-1
	$(MAKE) gate-2
	$(MAKE) gate-3
	-$(MAKE) gate-4
	$(MAKE) gate-5

gate-check: ## Verify all SE test files are assigned to a gate
	@echo "Checking that all SE test files are assigned to a gate..."
	@FAIL=0; \
	for f in $$(find backend/tests -type f -name '*.py' \( \
	  -name 'test_*setup_engine*.py' -o \
	  -name 'test_*detector*.py' -o \
	  -name 'test_*backfill_setup*.py' -o \
	  -name 'test_golden_*.py' -o \
	  -name 'test_*temporal*.py' -o \
	  -name 'test_*parity*.py' -o \
	  -name 'test_*aggregator*.py' \
	\) | sed 's|^backend/||' | sort); do \
	  FOUND=0; \
	  for g in $(ALL_GATE_FILES); do \
	    if [ "$$f" = "$$g" ]; then FOUND=1; break; fi; \
	  done; \
	  if [ $$FOUND -eq 0 ]; then \
	    if [ $$FAIL -eq 0 ]; then echo "ERROR: Unassigned SE test files:"; FAIL=1; fi; \
	    echo "  $$f"; \
	  fi; \
	done; \
	if [ $$FAIL -eq 1 ]; then exit 1; fi; \
	echo "All SE test files are assigned to a gate."

# ── Frontend ────────────────────────────────────────────────────────

frontend-lint: ## Lint frontend code
	@$(NVM_ACTIVATE) && cd frontend && npm run lint

frontend-test: ## Run frontend tests
	@$(NVM_ACTIVATE) && cd frontend && npm run test:run

frontend: frontend-lint frontend-test ## Frontend lint + test

# ── Utilities ───────────────────────────────────────────────────────

golden-update: ## Regenerate golden snapshots for review
	$(PYTEST) tests/unit/golden/ -v --tb=short --golden-update

all: gate-check gate-identity gates frontend ## Full CI (gate-check + identity + backend gates + frontend)
