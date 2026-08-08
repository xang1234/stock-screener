import pytest

from app.services.options_metrics import (
    aggregate_by_strike,
    compute_key_gamma_levels,
    compute_ivr,
    compute_options_metrics,
    compute_skew,
    find_current_atm_iv_from_chain,
)


def test_compute_key_gamma_levels_uses_call_put_walls_and_zero_gamma_crossing():
    strike_agg = {
        40.0: {"call_gex": 100.0, "put_gex": -80.0, "total_gex": -80.0},
        45.0: {"call_gex": 50.0, "put_gex": -100.0, "total_gex": 100.0},
        50.0: {"call_gex": 10.0, "put_gex": -20.0, "total_gex": 20.0},
    }

    levels = compute_key_gamma_levels(strike_agg)

    assert levels["call_wall"] == 40.0
    assert levels["put_wall"] == 45.0
    assert levels["zero_gamma"] == pytest.approx(44.0)


def test_compute_ivr_returns_none_when_historical_data_missing_or_invalid():
    assert compute_ivr(0.35, None, None) is None
    assert compute_ivr(0.35, 0.20, None) is None
    assert compute_ivr(0.35, 0.30, 0.25) is None


def test_find_current_atm_iv_from_chain_skips_stale_near_zero_iv():
    options_chain = [
        {"strike": 100.0, "type": "call", "iv": 0.0},
        {"strike": 105.0, "type": "call", "iv": 0.28},
        {"strike": 100.0, "type": "put", "iv": 0.30},
    ]

    # Nearest call strike (100) has a degenerate 0 IV, so it should walk
    # outward to the next usable call strike (105) rather than falling
    # through to puts.
    assert find_current_atm_iv_from_chain(options_chain, underlying_price=100.0) == pytest.approx(0.28)


def test_find_current_atm_iv_from_chain_falls_back_to_puts_when_no_valid_calls():
    options_chain = [
        {"strike": 100.0, "type": "call", "iv": 0.0},
        {"strike": 100.0, "type": "put", "iv": 0.32},
    ]

    assert find_current_atm_iv_from_chain(options_chain, underlying_price=100.0) == pytest.approx(0.32)


def test_find_current_atm_iv_from_chain_returns_none_when_no_usable_iv():
    options_chain = [
        {"strike": 100.0, "type": "call", "iv": 0.0},
        {"strike": 100.0, "type": "put", "iv": None},
    ]

    assert find_current_atm_iv_from_chain(options_chain, underlying_price=100.0) is None


def test_compute_options_metrics_populates_ivr_when_current_iv_and_range_provided():
    # Regression test: the nightly batch cache path previously called
    # compute_options_metrics() without current_iv/iv_52w_low/iv_52w_high,
    # so the cached payload always had ivr=None regardless of accumulated
    # IV history. Confirm ivr is populated end-to-end when they ARE passed.
    options_chain = [
        {"strike": 100.0, "type": "call", "gamma": 0.05, "open_interest": 10, "delta": 0.4, "iv": 0.30},
        {"strike": 100.0, "type": "put", "gamma": 0.05, "open_interest": 5, "delta": -0.4, "iv": 0.32},
    ]

    result = compute_options_metrics(options_chain, spot=100.0, current_iv=0.30, iv_52w_low=0.20, iv_52w_high=0.40)

    assert result["ivr"] == pytest.approx(50.0)


def test_compute_aggregate_skew_skips_invalid_ivs():
    options_chain = [
        {"strike": 100.0, "type": "call", "iv": 0.30, "delta": 0.25},
        {"strike": 100.0, "type": "put", "iv": 0.0, "delta": -0.25},
        {"strike": 105.0, "type": "put", "iv": 0.35, "delta": -0.25},
    ]

    assert compute_skew(options_chain, target_delta=0.25) == pytest.approx(0.05)


def test_aggregate_by_strike_assigns_negative_put_gex():
    options_chain = [
        {"strike": 100.0, "type": "call", "gamma": 0.05, "open_interest": 10, "delta": 0.4, "iv": 0.3},
        {"strike": 100.0, "type": "put", "gamma": 0.05, "open_interest": 5, "delta": -0.4, "iv": 0.32},
    ]

    spot = 100.0
    strike_agg = aggregate_by_strike(options_chain, spot)
    # Dollar gamma exposure per 1% underlying move: gamma * oi * 100 * spot^2 * 0.01
    # -- matches gex_batch.py's convention (see options_metrics.py::aggregate_by_strike).
    call_gex = 0.05 * 10 * 100 * spot * spot * 0.01
    put_gex = 0.05 * 5 * 100 * spot * spot * 0.01
    assert strike_agg[100.0]["call_gex"] == pytest.approx(call_gex)
    assert strike_agg[100.0]["put_gex"] == pytest.approx(-put_gex)
    assert strike_agg[100.0]["total_gex"] == pytest.approx(call_gex - put_gex)


def test_compute_key_gamma_levels_returns_none_zero_gamma_when_cumulative_starts_at_zero():
    strike_agg = {
        65.0: {"call_gex": 100.0, "put_gex": 0.0, "total_gex": 0.0},
        70.0: {"call_gex": 50.0, "put_gex": -10.0, "total_gex": 100.0},
    }

    levels = compute_key_gamma_levels(strike_agg)

    assert levels["call_wall"] == 65.0
    assert levels["put_wall"] is None
    assert levels["zero_gamma"] is None


def test_compute_key_gamma_levels_skips_zero_exposure_walls():
    strike_agg = {
        65.0: {"call_gex": 0.0, "put_gex": 0.0, "total_gex": 0.0},
        70.0: {"call_gex": 0.0, "put_gex": 0.0, "total_gex": 0.0},
    }

    levels = compute_key_gamma_levels(strike_agg)

    assert levels["call_wall"] is None
    assert levels["put_wall"] is None
    assert levels["zero_gamma"] is None
