import numpy as np
import pandas as pd
import pytest

from allocation_integrity import allocation_check, execution_check, finalize_weights, fund_buy_orders
import portfolio_optimizer_core as core


def history():
    return pd.DataFrame({"A": [.01, -.02, .01], "B": [-.01, .01, .02], "C": [.005, -.01, .005]})


def holdings():
    return pd.DataFrame([{"Yahoo Ticker": t, "Quantity": q, "Currency": "USD", "FX to INR": 1}
                         for t, q in [("A", 4.), ("B", 3.5), ("C", 3.)]])


def order(ticker="A", action="Sell", quantity=1., value=10.):
    return {"Yahoo Ticker": ticker, "Action": action, "Executable Quantity": quantity,
            "Executable Value": value, "Native Price": 10., "FX to INR": 1., "Currency": "USD"}


def check(orders, **kwargs):
    return execution_check(holdings(), orders, {t: 10. for t in "ABC"}, history(),
                           cost_bps=22., max_weight=.5, **kwargs)


def test_cleanup_cannot_push_half_weight_over_cap():
    raw = np.array([.5, .495, .005])
    cleaned = core.enforce_min_weight_postprocess(raw)
    assert cleaned[0] > .5
    weights, report = finalize_weights(raw, cleaned, history(), bounds=[(0., .5)] * 3)
    assert np.array_equal(weights, raw)
    assert report["passed"] and report["weights_used"] == "raw"
    assert report["cleanup_skipped"]


def test_one_asset_universe_does_not_silently_relax_configured_cap():
    bounds = core.weight_bounds(1, ["A"])
    assert bounds[0][1] == .5
    with pytest.raises(ValueError):
        core.feasible_initial_weights(bounds)


def test_initial_loss_counts_against_drawdown_and_solver_fallback_is_rejected():
    returns = pd.DataFrame({"A": np.log([.7, 1., 1.]), "B": np.zeros(3)})
    report = allocation_check([1., 0.], returns, max_drawdown=.2)
    assert report["historical_drawdown"] == pytest.approx(.3)
    assert not report["passed"]
    selected, report = finalize_weights([1., 0.], [1., 0.], returns, max_drawdown=.2)
    assert selected is None


def test_cleanup_that_breaks_volatility_or_frozen_weight_uses_raw():
    raw = [.2, .795, .005]
    cleaned = [.2, .8, 0.]
    weights, report = finalize_weights(raw, cleaned, history(),
                                      bounds=[(.2, .2), (0, 1), (0, 1)],
                                      target_volatility=.797, covariance=np.diag([0., 1., 0.]))
    assert report["passed"] and report["weights_used"] == "raw"
    assert weights[0] == .2
    assert not allocation_check([.21, .79, 0], history(), bounds=[(.2,.2),(0,1),(0,1)])["passed"]


@pytest.mark.parametrize("raw", [[.2,.2,.2], [float('nan'),.5,.5], [-.1,.5,.6]])
def test_invalid_raw_allocation_cannot_be_normalized_into_success(raw):
    weights, _ = finalize_weights(raw, [.4,.3,.3], history(), bounds=[(0,.5)] * 3)
    assert weights is None


def test_funded_sale_then_buy_retains_positive_cash_after_fees():
    report = check([order(), order("B", "Buy", .9, 9.)])
    assert report["passed"]
    assert report["cash_after_costs_inr"] == pytest.approx(.9582)
    assert sum(report["final_weights"].values()) < 1


@pytest.mark.parametrize("orders,reason", [
    ([order("B", "Buy")], "unrecorded cash"),
    ([order(quantity=5, value=50)], "more than owned"),
    ([order(value=999)], "does not match"),
    ([order(), order()], "Duplicate"),
])
def test_invalid_orders_fail_closed(orders, reason):
    report = check(orders)
    assert not report["passed"]
    assert any(reason in violation for violation in report["violations"])


def test_pause_and_frozen_constraints_are_checked_after_orders():
    assert not check([order()], excluded_tickers=["a"])["passed"]
    assert not check([order()], frozen_weights={"A": .4})["passed"]
    assert check([], frozen_weights={"A": .4})["passed"]


def test_unmodeled_owned_position_is_not_renormalized_away():
    held = holdings()
    held.loc[len(held)] = {"Yahoo Ticker": "UNKNOWN", "Quantity": 1., "Currency": "USD", "FX to INR": 1.}
    result = execution_check(held, [], {t:10. for t in ["A","B","C","UNKNOWN"]}, history(), cost_bps=22.)
    assert not result["passed"] and any("UNKNOWN" in e for e in result["violations"])


def test_funding_preserves_fractional_steps_and_never_invents_cash():
    held = holdings()
    held.loc[held["Yahoo Ticker"].eq("B"), "Quantity"] = 3.5
    orders = pd.DataFrame([order(), order("B", "Buy", 1., 10.)])
    funded, adjustments = fund_buy_orders(orders, held, cost_bps=22.)
    assert funded.loc[1,"Executable Quantity"] == pytest.approx(.9)
    assert adjustments[0]["requested_quantity"] == 1.
    no_sale, _ = fund_buy_orders(orders.iloc[1:], held, cost_bps=22.)
    assert no_sale.empty


def test_missing_quote_does_not_redistribute_target_into_remaining_security():
    rows = holdings().assign(Symbol=["A","B","C"], Weight=[.4,.3,.3], **{"Latest Price INR": 10.})
    plan, missing, _ = core.rebalance_plan_multi(rows, [.4,.3,.3], history(), {"A":10.,"B":10.},1)
    assert plan.empty and missing == ["C"]
    assert not plan.attrs["execution_validation"]["passed"]


def test_complete_plan_includes_rounded_final_risk_and_persists_withheld_rows():
    rows = holdings().assign(Symbol=["A","B","C"], Weight=[.4,.3,.3], **{"Latest Price INR":10.})
    plan, _, _ = core.rebalance_plan_multi(rows, [0.,1.,0.], history(), {t:10. for t in "ABC"},1)
    assert plan.empty
    assert plan.attrs["proposed_rebalancing_plan"]
    assert not plan.attrs["execution_validation"]["passed"]


def test_prepared_solver_fallback_cannot_bypass_drawdown_policy(monkeypatch):
    ns = core.optimize_prepared_return_matrix.__globals__
    monkeypatch.setitem(ns, "optimize_portfolio_max_return_given_daily_risk", lambda *a, **k: None)
    monkeypatch.setitem(ns, "optimize_max_sharpe_ratio", lambda *a, **k: np.array([.5,.5]))
    lr = pd.DataFrame({"A":np.log([.5,1.,1.]), "B":np.log([.5,1.,1.])})
    allocation = pd.DataFrame({"Yahoo Ticker":["A","B"], "Quantity":[1,1], "Weight":[.5,.5]})
    weights, _, _, _, screen = core.optimize_prepared_return_matrix(
        lr, allocation, maximum_assets=2, max_dd=.2, target_volatility=None)
    assert weights is None
    assert not screen["allocation_validation"]["passed"]
    assert "drawdown" in " ".join(screen["allocation_validation"]["violations"])
