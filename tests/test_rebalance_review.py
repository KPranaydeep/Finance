from rebalance_review import METRICS, assess_rebalance


def stats():
    return {name: (2.0 if name == "Sharpe Ratio" else .05) for name in METRICS}


def test_equal_portfolios_do_not_justify_turnover():
    current = stats()
    result = assess_rebalance(current, current, [{"Action": "Buy", "Executable Value": 1000}], 10000)
    assert result["decision"].startswith("Hold candidate")
    assert abs(result["cost_fraction"] - .00022) < 1e-10


def test_improvement_must_exceed_costs():
    current = stats()
    better = {name: value + (.01 if direction > 0 else -.01) for (name, direction), value in zip(METRICS.items(), current.values())}
    result = assess_rebalance(current, better, [{"Action": "Buy", "Executable Value": 5000}, {"Action": "Sell", "Executable Value": 5000}], 10000)
    assert result["decision"].startswith("Rebalance candidate")
    assert result["two_way_turnover"] == .5
    assert abs(result["cost_fraction"] - .0022) < 1e-10
    expensive = assess_rebalance(current, better, [{"Action": "Buy", "Executable Value": 10000}], 10000, cost_bps=200)
    assert expensive["decision"].startswith("Risk-reduction candidate")


def test_missing_cost_or_unknown_metric_requires_review():
    current = stats()
    result = assess_rebalance(current, current, [{"Action": "Buy"}], 10000)
    assert result["cost_fraction"] is None
    assert result["decision"].startswith("Review")
    result = assess_rebalance(dict(current, Unknown=1), dict(current, Unknown=2), [{"Action": "Buy", "Executable Value": 1}], 10000)
    assert result["decision"].startswith("Review")


def test_no_orders_means_hold():
    assert assess_rebalance(stats(), stats(), [], 10000)["decision"].startswith("Hold")
