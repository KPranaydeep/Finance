from contextlib import nullcontext
from copy import deepcopy
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import risk_repair as repair
from allocation_integrity import execution_check
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY


def evidence():
    rng = np.random.default_rng(42)
    history = pd.DataFrame(rng.normal(.0006,.012,(300,6)), index=pd.bdate_range("2025-01-01",periods=300),columns=list("ABCDEF"))
    held = pd.DataFrame({"Yahoo Ticker":list("ABCDEF"),"Symbol":list("ABCDEF"),"Currency":"USD",
                         "Quantity":[7.1]+[.58]*5,"Average Price":10.})
    policy = dict(DEFAULT_ALLOCATION_POLICY,adaptive_enabled=False,single_name_loss_budget=.1,scenario_paths=100)
    return held, history, policy


def run(**kwargs):
    held, history, policy = evidence()
    return repair.prepare_risk_repair(held, history,{t:10. for t in history},{"USD":1.},policy,**kwargs)


def test_minimum_tradable_sales_respect_post_fee_cap_and_keep_cash_not_old_targets():
    held, history, policy = evidence()
    saved = held.copy(deep=True)
    result = run()
    assert result["status"] == "checked_draft", result
    assert result["validation"]["passed"]
    assert [(r["Yahoo Ticker"],r["Executable Quantity"]) for r in result["orders"]] == [("A",5.2)]
    assert all(row["Action"] == "Sell" for row in result["orders"])
    assert result["cash_after_costs_inr"] == pytest.approx(52*(1-.0022))
    assert sum(result["validation"]["final_weights"].values()) < 1
    assert result["validation"]["allocation_policy"]["cvar"] <= result["limits"]["tail_limit"] + 1e-7
    # Independent replay proves the next smaller lot is insufficient after fees.
    smaller = deepcopy(result["orders"])
    smaller[0].update({"Executable Quantity":5.1,"Executable Value":51.})
    checked = execution_check(held.assign(**{"FX to INR":1.}),smaller,{t:10. for t in history},history,
                              cost_bps=22.,policy_limits=result["limits"])
    assert not checked["passed"] and any("shock budget" in v for v in checked["violations"])
    pd.testing.assert_frame_equal(held, saved)


@pytest.mark.parametrize("restriction",["paused","frozen"])
def test_restricted_overweight_blocks_complete_repair_without_releasing_other_orders(restriction):
    result = run(**{restriction:["A"]})
    assert result["status"] == "blocked" and not result["orders"]
    assert result["blocked_tickers"] == ["A"]
    assert result["rows"][0]["After repair (%)"] is None
    assert result["before"]["allocation_policy"]["position_breaches"]


def test_unrestricted_repair_never_changes_a_paused_non_overweight_quantity():
    result = run(paused=["B"])
    assert result["status"] == "checked_draft"
    assert all(row["Yahoo Ticker"] != "B" for row in result["orders"])


def test_whole_inr_lots_and_funded_costs_are_validated():
    held, history, policy = evidence()
    held["Currency"] = "INR"
    held["Quantity"] = [71.]+[5.8]*5
    result = repair.prepare_risk_repair(held,history,{t:10. for t in history},{"INR":1.},policy)
    assert result["status"] == "checked_draft", result
    assert all(float(r["Executable Quantity"]).is_integer() for r in result["orders"])
    assert result["validation"]["cash_after_costs_inr"] >= 0


def test_repair_uses_requested_common_ceiling_not_the_old_individual_caps():
    held,history,policy=evidence()
    policy["common_cap_enabled"]=True
    result=repair.prepare_risk_repair(held,history,{t:10. for t in history},{"USD":1.},policy)
    assert result["status"] == "checked_draft", result
    assert all(row["Cap (%)"] == pytest.approx(24.) for row in result["rows"])
    assert result["orders"][0]["Executable Quantity"] == 4.8


def test_adaptive_and_raw_tail_guards_and_group_constraints_remain_active():
    held, history, policy = evidence()
    policy.update(adaptive_enabled=True,group_weight_limit=.15)
    history["B"] = history["A"]
    result = repair.prepare_risk_repair(held,history,{t:10. for t in history},{"USD":1.},policy)
    assert result["status"] == "checked_draft", result
    check = result["validation"]["allocation_policy"]
    assert check["cvar"] <= result["limits"]["tail_limit"] + 1e-7
    assert check["historical_cvar"] <= result["limits"]["tail_limit"] + 1e-7
    assert sum(result["validation"]["final_weights"].get(t,0.) for t in "AB") <= .15+1e-7


def test_already_compliant_holdings_do_not_call_solver(monkeypatch):
    held, history, policy = evidence()
    held["Quantity"] = 1.
    monkeypatch.setattr(repair,"milp",lambda *a,**k:pytest.fail("unnecessary solve"))
    result = repair.prepare_risk_repair(held,history,{t:10. for t in history},{"USD":1.},policy)
    assert result["status"] == "no_repair_needed" and result["validation"]["passed"]


@pytest.mark.parametrize("case",["history","quote","fx","negative","duplicate"])
def test_missing_or_invalid_evidence_cannot_produce_a_draft(case):
    held, history, policy = evidence()
    prices, fx = {t:10. for t in history},{"USD":1.}
    if case == "history": history = history.drop(columns="A")
    if case == "quote": prices.pop("A")
    if case == "fx": fx.clear()
    if case == "negative": held.loc[0,"Quantity"] = -1.
    if case == "duplicate": held = pd.concat([held,held.iloc[:1]])
    result = repair.prepare_risk_repair(held,history,prices,fx,policy)
    assert result["status"] == "unavailable" and not result["orders"]


def test_timeout_never_releases_an_unproven_solver_incumbent(monkeypatch):
    monkeypatch.setattr(repair,"milp",lambda *a,**k:SimpleNamespace(status=1,success=False,message="Time limit",x=None))
    result = run()
    assert result["status"] == "time_limit" and not result["orders"]


def test_historical_drawdown_failure_retains_but_does_not_approve_proposal():
    result = run(max_drawdown=0.)
    assert result["status"] == "withheld" and result["orders"]
    assert not result["validation"]["passed"]
    assert any("drawdown" in v for v in result["validation"]["violations"])


def test_private_storage_preserves_archive_and_rejects_old_completion_after_restart():
    conn = sqlite3.connect(":memory:")
    result = run()
    held, _, policy = evidence()
    context = repair.repair_context(held,[],[],policy,{},"2026-11-10")
    repair.save_repair(conn,"a",result,context)
    assert repair.load_repair(conn,"b") is None
    restored = repair.load_repair(conn,"a")
    assert restored["orders"] == result["orders"] and restored["evidence_hash"]
    failed = {"computed_at":"2099-01-01T00:00:00+00:00","status":"unavailable","reason":"Missing evidence","orders":[]}
    repair.save_repair(conn,"a",failed,context)
    assert repair.load_repair(conn,"a")["status"] == "unavailable"
    assert conn.execute("SELECT count(*) FROM risk_repair_archive WHERE owner='a'").fetchone()[0] == 2
    with pytest.raises(ValueError,match="newer"):
        repair.save_repair(conn,"a",result,context)
    assert repair.repair_context(held,[],[],policy,{},"2026-11-11") != context
    held.loc[0,"Quantity"] += 1
    assert repair.repair_context(held,[],[],policy,{},"2026-11-10") != context
    conn.close()


def test_app_standalone_uses_only_owned_evidence_and_never_the_full_optimizer(monkeypatch):
    import portfolio_optimizer_core as core
    held, history, policy = evidence()
    payload = {"settings":{},"history":{"valid_start":"2025-01-01"}}
    closes = np.exp(history.cumsum())*10.
    conn = sqlite3.connect(":memory:")
    ns = core.prepare_saved_risk_repair.__globals__
    seen = []
    def sync(cache,names,*a,**k):
        seen.extend(names)
        return closes.copy(),None,None
    replacements = {"load_master_holdings":lambda owner:held.copy(),
        "load_latest_analysis":lambda owner:payload,
        "build_current_allocation_from_db":lambda owner:(held.copy(),[]),
        "get_db_connection":lambda:nullcontext(conn),"active_trade_deferrals":lambda *a:pd.DataFrame(),
        "ALLOCATION_POLICY":policy,"sync_market_history":sync,
        "apply_price_integrity_gate":lambda frame,**k:(frame,pd.DataFrame()),
        "verify_owned_history_fresh":lambda frame,owned,now:{t:{"expected":str(history.index[-1].date()),"observed":str(history.index[-1].date())} for t in history},
        "convert_price_history_to_inr":lambda frame,*a:frame,
        "get_latest_price_map":lambda names:{t:10. for t in names},"get_fx_to_inr_map":lambda *a:{"USD":1.},
        "_app_run_log":lambda *a,**k:None,
        "run_portfolio_analysis_multi":lambda *a,**k:pytest.fail("optimizer called"),
        "refresh_security_scenarios":lambda *a,**k:pytest.fail("ranking called")}
    for key,value in replacements.items(): monkeypatch.setitem(ns,key,value)
    original = deepcopy(payload)
    result = core.prepare_saved_risk_repair(payload,"synthetic")
    assert result["status"] == "checked_draft", result
    assert set(seen) == set(held["Yahoo Ticker"]) and payload == original
    assert repair.load_repair(conn,"synthetic")["orders"] == result["orders"]
    monkeypatch.setitem(ns,"get_latest_price_map",lambda *a:{})
    failed = core.prepare_saved_risk_repair(payload,"synthetic")
    assert failed["status"] == "unavailable"
    assert repair.load_repair(conn,"synthetic")["status"] == "unavailable"
    conn.close()


def test_stale_owned_history_is_rejected_against_listing_sessions():
    prices=pd.DataFrame({"A":[10.,11.]},index=pd.to_datetime(["2026-11-05","2026-11-06"]))
    held=pd.DataFrame({"Yahoo Ticker":["A"],"Exchange":["NMS"]})
    evidence=repair.verify_owned_history_fresh(prices,held,pd.Timestamp("2026-11-07T10:00:00Z"))
    assert evidence["A"]["expected"] == "2026-11-06"
    with pytest.raises(ValueError,match="Stale"):
        repair.verify_owned_history_fresh(prices.iloc[:1],held,pd.Timestamp("2026-11-07T10:00:00Z"))


def test_ui_only_calculates_on_click_and_hides_export_after_context_change():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("tests/risk_repair_ui_fixture.py",default_timeout=10).run()
    assert not app.exception and not list(app.dataframe)
    app.button[0].click().run()
    assert not app.exception and app.dataframe[0].value.iloc[0]["Sell quantity"] == 5.2
    assert any("passed its recorded checks" in x.value for x in app.success)
    assert len(app.get("download_button")) == 1
    app.checkbox[0].check().run()
    assert not app.exception
    assert any("Snapshot needs refresh" in x.value for x in app.warning)
    assert len(app.get("download_button")) == 0
    # Unchanged archived quantities remain visible, never silently re-optimized.
    assert app.dataframe[0].value.iloc[0]["Sell quantity"] == 5.2
