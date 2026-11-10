from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from buy_candidate_watchlist import buy_candidate_watchlist


def payload():
    return {"optimal_allocation":[{"ticker":"A","target_weight":.4},{"ticker":"B.NS","target_weight":.3},{"ticker":"C","target_weight":.3}],
        "current_allocation":[{"Yahoo Ticker":"A","Weight":.8,"Currency":"USD"},{"Yahoo Ticker":"B.NS","Weight":.2,"Currency":"INR"},{"Yahoo Ticker":"C","Weight":0.,"Currency":"USD"}],
        "rebalancing_plan":[],"execution_validation":{"passed":False,"violations":["Funding failed"]},
        "security_scenario_ranking":[{"Yahoo Ticker":t,"Expected net scenario return (%)":r,"Probability of positive net return (%)":50.,"Price data through":"2026-11-09"} for t,r in [("A",2.),("B.NS",5.),("C",3.)]]}


def test_blocked_empty_plan_still_lists_all_targets_sorted_by_score_without_cash_assumption():
    p=payload(); before=deepcopy(p)
    frame=buy_candidate_watchlist(p)
    assert frame["Security"].tolist()==["B.NS","C","A"]
    assert frame["Unfunded draft quantity"].isna().all()
    assert frame.set_index("Security").loc["C","Status"]=="Cash unknown · sizing/checks required"
    assert p==before


def test_released_sell_only_plan_does_not_replace_research_buys():
    p=payload()
    p["execution_validation"]={"passed":True,"profit_cap_exception":{"applied":True}}
    p["rebalancing_plan"]=[{"Yahoo Ticker":"A","Action":"Sell","Executable Quantity":3.}]
    p["buy_candidate_orders"]=[{"Yahoo Ticker":"C","Action":"Buy","Executable Quantity":12.,"Optimal Weight":.3}]
    frame=buy_candidate_watchlist(p).set_index("Security")
    assert len(frame)==3 and frame.loc["C","Unfunded draft quantity"]==12.


def test_paused_and_over_cap_candidates_remain_visible_with_restrictions():
    p=payload();p["allocation_policy"]={"limits":{"position_caps":{"A":.5,"B.NS":.25}}}
    frame=buy_candidate_watchlist(p,paused=["C"]).set_index("Security")
    assert "position cap" in frame.loc["A","Status"]
    assert "target exceeds cap" in frame.loc["B.NS","Status"]
    assert "Paused" in frame.loc["C","Status"]


def test_missing_scores_and_prices_remain_unavailable_and_duplicate_tickers_collapse():
    p=payload()
    p["security_scenario_ranking"].append(dict(p["security_scenario_ranking"][0],**{"Expected net scenario return (%)":.1}))
    assert buy_candidate_watchlist(p).set_index("Security").loc["A","Score (%)"]==1.
    p["security_scenario_ranking"]=[]
    frame=buy_candidate_watchlist(p)
    assert frame["Score (%)"].isna().all() and set(frame["Price data through"])=={"Unavailable"}


def test_prefunding_buy_orders_are_retained_when_no_cash_survives_plan_generation():
    import portfolio_optimizer_core as core
    held=pd.DataFrame({"Symbol":["A","B"],"Yahoo Ticker":["A","B"],"Currency":"INR","Quantity":[100.,0.],"FX to INR":1.})
    history=pd.DataFrame({"A":np.full(300,.001),"B":np.full(300,.002)},index=pd.bdate_range("2025-01-01",periods=300))
    # Pausing the sale leaves the desired B buy without any funding.
    plan,_,_=core.rebalance_plan_multi(held,[.5,.5],history,{"A":10.,"B":10.},1,excluded_trade_tickers=["A"])
    assert plan.empty
    assert plan.attrs["buy_candidate_orders"][0]["Yahoo Ticker"]=="B"
    assert plan.attrs["buy_candidate_orders"][0]["Executable Quantity"]>0


def test_render_shows_candidates_even_with_failed_execution_checks():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("tests/fixtures/buy_watchlist_app.py").run()
    assert not app.exception
    assert app.dataframe[0].value["Security"].tolist()==["NEW.NS","EXAMPLE"]
    assert "outside income/cash is unknown" in app.caption[0].value
