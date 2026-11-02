import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest

from public_execution_scope import select_execution_target
from public_lumpsum_allocator import allocate_public_lumpsum


TARGET = [{"ticker": "INDIA.NS", "target_weight": .4},
          {"ticker": "WORLD.NS", "target_weight": .1},
          {"ticker": "US", "target_weight": .5}]
PRICES = {"INDIA.NS": {"price": 100., "source_currency": "INR"},
          "WORLD.NS": {"price": 200., "source_currency": "INR"},
          "US": {"price": 500., "source_currency": "USD"}}


@pytest.mark.parametrize("mode,tickers,weights", [
    ("INR only", ["INDIA.NS", "WORLD.NS"], [.8, .2]),
    ("Overseas only", ["US"], [1.]),
    ("Global", ["INDIA.NS", "WORLD.NS", "US"], [.4, .1, .5]),
])
def test_scoped_targets_and_orders_preserve_source(mode, tickers, weights):
    original = deepcopy(TARGET)
    target = select_execution_target(TARGET, PRICES, mode)
    assert [row["ticker"] for row in target] == tickers
    assert [row["target_weight"] for row in target] == pytest.approx(weights)
    plan = allocate_public_lumpsum(target, PRICES, 10000.)
    assert {row["ticker"] for row in plan["orders"]} == set(tickers)
    assert plan["invested_inr"] + plan["residual_cash_inr"] == pytest.approx(10000.)
    assert TARGET == original


def test_missing_quote_does_not_change_scope_and_unknown_listing_is_not_guessed():
    assert len(select_execution_target(TARGET, {}, "INR only")) == 2
    assert len(select_execution_target(TARGET, {}, "Overseas only")) == 1
    unknown = [{"ticker": "UNKNOWN.XY", "target_weight": 1.}]
    assert select_execution_target(unknown, {}, "Overseas only") == []
    assert select_execution_target(unknown, {}, "INR only") == []
    assert select_execution_target(unknown, {}, "Global") == unknown
    assert select_execution_target(TARGET[:2], PRICES, "Overseas only") == []


def test_explicit_native_currency_takes_precedence():
    target = [{"ticker": "LONDON.L", "target_weight": 1.}]
    assert select_execution_target(target, {"LONDON.L": {"source_currency": "GBP"}}, "Overseas only") == target
    with pytest.raises(ValueError):
        select_execution_target(TARGET, PRICES, "Unknown mode")
    with pytest.raises(ValueError):
        select_execution_target([{"ticker": "US", "target_weight": float("nan")}], PRICES, "Global")


@pytest.mark.parametrize("mode", ["INR only", "Overseas only", "Global"])
def test_exported_prompt_has_scoped_weights_and_protects_outside_holdings(mode):
    source = Path(__file__).resolve().parents[1] / "public_portfolio_performance.py"
    function = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_execution_plan_prompt")
    namespace = {"json": json, "EXECUTION_SCENARIOS": {"Rebalance existing holdings": "Rebalance"}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    current = dict.fromkeys(("basket_id", "publication_id", "portfolio_fingerprint", "strategy_version",
                             "calculation_version", "as_of", "published_at"), "source")
    current.update(portfolio_version=11, cash_weight=0)
    target = select_execution_target(TARGET, PRICES, mode)
    prompt, payload = namespace["build_execution_plan_prompt"](
        current, target, PRICES, "Rebalance existing holdings", execution_mode=mode)
    assert payload["execution_mode"] == mode
    assert sum(row["target_weight_pct"] for row in payload["target_positions"]) == pytest.approx(100.)
    assert {row["ticker"] for row in payload["target_positions"]} == {row["ticker"] for row in target}
    assert payload["portfolio_fingerprint"] == "source"
    if mode != "Global":
        assert "Leave all outside-scope holdings untouched" in prompt
        assert "A holding outside this scope is not a zero-target exit" in prompt
        assert "rescaled" in payload["target_basis"]


@pytest.mark.parametrize("mode", ["INR only", "Overseas only", "Global"])
def test_real_execution_section_scopes_buy_withdrawal_and_downloads(mode):
    from streamlit.testing.v1 import AppTest

    source = (Path(__file__).resolve().parents[1] / "public_portfolio_performance.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "build_execution_plan_prompt")
    scenarios = next(node for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "EXECUTION_SCENARIOS" for t in node.targets))
    section = source[source.index('st.subheader("Build your private execution plan")'):
                     source.index('st.subheader("Performance')]
    prefix = '''
import json, logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from public_execution_scope import EXECUTION_MODES, select_execution_target
from public_lumpsum_allocator import allocate_public_lumpsum, estimate_minimum_entry_capital
from public_cash_withdrawal import suggested_withdrawal, withdrawal_instructions
from public_withdrawal_allocator import allocate_withdrawal
from public_allocation_card import render_buy_plan_card, listing_descriptor
IST=ZoneInfo("Asia/Kolkata")
LOGGER=logging.getLogger(__name__)
DEFAULT_BASKET_ID="TEST"
share_prompt_button=lambda *a: None
copy_prompt_button=lambda *a: None
current=dict.fromkeys(("basket_id","publication_id","portfolio_fingerprint","strategy_version","calculation_version"),"source")
current.update(portfolio_version=11,cash_weight=0,as_of=datetime(2026,10,30,tzinfo=timezone.utc),published_at=datetime(2026,10,30,tzinfo=timezone.utc))
'''
    prefix += f"record={{'constituents':{TARGET!r}}}\nprice_snapshot={PRICES!r}\n"
    prefix += f"st.session_state.setdefault('private_execution_mode', {mode!r})\n"
    prefix += "entry_estimate=estimate_minimum_entry_capital(record['constituents'],price_snapshot)\n"
    prefix += ast.unparse(scenarios) + "\n" + ast.unparse(function) + "\n"
    suffix = "\nst.session_state['test_target']=execution_constituents\nst.session_state['test_plan']=calculated_plan\n"
    suffix += "if execution_scenario == 'Raise cash from existing holdings':\n    st.session_state['test_sales']=sale_plan\n"
    app = AppTest.from_string(prefix + section + suffix).run(timeout=30)
    assert not app.exception
    expected = {r["ticker"] for r in select_execution_target(TARGET, PRICES, mode)}
    assert {r["ticker"] for r in app.session_state["test_target"]} == expected
    assert {r["ticker"] for r in app.session_state["test_plan"]["orders"]} <= expected
    assert not any("cannot be calculated" in item.value for item in app.info)
    app.selectbox[0].select("Raise cash from existing holdings").run(timeout=30)
    assert not app.exception
    assert {r["Ticker"] for r in app.session_state["test_sales"]["orders"]} <= expected
