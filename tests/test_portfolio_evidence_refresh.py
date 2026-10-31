"""Exercise the fixed-weight refresh without importing the interactive app."""
import ast
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from us_election_risk import unpack_prices, allocation_inputs
from portfolio_risk_metrics import empirical_expected_shortfall, moving_block_bootstrap_return_distribution, bootstrap_tail_summary


def functions():
    source=ast.parse(Path('portfolio_rebalancer_database.py').read_text(encoding='utf-8'))
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in {'portfolio_stats','refresh_portfolio_evidence'}]
    calls=[]
    namespace=dict(np=np,pd=pd,datetime=datetime,timezone=timezone,allocation_inputs=allocation_inputs,unpack_prices=unpack_prices,
                   TRADING_DAYS_PER_YEAR=252,RISK_FREE_RATE_ANNUAL=.08225,MARKET_HISTORY_CACHE_PATH='unused',
                   empirical_expected_shortfall=empirical_expected_shortfall,
                   moving_block_bootstrap_return_distribution=moving_block_bootstrap_return_distribution,
                   bootstrap_tail_summary=bootstrap_tail_summary)
    def check(returns,*args,**kwargs):
        calls.append(str(returns.index[-1].date()))
        return {'as_of':calls[-1],'horizon_sessions':30}
    namespace.update(assess_executable_rebalance=check,estimate_review_window=check)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<app-refresh>','exec'),namespace)
    return namespace,calls


def test_refresh_updates_dependent_dates_at_fixed_weights():
    ns,calls=functions()
    dates=pd.bdate_range('2025-01-01',periods=180)
    history=dict(schema_version=2,basis='INR',weights={'A':.6,'B':.4},columns=['A','B'],
                 dates=[str(d.date()) for d in dates],values=np.column_stack([np.linspace(100,120,180),np.linspace(100,110,180)]).tolist(),
                 expected_through=str(dates[-1].date()))
    ns['portfolio_prices_from_cache']=lambda *a:history
    p={'optimal_allocation':[{'ticker':'A','target_weight':.6},{'ticker':'B','target_weight':.4}],
       'current_allocation':[{'Yahoo Ticker':'A','Quantity':2},{'Yahoo Ticker':'B','Quantity':1}],
       'optimal_stats':{'old':1},'leverage_assessment':{'old':True}}
    ns['refresh_portfolio_evidence'](p)
    assert calls==[history['expected_through']]*2
    assert p['portfolio_metrics_as_of']==history['expected_through']
    assert len(p['optimal_stats'])==len(p['current_stats'])==7
    assert p['portfolio_evidence_history']['weights']=={'A':.6,'B':.4}
    assert p['leverage_assessment'] is None
    assert len(p['leverage_return_history']['dates'])==179


def test_incomplete_common_tail_preserves_previous_assessments():
    ns,_=functions()
    ns['portfolio_prices_from_cache']=lambda *a:dict(schema_version=2,basis='INR',columns=['A'],dates=['2026-10-29'],values=[[100]],expected_through='2026-10-30')
    p={'optimal_allocation':[{'ticker':'A','target_weight':1.}],'optimal_stats':{'old':1}}
    with pytest.raises(ValueError):ns['refresh_portfolio_evidence'](p)
    assert p['optimal_stats']=={'old':1} and 'portfolio_metrics_as_of' not in p
