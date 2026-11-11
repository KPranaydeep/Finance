from contextlib import contextmanager
import sqlite3

import numpy as np
import pandas as pd
import pytest

from weight_freeze_overrides import load_weight_freeze_overrides,set_weight_freeze_override,released_price_freezes
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY


def test_override_is_profile_scoped_reversible_and_audited_after_reopen(tmp_path):
    path=tmp_path/'override.sqlite'
    with sqlite3.connect(path) as conn:
        assert load_weight_freeze_overrides(conn,'alice')=={}
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='weight_freeze_override_events'").fetchone()[0]==0
        assert set_weight_freeze_override(conn,'alice',' gvt&d.ns ',enabled=True,reason='Explicit user request')=='GVT&D.NS'
        assert not load_weight_freeze_overrides(conn,'bob')
    with sqlite3.connect(path) as conn:
        active=load_weight_freeze_overrides(conn,'alice')
        assert active['GVT&D.NS']['recorded_at'].endswith('+00:00')
        set_weight_freeze_override(conn,'alice','GVT&D.NS',enabled=False,reason='Restore freeze')
        assert not load_weight_freeze_overrides(conn,'alice')
        assert conn.execute('SELECT COUNT(*) FROM weight_freeze_override_events').fetchone()[0]==2
    with sqlite3.connect(':memory:') as conn:
        with pytest.raises(ValueError):
            set_weight_freeze_override(conn,'alice','A; DROP TABLE x',enabled=True,reason='x')


def test_only_named_quarantined_listing_can_be_released():
    allocation=pd.DataFrame();allocation.attrs['weight_freeze_overrides']=['GVT&D.NS']
    assert released_price_freezes({'GVT&D.NS','GVT&D.BO','STLN'},allocation)=={'GVT&D.NS'}
    assert released_price_freezes({'STLN'},allocation)==set()
    assert released_price_freezes({'GVT&D.NS'},pd.DataFrame())==set()


def test_full_and_prepared_solver_can_reduce_a_released_holding_without_faking_verified_history(monkeypatch):
    import portfolio_optimizer_core as core
    names=['GVT&D.NS','B']+[f'C{i}' for i in range(10)]
    rng=np.random.default_rng(74)
    history=pd.DataFrame(rng.normal(.001,.012,(300,12)),columns=names,index=pd.bdate_range('2025-01-01',periods=300))
    allocation=pd.DataFrame({'Yahoo Ticker':names,'Quantity':[40.,60.]+[0.]*10,'Weight':[.4,.6]+[0.]*10,'Currency':'INR'})
    evidence=pd.DataFrame([{'Ticker':'GVT&D.NS','Owned':True,'Status':'QUARANTINED_UNEXPLAINED_JUMP'},
                           {'Ticker':'B','Owned':True,'Status':'VERIFIED'}])
    policy=dict(DEFAULT_ALLOCATION_POLICY,adaptive_enabled=False,scenario_paths=100,
                single_name_loss_budget=.1,group_weight_limit=.25,solver_time_limit_seconds=5.)
    namespace=core.run_portfolio_analysis_multi.__globals__
    monkeypatch.setitem(namespace,'ALLOCATION_POLICY',policy)
    monkeypatch.setitem(namespace,'get_daily_log_returns',lambda *a,**kw:(history,{'price_integrity_report':evidence.copy()}))
    # The same data is a frozen-cap conflict until explicitly overridden.
    w,_,_,_,meta=core.optimize_prepared_return_matrix(history,allocation,maximum_assets=12,max_dd=.95,integrity_report=evidence)
    assert w is None and meta['frozen_weights']=={'GVT&D.NS':.4}
    allocation.attrs['weight_freeze_overrides']=['GVT&D.NS']
    for prepared in (False,True):
        if prepared:
            w,returns,_,_,meta=core.optimize_prepared_return_matrix(history,allocation,maximum_assets=12,max_dd=.95,integrity_report=evidence)
            assert meta['frozen_weights']=={}
        else:
            w,returns,_,_,meta=core.run_portfolio_analysis_multi(tuple(names),allocation,exact_optimizer_asset_cap=12,max_dd=.95)
            assert meta['price_integrity_frozen_owned_weights']=={}
            assert meta['price_integrity_report'].iloc[0]['Status']=='QUARANTINED_UNEXPLAINED_JUMP'
        assert w is not None and w[returns.columns.get_loc('GVT&D.NS')]<=.2+1e-7
        report=meta['allocation_policy']
        assert report['limits']['position_caps']['GVT&D.NS']==pytest.approx(.2)
        assert report['status']=='conditional_research' and report['research_kind']=='price_freeze_override'
        assert report['strict_allocation_validation']['passed']
        assert not meta['allocation_validation']['passed'] and not report['constraints_applied']
        assert allocation.loc[0,'Weight']==.4
    # A different quarantined holding retains its equality constraint.
    evidence.loc[1,'Status']='QUARANTINED_UNEXPLAINED_JUMP'
    w,returns,_,_,meta=core.optimize_prepared_return_matrix(history,allocation,maximum_assets=12,max_dd=.95,integrity_report=evidence)
    assert w is not None and meta['frozen_weights']=={'B':.6}
    assert w[returns.columns.get_loc('B')]==pytest.approx(.6)
    # Pausing the released name still removes it, including from prepared history.
    from paused_allocation import prepare_pause_allocation
    eligible,_=prepare_pause_allocation(allocation,pd.DataFrame({'yahoo_ticker':['GVT&D.NS']}))
    w,returns,_,_,meta=core.optimize_prepared_return_matrix(history,eligible,maximum_assets=12,max_dd=.95,integrity_report=evidence.iloc[:1])
    assert w is not None and 'GVT&D.NS' not in returns
    assert meta['allocation_policy'].get('research_kind')!='price_freeze_override'


def test_override_controls_persist_without_an_optimizer_or_page_load_write(tmp_path):
    from streamlit.testing.v1 import AppTest
    path=tmp_path/'override-ui.sqlite'
    app=AppTest.from_file('tests/fixtures/weight_freeze_app.py',default_timeout=10)
    app.session_state['db']=str(path);app.run()
    assert not app.exception
    app.text_input[0].set_value('GVT&D.NS')
    app.button[0].click().run()
    assert not app.exception and app.success
    with sqlite3.connect(path) as conn:
        assert 'GVT&D.NS' in load_weight_freeze_overrides(conn,'synthetic')
    app.button[1].click().run()
    with sqlite3.connect(path) as conn:
        assert not load_weight_freeze_overrides(conn,'synthetic')
