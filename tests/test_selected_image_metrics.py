from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from selected_image_metrics import selected_image_statistics
from portfolio_risk_metrics import (
    empirical_expected_shortfall, moving_block_bootstrap_return_distribution,
    bootstrap_tail_summary,
)


def evidence():
    rng=np.random.default_rng(183)
    returns=rng.normal([.001,.0001,-.0002,.0008],[.02,.008,.03,.015],(300,4))
    prices=100*np.exp(np.vstack([np.zeros(4),returns.cumsum(axis=0)]))
    dates=pd.bdate_range('2025-01-01',periods=301)
    history=dict(schema_version=2,basis='INR',columns=['A.NS','B.NS','C','D'],
        dates=[str(d.date()) for d in dates],values=prices.tolist())
    payload=dict(current_allocation=[{'Yahoo Ticker':t,'Weight':w} for t,w in zip(history['columns'],[.4,.3,.2,.1])],
        current_stats={'Annual Return':9},optimal_stats={'Annual Return':8,'Block-Bootstrap ES 95% (30 Sessions)':7},
        portfolio_metrics_config={'risk_free_rate_annual':.08225,'trading_days_per_year':252})
    rows=[(t,'INR' if t.endswith('.NS') else 'USD',w,False) for t,w in zip(history['columns'],[.1,.2,.3,.4])]
    return payload,rows,history,returns


def expected(series):
    annual=series.mean()*252
    vol=series.std(ddof=1)*np.sqrt(252)
    paths=moving_block_bootstrap_return_distribution(series,horizon_sessions=30)
    tail=bootstrap_tail_summary(paths)
    return [f'{annual:.2%}',f'{vol:.2%}',f'{(annual-.08225)/vol:.2f}',
        f'{tail["median_return"]:.2%}',f'{empirical_expected_shortfall(np.expm1(series)):.2%}',
        f'{tail["expected_shortfall"]:.2%}',f'{tail["tail_loss_sd"]:.2%}']


@pytest.mark.parametrize('indices',[[0,1],[2,3],[0,1,2,3]])
def test_selected_current_and_optimal_are_recomputed_on_separate_normalized_weights(indices):
    payload,rows,history,returns=evidence();before=deepcopy((payload,rows,history))
    selected=[rows[i] for i in indices]
    metrics,horizon,details=selected_image_statistics(payload,selected,history)
    current=np.array([.4,.3,.2,.1])[indices];target=np.array([.1,.2,.3,.4])[indices]
    assert [r[1] for r in metrics[:-1]]==expected(returns[:,indices]@(current/current.sum()))
    assert [r[2] for r in metrics[:-1]]==expected(returns[:,indices]@(target/target.sum()))
    assert horizon==30 and details['observations']==300 and not details['error']
    assert (payload,rows,history)==before


def test_unselected_prices_do_not_affect_selection_and_saved_history_is_inr_based():
    payload,rows,history,_=evidence()
    first=selected_image_statistics(payload,rows[:2],history)
    for row in history['values']:row[2:]=[None,None]
    assert selected_image_statistics(payload,rows[:2],history)==first
    history['basis']='USD'
    metrics,_,details=selected_image_statistics(payload,rows[:2],history)
    assert all(r[1:3]==('Unavailable','Unavailable') for r in metrics[:-1])
    assert 'INR' in details['error']


@pytest.mark.parametrize('problem',['missing_history','missing_security','latest_gap','too_few','bad_dates','missing_config'])
def test_incomplete_evidence_never_reuses_whole_portfolio_statistics(problem):
    payload,rows,history,_=evidence()
    if problem=='missing_history':history=None
    if problem=='missing_security':history['columns'][0]='OTHER'
    if problem=='latest_gap':history['values'][-1][0]=None
    if problem=='too_few':history['values']=history['values'][-100:];history['dates']=history['dates'][-100:]
    if problem=='bad_dates':history['dates'][-1]=history['dates'][-2]
    if problem=='missing_config':payload['portfolio_metrics_config']={}
    metrics,_,details=selected_image_statistics(payload,rows[:2],history)
    assert details['error'] and all(r[1:3]==('Unavailable','Unavailable') for r in metrics[:-1])


def test_gap_keeps_earlier_valid_daily_returns_and_zero_current_is_unavailable():
    payload,rows,history,_=evidence()
    history['values'][30][0]=None
    payload['current_allocation']=[]
    metrics,_,details=selected_image_statistics(payload,rows[:2],history)
    assert details['observations']==298 and details['excluded_returns']==2
    assert 'No current allocation' in details['error']
    assert all(r[1]=='Unavailable' for r in metrics[:-1])
    assert metrics[0][2]!='Unavailable'


def test_recent_sparse_gap_does_not_erase_years_of_history_or_bridge_prices():
    payload,rows,history,returns=evidence()
    history['values'][-6][0]=None  # five recent prices alone cannot support the horizon
    metrics,_,details=selected_image_statistics(payload,rows[:2],history)
    assert not details['error'] and details['observations']==298 and details['excluded_returns']==2
    # The two-day change around the missing mark must not masquerade as one day.
    target=np.array([1/3,2/3])
    series=returns[:,:2]@target
    series[-6:-4]=np.nan
    observed=series[np.isfinite(series)]
    assert metrics[0][2]==f'{observed.mean()*252:.2%}'
    paths=moving_block_bootstrap_return_distribution(series,horizon_sessions=30,preserve_gaps=True)
    assert metrics[3][2]==f'{np.median(paths):.2%}'


def test_missing_current_weight_does_not_invent_current_stats_or_discard_optimal():
    payload,rows,history,_=evidence()
    payload['current_allocation'][0].pop('Weight')
    metrics,_,details=selected_image_statistics(payload,rows[:2],history)
    assert 'Current selected weights unavailable'==details['error']
    assert all(r[1]=='Unavailable' for r in metrics[:-1]) and metrics[0][2]!='Unavailable'
