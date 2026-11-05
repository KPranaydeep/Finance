import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
from market_history_cache import sync_market_history
from us_election_risk import portfolio_prices_from_cache, us_election_reference, latest_joint_session, listing_calendar, next_us_event_review
from us_election_risk import us_event_context, us_federal_events, allocation_inputs


def fixture_history(tmp_path):
    nse=mcal.get_calendar('NSE').valid_days('2020-01-01','2026-10-30').tz_localize(None)
    nyse=mcal.get_calendar('NYSE').valid_days('2020-01-01','2026-10-30').tz_localize(None)
    india=pd.Series(100.,index=nse)
    us=pd.Series(100.,index=nyse)
    us.loc[us.index>='2024-11-06']=80.
    fx=pd.Series(80.,index=pd.bdate_range('2020-01-01','2026-10-30'))
    fx.loc[fx.index>='2024-11-06']=88.
    prices=pd.DataFrame({'INDIA':india,'FOREIGN_ADR':us,'INR=X':fx})
    path=tmp_path/'history.duckdb'
    sync_market_history(path,tuple(prices),'2020-01-01','2026-10-31',lambda *a:(prices,pd.DataFrame(),{}))
    payload={'optimal_allocation':[{'ticker':'INDIA','target_weight':.5},{'ticker':'FOREIGN_ADR','target_weight':.5}],
             'current_allocation':[{'Yahoo Ticker':'INDIA','Currency':'INR','Exchange':'NSI','Quantity':1},
                                   {'Yahoo Ticker':'FOREIGN_ADR','Currency':'USD','Exchange':'NYQ','Quantity':1}]}
    return portfolio_prices_from_cache(payload,path,'2026-10-31')


def test_us_event_whole_mix_fx_and_native_holidays(tmp_path):
    h=fixture_history(tmp_path)
    r=us_election_reference(h,'2026-10-31')
    assert r['active_events'][0]['event_date']=='2026-11-03'
    assert r['event_reserve_fraction']==pytest.approx(.06)
    assert len(r['windows'])==3
    assert r['data_through']=='2026-10-30'
    assert r['coverage_complete']


def test_no_us_history_never_means_zero_risk():
    r=us_election_reference(None,'2026-10-31')
    assert r['event_reserve_fraction'] is None and not r['coverage_complete']
    assert 'not assessed' in r['reason']


def test_stale_common_prices_withhold_us_amount(tmp_path):
    h=fixture_history(tmp_path)
    h['values'][-1][1]=None
    r=us_election_reference(h,'2026-10-31')
    assert r['status']=='stale_or_incomplete_prices'
    assert r['event_reserve_fraction'] is None


def test_simultaneous_events_not_summed_or_called_complete(tmp_path):
    h=fixture_history(tmp_path)
    r=us_election_reference(h,'2026-10-31',[{'verified':True,'counting_date':'2026-11-05'}])
    assert r['event_reserve_fraction']==pytest.approx(.06)
    assert not r['coverage_complete']
    assert 'joint stress not established' in r['reason']


def test_quote_currency_does_not_supply_calendar():
    with pytest.raises(ValueError): listing_calendar({'Currency':'USD','Exchange':'US/Global'})
    assert listing_calendar({'Exchange':'CCC'})=='24/7'
    assert latest_joint_session('2026-10-31')=='2026-10-30'
    assert next_us_event_review('2026-10-31')=='2026-11-02'
    assert next_us_event_review('2026-01-01') is None


def test_midterms_move_to_post_event_watch_then_expire():
    context=us_event_context('2026-11-05')
    assert context['active_events'][0]['phase']=='post_election'
    assert context['active_events'][0]['event_date']=='2026-11-03'
    assert context['active_events'][0]['watch_ends']=='2026-11-22'
    assert context['next_event']['event_date']=='2028-11-07'
    assert next_us_event_review('2026-11-05') is None
    assert us_event_context('2026-11-23')['active_events']==[]
    r=us_election_reference(None,'2026-11-23')
    assert r['status']=='no_active_us_event'
    assert 'covers through 2026' not in r['reason']


def test_future_statutory_dates_do_not_stop_in_2026():
    events=us_federal_events('2031-01-01')
    assert next(e for e in events if e['event_date'].startswith('2032'))['event_date']=='2032-11-02'
    assert us_event_context('2030-10-25')['active_events'][0]['phase']=='upcoming'
    assert all(e['date_basis']=='statutory regular federal schedule' for e in events)


def test_old_snapshot_expected_date_does_not_make_stale_prices_current(tmp_path):
    h=fixture_history(tmp_path)
    assert h['expected_through']=='2026-10-30'
    result=us_election_reference(h,'2026-11-05')
    assert result['status']=='stale_or_incomplete_prices'
    assert result['event_reserve_fraction'] is None


def test_current_optimal_weights_override_cached_election_weights(tmp_path):
    h=fixture_history(tmp_path)
    result=us_election_reference(h,'2026-10-31',current_weights={'INDIA':.8,'FOREIGN_ADR':.2})
    assert result['status']=='stale_or_incomplete_prices'
    assert result['event_reserve_fraction'] is None
    weights,_=allocation_inputs({'election_reserve_history':{'weights':{'OLD':1}},
                                'optimal_allocation':[{'ticker':'NEW','target_weight':1}]})
    assert weights=={'NEW':1}
