import pytest
from election_reserve import reserve_decision_summary


def india(amount):
    return dict(status='active_event_stress',active_events=[{}],event_reserve_fraction=amount)


def us(amount):
    return dict(status='historical_scenario',active_events=[{}],event_reserve_fraction=amount)


def test_us_whole_mix_can_drive_reserve():
    result=reserve_decision_summary(.0185,{},us(.06))
    assert result['fraction']==.06
    assert result['driver']=='Driven by US whole-portfolio stress'
    assert '6.00%' in result['comparison']


def test_competing_active_events_use_maximum_not_sum():
    result=reserve_decision_summary(.02,india(.07),us(.06))
    assert result['fraction']==.07
    assert result['driver']=='Driven by India election stress'


def test_unassessed_us_is_not_reported_as_zero():
    result=reserve_decision_summary(.0185,{},dict(status='stale_or_incomplete_prices',active_events=[{}],event_reserve_fraction=None))
    assert result['fraction']==.0185
    assert result['driver']=='Driven by portfolio buffer'
    assert 'US whole-portfolio stress not assessed' in result['pending']
    assert '0.00%' not in result['comparison']


def test_inactive_numeric_event_is_excluded():
    result=reserve_decision_summary(.02,dict(india(.07),active_events=[]),dict(us(.06),status='no_active_us_event'))
    assert result['fraction']==.02
    assert len(result['components'])==1


@pytest.mark.parametrize('amount',[float('nan'),float('inf'),-.1])
def test_bad_event_value_cannot_drive_reserve(amount):
    result=reserve_decision_summary(.02,{},us(amount))
    assert result['fraction']==.02
    assert result['pending']
