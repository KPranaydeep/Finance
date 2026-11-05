import pytest

from election_seat_reserve import rolling_seat_reserve, DENOMINATORS
from election_schedule import parse_schedule, monthly_schedule
from election_reserve import reserve_decision_summary


def event(identity, day, assembly=0, ls=0, rs=0):
    return dict(name=identity, election_id=identity, counting_date=day, verified=True,
                source="https://www.eci.gov.in/example",
                contested_seats=dict(assembly=assembly, ls=ls, rs=rs))


def test_twenty_calendar_days_sum_each_house_then_max_and_portfolio_scale():
    events = [event('a', '2027-03-01', assembly=60), event('b','2027-03-20',assembly=40),
              event('ls','2027-03-10',ls=15), event('rs','2027-03-15',rs=20),
              event('too-late','2027-03-21',ls=543), event('past','2027-02-28',ls=543)]
    result = rolling_seat_reserve(events, '2027-03-01', .4)
    assert result['totals'] == dict(assembly=100,ls=15,rs=20)
    assert result['window_end'] == '2027-03-20'
    assert result['driver'] == 'Rajya Sabha'
    assert result['reserve_fraction'] == pytest.approx(.4 * 20/233)
    # A weekend or a holiday still counts as a calendar day.
    assert rolling_seat_reserve(events,'2027-03-21',.4)['reserve_fraction'] == .4


def test_repeated_sources_and_multiphase_polling_count_once():
    a = event('assembly:Delhi:2027','2027-03-15',assembly=70)
    a['poll_dates'] = ['2027-03-01','2027-03-05']
    b = dict(a,source='https://www.pib.gov.in/mirror')
    result = rolling_seat_reserve([a,b,a],'2027-03-01',1.)
    assert result['totals']['assembly'] == 70
    assert len(result['events']) == 1


def test_term_ends_geographic_allocations_unknown_counts_and_untrusted_sources():
    events = [dict(name='term',verified=True,source='https://www.eci.gov.in/test',
                   **{'Term ends':'2027-03-10','Lok Sabha allocation':80,'Rajya Sabha allocation':31}),
              dict(event('untrusted','2027-03-10',ls=543),source='https://example.com'),
              dict(name='unknown',verified=True,source='https://www.eci.gov.in/test',counting_date='2027-03-10')]
    result=rolling_seat_reserve(events,'2027-03-01',.4)
    assert result['reserve_fraction'] == 0
    assert not result['coverage_complete']
    assert result['missing_counts'] == ['unknown']


def test_conflicting_revisions_and_overlapping_totals_are_withheld():
    a=event('same','2027-03-10',assembly=60)
    r=rolling_seat_reserve([a,dict(a,counting_date='2027-03-11')],'2027-03-01',1)
    assert r['totals']['assembly']==0 and not r['coverage_complete']
    r=rolling_seat_reserve([event('x','2027-03-10',ls=543),event('y','2027-03-11',ls=1)],'2027-03-01',1)
    assert r['reserve_fraction']==0 and not r['coverage_complete']


@pytest.mark.parametrize('count',[float('nan'),float('inf'),-1,1.5,True])
def test_invalid_counts_do_not_create_a_reserve(count):
    r=rolling_seat_reserve([event('x','2027-03-10',ls=count)],'2027-03-01',1.)
    assert r['reserve_fraction']==0 and not r['coverage_complete']


def test_assembly_schedule_changes_no_direct_parliamentary_seats():
    text='General Election to Legislative Assemblies of Goa and Punjab Date of Poll: 05-03-2027 Date of Counting: 10-03-2027'
    events=parse_schedule(text,'https://www.eci.gov.in/test')
    r=rolling_seat_reserve(events,'2027-03-01',.5)
    assert r['totals']==dict(assembly=157,ls=0,rs=0)
    assert r['reserve_fraction']==pytest.approx(.5*157/DENOMINATORS['assembly'])


def test_rs_schedule_uses_actual_contests_not_state_allocation():
    text='Biennial Election to 24 seats in the Council of States Date of Poll: 05-03-2027 Date of Counting: 05-03-2027'
    r=rolling_seat_reserve(parse_schedule(text,'https://www.eci.gov.in/test'),'2027-03-01',.5)
    assert r['totals']==dict(assembly=0,ls=0,rs=24)


def test_selected_seat_policy_participates_in_max_never_sum():
    seat=rolling_seat_reserve([event('x','2027-03-10',ls=100)],'2027-03-01',.5)
    summary=reserve_decision_summary(.02,{}, {},seat)
    assert summary['fraction']==pytest.approx(.5*100/543)
    assert '20-day seat policy' in summary['driver']


def test_current_dated_assembly_denominator_and_partial_coverage():
    from election_watch import seed_snapshot
    snapshot=seed_snapshot()
    snapshot['rows'][0]['Assembly seats'] += 1
    snapshot['verified_at']='2027-01-02'
    r=rolling_seat_reserve([event('a','2027-03-10',assembly=100)],'2027-03-01',.4,assembly_snapshot=snapshot)
    assert r['denominators']['assembly']==4124
    assert r['denominator_verified_at']=='2027-01-02'
    assert r['reserve_fraction']==pytest.approx(.4*100/4124)
    snapshot['rows']=snapshot['rows'][:5]
    r=rolling_seat_reserve([event('a','2027-03-10',assembly=100)],'2027-03-01',.4,assembly_snapshot=snapshot)
    assert r['fractions']['assembly'] is None and not r['coverage_complete']


def test_daily_schedule_recheck_near_counting_has_one_attempt_per_date(tmp_path):
    import json
    path=tmp_path/'schedule.json'
    path.write_text(json.dumps(dict(events=[event('x','2027-03-10',assembly=60)],
                                   attempted_month='2027-03',attempted_date='2027-03-01',discovery_version=3)))
    calls=[]
    def fail(*a,**k):
        calls.append(a)
        raise OSError('offline')
    first=monthly_schedule(path,'2027-03-02',fetch=fail)
    second=monthly_schedule(path,'2027-03-02',fetch=fail)
    assert len(calls)==2  # one ECI attempt and one PIB fallback, no rerun loop
    assert first['events']==second['events'] and not second['coverage_complete']


def test_assembly_success_still_checks_parliament_notices_and_keeps_each_state(tmp_path):
    from election_schedule import INDEX, PIB_INDEX
    class Response:
        def __init__(self,text): self.text=text
        def raise_for_status(self): pass
    calls=[]
    def fetch(url,**kwargs):
        calls.append(url)
        if url==INDEX:
            return Response('<a href="/election-details/2027/S01/3">Assembly schedule</a>')
        if '/election-details/' in url:
            return Response('General Election to Legislative Assemblies of Goa and Punjab Date of Poll: 05-03-2027 Date of Counting: 10-03-2027')
        if url==PIB_INDEX:
            return Response('<a href="PressReleasePage.aspx?PRID=12345">Schedule for Biennial Election to Rajya Sabha</a>')
        return Response('Biennial Election to 24 seats in the Council of States Date of Poll: 05-03-2027 Date of Counting: 05-03-2027')
    result=monthly_schedule(tmp_path/'schedule.json','2027-03-01',fetch=fetch)
    assert PIB_INDEX in calls
    policy=rolling_seat_reserve(result['events'],'2027-03-01',.5)
    assert policy['totals']==dict(assembly=157,ls=0,rs=24)
    assert len(result['events'])==6
