import unittest
import json
from datetime import date, datetime, timezone
from unittest.mock import patch
from public_review.core import digest
from public_review import store, market
from public_review.notifications import send
from review_fixtures import policy


class FakeResult:
    def __init__(self, rows): self.rows=rows
    def fetchall(self): return self.rows
    def fetchone(self): return self.rows[0] if self.rows else None


class FakeDB:
    def __init__(self): self.rows=[]; self.sql=[]
    def commit(self): pass
    def rollback(self): pass
    def execute(self, sql, params=None):
        self.sql.append(sql)
        if 'to_regclass' in sql: return FakeResult([{'name':'public_review_events'}])
        if sql.startswith('SELECT *'): return FakeResult(self.rows)
        if 'INSERT INTO public_review_events' in sql:
            b,k,t,i,p,prev,h=params
            self.rows.append(dict(basket_id=b,event_key=k,kind=t,baseline_id=i,payload=p.obj,
                                  previous_hash=prev,event_hash=h,seq=len(self.rows)+1))
        return FakeResult([])


class OperationTests(unittest.TestCase):
    def test_audit_idempotency_and_tampering(self):
        db=FakeDB()
        self.assertTrue(store.append(db,'B','event1','BASELINE','ID',{'cash':100}))
        self.assertFalse(store.append(db,'B','event1','BASELINE','ID',{'cash':200}))
        self.assertTrue(store.append(db,'B','event2','ASSESSMENT','ID',{'net':120}))
        self.assertEqual(len(store.read(db,'B')),2)
        db.rows[0]['payload']['cash']=101
        with self.assertRaises(ValueError): store.read(db,'B')

    def test_no_publication_or_trade_writes(self):
        db=FakeDB(); store.append(db,'B','x','ASSESSMENT','ID',{})
        self.assertFalse(any('INSERT INTO trade_' in q or 'UPDATE public_portfolio_' in q for q in db.sql))

    def test_notifications_opt_in(self):
        with patch.dict('os.environ',{'PUBLIC_REVIEW_ALERT_CHANNEL':'none'}), patch('public_review.notifications.urlopen') as net:
            self.assertFalse(send('test'))
            net.assert_not_called()

    def test_publication_after_close_uses_next_open(self):
        p=policy()
        entry,latest,future=market.sessions(datetime(2026,9,9,13,tzinfo=timezone.utc),'2026-09-05T12:15:55+00:00',p)
        self.assertEqual(entry,'2026-09-07')
        self.assertEqual(latest,'2026-09-09')
        self.assertNotIn('2026-09-14',future)

    def test_awaiting_entry_and_naive_time(self):
        p=policy()
        with self.assertRaises(ValueError): market.sessions(datetime(2026,9,6,13,tzinfo=timezone.utc),'2026-09-05T12:15:55+00:00',p)
        with self.assertRaises(ValueError): market.sessions(datetime(2026,9,9,13,tzinfo=timezone.utc),'2026-09-05T12:15:55',p)

    def test_unapproved_policy_blocks(self):
        from public_review.config import load_policy
        for approval in (False, None, 'true', 1):
            with self.subTest(approval=approval):
                fixture=policy()
                fixture['policy_approved']=approval
                with patch('public_review.config.Path.read_text',return_value=json.dumps(fixture)):
                    with self.assertRaisesRegex(ValueError,'POLICY_APPROVAL_REQUIRED'):
                        load_policy(today=date(2026,9,9))

    def test_approved_policy_loads(self):
        from public_review.config import load_policy
        fixture=policy()
        with patch('public_review.config.Path.read_text',return_value=json.dumps(fixture)):
            self.assertEqual(load_policy(today=date(2026,9,9)),fixture)

    def test_approved_but_stale_tariff_blocks(self):
        from public_review.config import load_policy
        with patch('public_review.config.Path.read_text',return_value=json.dumps(policy())):
            with self.assertRaisesRegex(ValueError,'TARIFF_REVIEW_REQUIRED'):
                load_policy(today=date(2026,12,9))

    def test_policy_fixture_is_independent_and_returns_fresh_data(self):
        with patch('pathlib.Path.read_text',side_effect=AssertionError('No production-file reads in fixture')):
            first=policy()
            first['policy_approved']=False
            first['instrument_kinds'].clear()
            fresh=policy()
        self.assertTrue(fresh['policy_approved'])
        self.assertEqual(len(fresh['instrument_kinds']),2)

    def test_delivery_error_is_recorded_without_exception_or_secrets(self):
        from public_review.service import notify_safely
        db=FakeDB()
        with patch('public_review.service.send',side_effect=RuntimeError('SECRET_DO_NOT_STORE')):
            ok=notify_safely(db,'B','ID',{'status':'CANNOT_ASSESS'},datetime(2026,9,9,13,tzinfo=timezone.utc))
        self.assertFalse(ok)
        self.assertEqual(db.rows[-1]['kind'],'ALERT_FAILED')
        self.assertNotIn('SECRET_DO_NOT_STORE',json.dumps(db.rows))

    def test_alert_unchanged_is_deduplicated(self):
        from public_review.service import maybe_notify
        db=FakeDB(); now=datetime(2026,9,9,13,tzinfo=timezone.utc)
        with patch('public_review.service.send',return_value=True) as sender:
            maybe_notify(db,'B','ID',{'status':'CANNOT_ASSESS'},now)
            maybe_notify(db,'B','ID',{'status':'CANNOT_ASSESS'},now)
            self.assertEqual(sender.call_count,1)

    def test_actual_schema_publication_status_column(self):
        from public_review.service import publications
        db=FakeDB()
        publications(db,'B')
        self.assertIn('v.publication_status',db.sql[0])
        self.assertNotIn('v.status=',db.sql[0])
