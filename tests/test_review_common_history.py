import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from public_review.history import common_history
from public_review.forecast import paths, validate
from public_review.market import fetch
from public_world_benchmark import compare_world_benchmark, LABEL
from review_fixtures import policy, baseline
from public_price_currency import to_inr


class CommonHistoryTests(unittest.TestCase):
    def test_direct_us_quotes_converted_but_nse_etf_unchanged(self):
        dates=['2026-09-08','2026-09-09']
        native=pd.DataFrame({'CWT':[50.,50.],'MAFANG.NS':[200.,200.]},index=dates)
        result=to_inr(native,{'CWT':'USD','MAFANG.NS':'INR'},pd.Series([80.,88.],index=dates))
        self.assertEqual(list(result['CWT']),[4000.,4400.])
        self.assertEqual(list(result['MAFANG.NS']),[200.,200.])

    def test_missing_fx_is_not_filled_or_silently_treated_as_inr(self):
        dates=['2026-09-08','2026-09-09']
        native=pd.DataFrame({'CWT':[50.,50.]},index=dates)
        result=to_inr(native,{'CWT':'USD'},pd.Series([80.],index=dates[:1]))
        self.assertTrue(pd.isna(result['CWT'].iloc[-1]))
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_QUOTE_CURRENCY'):
            to_inr(native,{})

    def test_gap_does_not_become_one_day_return(self):
        dates = pd.bdate_range('2026-08-03', periods=5)
        labels = [str(d.date()) for d in dates]
        h = {'A.NS':pd.DataFrame({'Close':[100,101,np.nan,150,153]},index=labels)}
        with patch('public_review.market.calendar', return_value=pd.DataFrame(index=dates)):
            _, r, info = common_history(h, labels[-1], policy())
        self.assertEqual(list(r.index), [labels[1],labels[4]])
        self.assertAlmostEqual(r.iloc[-1,0], .02)
        self.assertEqual(info['missing_sessions'],[labels[2]])

    def test_invalid_endpoint_stays_blocked(self):
        dates = pd.bdate_range('2026-08-03', periods=3)
        h = {'A.NS':pd.DataFrame({'Close':[100,101,np.inf]},index=[str(d.date()) for d in dates])}
        with patch('public_review.market.calendar', return_value=pd.DataFrame(index=dates)):
            with self.assertRaisesRegex(ValueError,'STALE_OR_INCOMPLETE'):
                common_history(h,str(dates[-1].date()),policy())

    def test_bootstrap_cannot_continue_across_missing_session(self):
        r = pd.DataFrame({'A':np.arange(126)/1000})
        # Every row is separated by a gap. Every continuation MUST restart.
        r.attrs['session_positions'] = {i:2*i for i in r.index}
        a = paths(r, 4, 100, 100000, 15)
        b = paths(r, 4, 100, 1, 15)
        np.testing.assert_array_equal(a,b)

    def test_walk_forward_excludes_incomplete_test_windows(self):
        p=policy()
        dates=[str(d.date()) for d in pd.bdate_range(end='2026-09-09',periods=292)]
        r=pd.DataFrame(np.zeros((292,2)),columns=['A.NS','B.NS'],index=dates)
        r.attrs['session_positions']={d:i+(1 if i>=260 else 0) for i,d in enumerate(dates)}
        v=validate(r,baseline(),{'A.NS':100,'B.NS':50},dates,p)
        self.assertEqual(v['folds'],1)

    def test_fetch_rejects_nan_current_price(self):
        frame=pd.DataFrame({'Open':[100.], 'Close':[np.nan], 'Volume':[10.]},
                           index=pd.to_datetime(['2026-09-09']))
        with patch('public_review.market.yf.Ticker') as ticker:
            ticker.return_value.history.return_value=frame
            with self.assertRaisesRegex(ValueError,'INVALID_OR_NONTRADING_PRICE'):
                fetch(['A.NS'],'2026-09-09','2026-09-09',policy())

    def test_fx_is_included_once_in_world_benchmark(self):
        dates=['2026-09-08','2026-09-09']
        nav=[{'nav_date':d,'nav':100.} for d in dates]
        result=compare_world_benchmark(nav,pd.Series([100.,100.],index=dates),
                                       pd.Series([80.,88.],index=dates))
        self.assertAlmostEqual(result[LABEL].iloc[-1],110.)
        self.assertAlmostEqual(result['Portfolio — estimated net'].iloc[-1],100.)
