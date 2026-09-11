import json
import unittest
from datetime import datetime, timezone
from public_market_mood import parse_mmi_page


class MarketMoodTests(unittest.TestCase):
    now=datetime(2026,9,7,8,tzinfo=timezone.utc)
    def page(self,score=46.4,stamp="2026-09-07T07:24:00Z"):
        data={"props":{"pageProps":{"nowData":{"indicator":score,"date":stamp}}}}
        return '<script id="__NEXT_DATA__" type="application/json">'+json.dumps(data)+'</script>'
    def test_current_reading_and_timestamp(self):
        result=parse_mmi_page(self.page(),now=self.now)
        self.assertEqual(result["zone"],"Fear")
        self.assertEqual(result["score"],46.4)
        self.assertFalse(result["older_reading"])
    def test_weekend_reading_is_labelled(self):
        result=parse_mmi_page(self.page(stamp="2026-09-04T10:00:00Z"),now=self.now)
        self.assertTrue(result["older_reading"])
    def test_official_scale_is_not_clipped_to_10_90(self):
        for score,zone in [(0,"Extreme fear"),(9,"Extreme fear"),(30,"Fear"),
                           (50,"Greed"),(70,"Greed"),(91,"Extreme greed"),(100,"Extreme greed")]:
            result=parse_mmi_page(self.page(score),now=self.now)
            self.assertEqual(result["score"],score)
            self.assertEqual(result["zone"],zone)
    def test_invalid_score_and_timestamp_fail(self):
        for score in [-1,101,float("nan"),True]:
            with self.assertRaises(ValueError):
                parse_mmi_page(self.page(score),now=self.now)
        for stamp in ["2026-01-01T00:00:00Z","2026-09-08T00:00:00Z","2026-09-07T07:00:00"]:
            with self.assertRaises(ValueError):
                parse_mmi_page(self.page(stamp=stamp),now=self.now)
    def test_page_change_does_not_use_unrelated_numbers(self):
        with self.assertRaises((ValueError,KeyError)):
            parse_mmi_page("<html>Sample MMI 55</html>",now=self.now)


if __name__=="__main__": unittest.main()
