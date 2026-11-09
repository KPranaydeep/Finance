import requests

from price_move_confirmation import download_confirmation


def test_confirmation_download_is_date_scoped_and_bounded():
    calls=[]
    class Response:
        text="Date,Close\n2026-01-02,5\n2026-01-01,10\n"
        def raise_for_status(self): pass
    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    result=download_confirmation("STLN","2026-01-01","2026-01-03",fetch=fetch)
    assert result["confirmation_close"].iloc[0] == 10
    assert calls[0][1]["timeout"] == (3,5)
    assert calls[0][1]["params"]["s"] == "stln.us"


def test_unsupported_listing_and_provider_failure_remain_unavailable():
    def fail(*args, **kwargs): raise requests.Timeout()
    assert download_confirmation("STLN","2026-01-01","2026-01-03",fetch=fail) is None
    assert download_confirmation("RELIANCE.NS","2026-01-01","2026-01-03",fetch=lambda *a,**k: (_ for _ in ()).throw(AssertionError())) is None
