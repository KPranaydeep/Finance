"""Read-only Tickertape MMI page data. No portfolio or database dependencies."""
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import json
import math
from urllib.request import Request, urlopen

SOURCE_URL="https://www.tickertape.in/market-mood-index"
MAX_PAGE_BYTES=2_000_000


class _PageData(HTMLParser):
    def __init__(self):
        super().__init__()
        self.capture=False
        self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self.capture=True
    def handle_endtag(self,tag):
        if tag == "script":
            self.capture=False
    def handle_data(self,data):
        if self.capture:
            self.parts.append(data)


def parse_mmi_page(page, *, now=None):
    parser=_PageData()
    parser.feed(page)
    payload=json.loads("".join(parser.parts))
    reading=payload["props"]["pageProps"]["nowData"]
    raw=reading["indicator"]
    if isinstance(raw,bool):
        raise ValueError("Invalid MMI score")
    score=float(raw)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("MMI score outside the official scale")
    source_at=datetime.fromisoformat(reading["date"].replace("Z","+00:00"))
    if source_at.tzinfo is None:
        raise ValueError("MMI source timestamp lacks a timezone")
    source_at=source_at.astimezone(timezone.utc)
    now=now or datetime.now(timezone.utc)
    age=now-source_at
    if age < -timedelta(minutes=10) or age > timedelta(days=7):
        raise ValueError("MMI source timestamp is outside the display window")
    zone=("Extreme fear" if score < 30 else "Fear" if score < 50
          else "Greed" if score <= 70 else "Extreme greed")
    return {"score":score,"zone":zone,"source_at":source_at.isoformat(),
            "fetched_at":now.isoformat(),"older_reading":age>timedelta(hours=24)}


def fetch_mmi():
    request=Request(SOURCE_URL,headers={"User-Agent":"Finance-MMI-Card/1.0","Accept":"text/html"})
    with urlopen(request,timeout=6) as response:
        page=response.read(MAX_PAGE_BYTES+1)
        if len(page)>MAX_PAGE_BYTES:
            raise ValueError("MMI page exceeded the size limit")
    return parse_mmi_page(page.decode("utf-8"))
