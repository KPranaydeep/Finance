"""Monthly, source-dated election context; independent of optimizer/trades."""
from datetime import date, datetime, timezone
from pathlib import Path
import io
import json
import os
import tempfile
import pandas as pd
import requests

SOURCE = "https://www.eci.gov.in/term-of-the-houses/"
PARLIAMENT_SOURCE = "https://sansad.in/rs/faq"
# Verified 30 Oct 2026 from ECI's house-terms table and official publications.
# All 31 assemblies; allocations are geographic, never party/coalition counts.
SEED = [
    ("Manipur", "2027-03-13", 60, 2, 1),
    ("Goa", "2027-03-14", 40, 2, 1),
    ("Punjab", "2027-03-16", 117, 13, 7),
    ("Uttarakhand", "2027-03-28", 70, 5, 3),
    ("Uttar Pradesh", "2027-05-22", 403, 80, 31),
    ("Gujarat", "2027-12-19", 182, 26, 11),
    ("Himachal Pradesh", "2028-01-03", 68, 4, 3),
    ("Andhra Pradesh", "2029-06-20", 175, 25, 11),
    ("Arunachal Pradesh", "2029-06-13", 60, 2, 1),
    ("Assam", "2031-05-20", 126, 14, 7),
    ("Bihar", "2030-11-30", 243, 40, 16),
    ("Chhattisgarh", "2028-12-18", 90, 11, 5),
    ("Haryana", "2029-10-24", 90, 10, 5),
    ("Jharkhand", "2029-12-08", 81, 14, 6),
    ("Karnataka", "2028-05-21", 224, 28, 12),
    ("Kerala", "2031-05-20", 140, 20, 9),
    ("Madhya Pradesh", "2028-12-17", 230, 29, 11),
    ("Maharashtra", "2029-12-06", 288, 48, 19),
    ("Meghalaya", "2028-03-05", 60, 2, 1),
    ("Mizoram", "2028-12-11", 40, 1, 1),
    ("Nagaland", "2028-03-19", 60, 1, 1),
    ("NCT Delhi", "2030-02-23", 70, 7, 3),
    ("Odisha", "2029-07-21", 147, 21, 10),
    ("Puducherry", "2031-05-19", 30, 1, 1),
    ("Rajasthan", "2028-12-19", 200, 25, 10),
    ("Sikkim", "2029-06-11", 32, 1, 1),
    ("Tamil Nadu", "2031-05-10", 234, 39, 18),
    ("Telangana", "2028-12-08", 119, 17, 7),
    ("Tripura", "2028-03-23", 60, 2, 1),
    ("Jammu & Kashmir", "2029-11-03", 90, 5, 4),
    ("West Bengal", "2031-05-14", 294, 42, 16),
]


def seed_snapshot():
    return {"verified_at": "2026-10-30", "source": SOURCE,
            "coverage": "31-assembly ECI term baseline; polling/counting dates not verified",
            "rows": [dict(zip(("State", "Term ends", "Assembly seats", "Lok Sabha allocation", "Rajya Sabha allocation"), row)) for row in SEED]}


def parse_eci_terms(html):
    """Accept only a recognizable official house-terms schema; never guess dates."""
    for table in pd.read_html(io.StringIO(html)):
        # ECI can put headings in the first data row rather than a thead.
        if not any("ASSEMBLY" in str(col).upper() for col in table.columns):
            if not table.empty:
                table.columns = table.iloc[0].astype(str)
                table = table.iloc[1:]
        columns = {" ".join(str(col).upper().split()): col for col in table.columns}
        state = next((v for k,v in columns.items() if "STATE" in k), None)
        end = columns.get("TO")
        assembly = next((v for k,v in columns.items() if "ASSEMBLY" in k and "SEAT" in k), None)
        lok = next((v for k,v in columns.items() if "LOK" in k and "SEAT" in k), None)
        rajya = next((v for k,v in columns.items() if "RAJYA" in k and "SEAT" in k), None)
        if any(col is None for col in (state,end,assembly,lok,rajya)):
            continue
        rows = []
        for _, row in table.iterrows():
            end_date = pd.to_datetime(str(row[end]).strip(), dayfirst=True, errors="coerce")
            seats = [pd.to_numeric(row[col], errors="coerce") for col in (assembly,lok,rajya)]
            if pd.isna(end_date) or any(pd.isna(x) for x in seats):
                continue
            if seats[0] <= 0 or seats[0] > 500 or seats[1] < 0 or seats[1] > 80 or seats[2] < 0 or seats[2] > 31:
                continue
            rows.append({"State": str(row[state]).strip().title(), "Term ends": str(end_date.date()),
                         "Assembly seats": int(seats[0]), "Lok Sabha allocation": int(seats[1]), "Rajya Sabha allocation": int(seats[2])})
        if len(rows) >= 31 and len({r["State"] for r in rows}) == len(rows):
            return rows
    raise ValueError("ECI response did not contain a valid state-house term table")


def _atomic_save(path, snapshot):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2)
        temporary = handle.name
    os.replace(temporary, path)


def monthly_snapshot(path, today=None, force=False, fetch=None):
    today = today or date.today()
    path = Path(path)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not saved.get("rows"):
            raise ValueError("empty snapshot")
    except (OSError, ValueError, TypeError):
        saved = seed_snapshot()
    month = today.strftime("%Y-%m")
    if not force and saved.get("attempted_month") == month:
        return saved
    updated = dict(saved, attempted_month=month, attempted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        response = (fetch or requests.get)(SOURCE, timeout=15, headers={"User-Agent": "Mozilla/5.0 portfolio election watch"})
        response.raise_for_status()
        rows = parse_eci_terms(response.text)
        updated.update(rows=rows, verified_at=str(today), source=SOURCE, coverage="Official ECI house-terms table; polling/counting dates require separate ECI notifications")
        updated.pop("refresh_error", None)
    except Exception as exc:
        updated["refresh_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
    _atomic_save(path, updated)
    return updated


def upcoming_terms(snapshot, today, months=18):
    end = (pd.Timestamp(today) + pd.DateOffset(months=months)).date()
    rows = []
    for item in snapshot.get("rows", []):
        term_end = date.fromisoformat(item["Term ends"])
        if today <= term_end <= end:
            rows.append(dict(item, **{"Days to term end": (term_end-today).days,
                                       "Full-house majority": int(item["Assembly seats"])//2 + 1,
                                       "Election dates": "Not verified — term end is not polling day"}))
    return sorted(rows, key=lambda row: row["Term ends"])


def market_context(closes):
    prices = pd.to_numeric(closes, errors="coerce").dropna().sort_index()
    if len(prices) < 253:
        raise ValueError("At least 253 daily index observations are required")
    window = prices.iloc[-252:]
    drawdown = float(prices.iloc[-1] / window.max() - 1)
    returns = prices.pct_change(fill_method=None).dropna()
    latest_vol = float(returns.iloc[-30:].std() * 250 ** .5)
    historic_vol = returns.rolling(30).std().dropna().iloc[-252:]
    return {"as_of": str(prices.index[-1])[:10], "drawdown": drawdown,
            "volatility": latest_vol, "volatility_percentile": float((historic_vol <= latest_vol).mean()*100)}
