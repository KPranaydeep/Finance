"""Explicit NSE metadata -> existing review-model categories. No ticker guessing."""
import csv
import io
import json
import os
import re
import tempfile
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

EQUITIES = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
ETFS = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"
ETF_KINDS = {"EQUITY": "equity_etf", "GLOBAL INDICES": "listed_non_equity_etf",
             "COMMODITY": "listed_non_equity_etf", "DEBT": "specified_debt_etf"}


def parse_registry(equities, etfs):
    def rows(text, required):
        reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
        if not reader.fieldnames:
            raise ValueError("INSTRUMENT_METADATA_INVALID")
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        if not required.issubset(reader.fieldnames):
            raise ValueError("INSTRUMENT_METADATA_INVALID")
        return [{k: (v or "").strip() for k, v in r.items() if k} for r in reader]
    result = {}
    for r in rows(equities, {"SYMBOL", "ISIN NUMBER", "SERIES", "NAME OF COMPANY"}):
        # INE is necessary, not sufficient: restrict to ordinary share series
        # from the equity master, not rights, preference shares, REITs or InvITs.
        if (r["SERIES"] in {"EQ", "BE", "BZ", "SM", "ST"} and
                re.fullmatch(r"INE[A-Z0-9]{9}", r["ISIN NUMBER"]) and
                not re.search(r"\b(REIT|INVIT|TRUST)\b", r["NAME OF COMPANY"], re.I)):
            result[r["SYMBOL"] + ".NS"] = "equity"
    for r in rows(etfs, {"Symbol", "ETF Underlying", "ISINNumber"}):
        ticker = r["Symbol"] + ".NS"
        # Even unsupported ETFs must not fall through as company equity.
        result.pop(ticker, None)
        kind = ETF_KINDS.get(r["ETF Underlying"].upper())
        if kind and re.fullmatch(r"INF[A-Z0-9]{9}", r["ISINNumber"]):
            result[ticker] = kind
    return result


@lru_cache(maxsize=2)
def _registry(bucket):
    try:
        def get(url):
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=15) as response:
                return response.read(2_000_000).decode("utf-8-sig")
        return parse_registry(get(EQUITIES), get(ETFS))
    except Exception:
        raise ValueError("INSTRUMENT_METADATA_UNAVAILABLE") from None


def complete_policy(policy, tickers, registry=None):
    result = deepcopy(policy)
    missing = set(tickers) - set(result["instrument_kinds"])
    if not missing:
        return result
    registry = registry if registry is not None else _registry(int(time.time() // 3600))
    if any(t not in registry for t in missing):
        raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
    result["instrument_kinds"].update({t: registry[t] for t in sorted(missing)})
    return result


def sync_policy_file(tickers):
    """Update only missing kinds. Exclusive lock + atomic replacement; keep approval."""
    from .config import load_policy
    path = Path(os.environ.get("PUBLIC_REVIEW_POLICY_PATH",
                               str(Path(__file__).resolve().parents[1] / "public_review_policy.json")))
    original = load_policy()
    updated = complete_policy(original, tickers)
    if updated == original:
        return updated
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("POLICY_UPDATE_BUSY") from None
    temporary = None
    try:
        os.close(fd)
        if json.loads(path.read_text(encoding="utf-8")) != original:
            raise ValueError("POLICY_CHANGED_RETRY")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temporary = f.name
            json.dump(updated, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    return updated
