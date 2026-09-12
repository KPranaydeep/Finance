"""Explicit NSE metadata -> existing review-model categories. No ticker guessing."""
import csv
import io
import re
import time
from copy import deepcopy
from functools import lru_cache
from urllib.request import Request, urlopen
from .costs import KINDS

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


@lru_cache(maxsize=512)
def _foreign_kind(ticker, bucket):
    # The price engine already defines unsuffixed Yahoo symbols as direct-US
    # listings and converts them through USD/INR. Keep classification aligned
    # with that deterministic contract instead of depending on flaky get_info.
    # Exchange-suffixed foreign symbols remain unsupported and fail closed.
    if re.fullmatch(r"[A-Z][A-Z0-9-]{0,14}", ticker):
        return "foreign_us_listing"
    raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")


def require_supported_review(tickers):
    """Compatibility hook retained for callers; classification is checked later."""
    if not tickers:
        raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
    return True


def frozen_instrument_kinds(events):
    """Recover immutable classifications already recorded by the ledger."""
    result = {}
    for row in events or []:
        if row.get("kind") == "BASELINE":
            lots = row.get("payload", {}).get("lots", [])
        elif row.get("kind") == "SECURITY_ENTRY":
            lots = [row.get("payload", {})]
        else:
            continue
        for lot in lots:
            ticker, kind = lot.get("ticker"), lot.get("kind")
            if not ticker or not kind:
                continue
            if (kind not in KINDS or
                    (ticker.endswith(".NS") and kind == "foreign_us_listing") or
                    (not ticker.endswith(".NS") and kind != "foreign_us_listing")):
                raise ValueError("FROZEN_CLASSIFICATION_REVIEW_REQUIRED")
            prior = result.get(ticker)
            if prior is not None and prior != kind:
                raise ValueError("FROZEN_CLASSIFICATION_REVIEW_REQUIRED")
            result[ticker] = kind
    return result


def complete_policy(policy, tickers, registry=None, frozen_kinds=None):
    """Return an ephemeral policy with every requested ticker classified.

    ``instrument_kinds`` is deliberately absent from the owner policy file.
    A legacy in-memory mapping is still honoured as an explicit override so a
    rolling deployment cannot silently change an already supplied category.
    """
    result = deepcopy(policy)
    configured = {ticker: kind for ticker, kind in (frozen_kinds or {}).items()
                  if ticker in tickers}
    # Explicit compatibility overrides take precedence over recovered history.
    configured.update(result.pop("instrument_kinds", {}))
    missing = set(tickers) - set(configured)
    if not missing:
        result["instrument_kinds"] = {ticker: configured[ticker] for ticker in tickers}
        return result
    domestic = {t for t in missing if t.endswith(".NS")}
    bucket = int(time.time() // 3600)
    registry = registry if registry is not None else (_registry(bucket) if domestic else {})
    if any(t not in registry for t in domestic):
        raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
    configured.update({t: registry[t] for t in sorted(domestic)})
    configured.update({t: _foreign_kind(t, bucket) for t in sorted(missing - domestic)})
    result["instrument_kinds"] = {ticker: configured[ticker] for ticker in tickers}
    return result


def sync_policy_file(tickers):
    """Compatibility entry point: resolve kinds without modifying owner policy."""
    from .config import load_policy
    return complete_policy(load_policy(), tickers)
