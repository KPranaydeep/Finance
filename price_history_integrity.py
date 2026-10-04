"""Corporate-action reconciliation for optimizer price histories.

The optimizer must never interpret a share-count change as investment return.
This module is deliberately provider-agnostic: callers supply an adjusted price
series plus raw close, adjusted close, dividends and split evidence for only the
tickers whose downloaded history contains a suspicious discontinuity.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


PRICE_INTEGRITY_VERSION = "corporate-action-reconciliation-v1"
DEFAULT_EXTREME_RETURN_THRESHOLD = 0.45


def extreme_return_dates(
    prices: pd.Series,
    *,
    threshold: float = DEFAULT_EXTREME_RETURN_THRESHOLD,
) -> list[pd.Timestamp]:
    """Return sessions whose adjacent observed prices move beyond ``threshold``."""
    clean = (
        pd.to_numeric(prices, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    clean = clean[clean > 0]
    if len(clean) < 2:
        return []
    simple_returns = clean.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return [pd.Timestamp(value) for value in simple_returns.index[simple_returns.abs() >= float(threshold)]]


def split_normalized_adjusted_close(
    adjusted_close: pd.Series,
    stock_splits: pd.Series,
) -> pd.Series:
    """Express all adjusted prices on the latest share basis.

    Yahoo represents a 1-for-10 reverse split as ``0.1`` and a 2-for-1 forward
    split as ``2``. Dividing observations before the event by the product of all
    later ratios makes both forms continuous while retaining dividend adjustment
    already present in ``Adj Close``.
    """
    prices = pd.to_numeric(adjusted_close, errors="coerce").astype(float)
    splits = (
        pd.to_numeric(stock_splits, errors="coerce")
        .reindex(prices.index)
        .fillna(0.0)
        .astype(float)
    )
    event_ratios = splits.where(splits > 0.0, 1.0)
    including_current = event_ratios.iloc[::-1].cumprod().iloc[::-1]
    later_event_factor = including_current / event_ratios
    normalized = prices / later_event_factor.replace(0.0, np.nan)
    return normalized.replace([np.inf, -np.inf], np.nan)


def total_return_index_from_actions(
    raw_close: pd.Series,
    dividends: pd.Series | None,
    stock_splits: pd.Series | None,
) -> pd.Series:
    """Build a split-safe total-return index from raw close and cash distributions.

    Yahoo can leave ``Adj Close`` internally inconsistent around a newly recorded
    reverse split. Its dividend field is already expressed on the current-share
    basis, so dividends are added to a raw close series first normalized onto that
    same basis. The resulting wealth index preserves economic returns while its
    absolute level remains irrelevant to optimization.
    """
    raw = pd.to_numeric(raw_close, errors="coerce").astype(float)
    splits = (
        pd.to_numeric(stock_splits, errors="coerce").reindex(raw.index).fillna(0.0)
        if stock_splits is not None
        else pd.Series(0.0, index=raw.index)
    )
    normalized_close = split_normalized_adjusted_close(raw, splits)
    cash = (
        pd.to_numeric(dividends, errors="coerce").reindex(raw.index).fillna(0.0)
        if dividends is not None
        else pd.Series(0.0, index=raw.index)
    )
    previous_close = normalized_close.ffill().shift(1)
    daily_total_return = (
        (normalized_close + cash) / previous_close - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    valid_close = normalized_close.notna()
    first_valid = normalized_close.first_valid_index()
    if first_valid is None:
        return normalized_close
    wealth = pd.Series(np.nan, index=raw.index, dtype=float)
    start_position = raw.index.get_loc(first_valid)
    growth = (1.0 + daily_total_return.iloc[start_position:].fillna(0.0)).cumprod()
    wealth.iloc[start_position:] = float(normalized_close.loc[first_valid]) * growth
    wealth.loc[~valid_close] = np.nan
    return wealth


def reconcile_flagged_history(
    downloaded_prices: pd.Series,
    *,
    raw_close: pd.Series,
    adjusted_close: pd.Series,
    dividends: pd.Series | None = None,
    stock_splits: pd.Series | None = None,
    threshold: float = DEFAULT_EXTREME_RETURN_THRESHOLD,
) -> tuple[pd.Series, dict[str, object]]:
    """Reconcile one flagged ticker and return a corrected series plus audit data."""
    original_dates = extreme_return_dates(downloaded_prices, threshold=threshold)
    raw = pd.to_numeric(raw_close, errors="coerce")
    adjusted = pd.to_numeric(adjusted_close, errors="coerce")
    if raw.dropna().empty:
        raw = adjusted
    splits = (
        pd.to_numeric(stock_splits, errors="coerce")
        if stock_splits is not None
        else pd.Series(0.0, index=raw.index)
    )
    splits = splits.reindex(raw.index).fillna(0.0)
    corrected = total_return_index_from_actions(raw, dividends, splits)
    remaining_dates = extreme_return_dates(corrected, threshold=threshold)
    split_events = [
        {
            "date": pd.Timestamp(date).date().isoformat(),
            "ratio": float(ratio),
        }
        for date, ratio in splits.items()
        if pd.notna(ratio) and float(ratio) > 0.0 and not np.isclose(float(ratio), 1.0)
    ]

    if not remaining_dates and split_events:
        status = "REPAIRED_CORPORATE_ACTION"
        usable = corrected
    elif remaining_dates:
        status = "QUARANTINED_UNEXPLAINED_JUMP"
        usable = corrected
    else:
        status = "VERIFIED_AFTER_RECHECK"
        usable = corrected

    report = {
        "version": PRICE_INTEGRITY_VERSION,
        "status": status,
        "trigger_dates": [value.date().isoformat() for value in original_dates],
        "remaining_extreme_dates": [value.date().isoformat() for value in remaining_dates],
        "split_events": split_events,
        "dividend_events": int(
            pd.to_numeric(dividends, errors="coerce").fillna(0.0).ne(0.0).sum()
        )
        if dividends is not None
        else 0,
    }
    return usable, report


def reconcile_price_frame(
    prices: pd.DataFrame,
    evidence: Mapping[str, Mapping[str, pd.Series] | None],
    *,
    owned_tickers=(),
    threshold: float = DEFAULT_EXTREME_RETURN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit suspicious columns, correcting splits and quarantining candidates."""
    corrected = prices.copy()
    owned = {str(ticker).strip().upper() for ticker in owned_tickers if str(ticker).strip()}
    rows: list[dict[str, object]] = []
    for column in list(corrected.columns):
        ticker = str(column).strip().upper()
        trigger_dates = extreme_return_dates(corrected[column], threshold=threshold)
        if not trigger_dates:
            continue
        ticker_evidence = evidence.get(ticker)
        if not ticker_evidence:
            status = "QUARANTINED_VALIDATION_UNAVAILABLE"
            report = {
                "version": PRICE_INTEGRITY_VERSION,
                "status": status,
                "trigger_dates": [value.date().isoformat() for value in trigger_dates],
                "remaining_extreme_dates": [value.date().isoformat() for value in trigger_dates],
                "split_events": [],
                "dividend_events": 0,
            }
        else:
            repaired, report = reconcile_flagged_history(
                corrected[column],
                raw_close=ticker_evidence["raw_close"],
                adjusted_close=ticker_evidence["adjusted_close"],
                dividends=ticker_evidence.get("dividends"),
                stock_splits=ticker_evidence.get("stock_splits"),
                threshold=threshold,
            )
            corrected[column] = repaired.reindex(corrected.index)

        is_owned = ticker in owned
        status = str(report["status"])
        quarantine = status.startswith("QUARANTINED_")
        rows.append(
            {
                "Ticker": ticker,
                "Owned": is_owned,
                "Status": status,
                "Optimizer action": (
                    "FREEZE_OWNED_WEIGHT" if quarantine and is_owned
                    else "EXCLUDE_NEW_CANDIDATE" if quarantine
                    else "USE_RECONCILED_HISTORY"
                ),
                "Trigger dates": ", ".join(report["trigger_dates"]),
                "Remaining extreme dates": ", ".join(report["remaining_extreme_dates"]),
                "Split events": ", ".join(
                    f"{item['date']} x {item['ratio']:g}" for item in report["split_events"]
                ),
                "Dividend events": int(report["dividend_events"]),
                "Integrity version": report["version"],
            }
        )
        if quarantine and not is_owned:
            corrected = corrected.drop(columns=[column])

    return corrected, pd.DataFrame(rows)
