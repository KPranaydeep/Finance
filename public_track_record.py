from __future__ import annotations

import hashlib
import json
import textwrap
import zipfile
from datetime import date, datetime, time, timedelta
from io import BytesIO, StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import streamlit as st
import yfinance as yf


BENCHMARK_TICKER = "^NSEI"
BENCHMARK_LABEL = "Nifty 50"
WORLD_TICKER = "VT"
FX_TICKER = "INR=X"
WORLD_LABEL = "Global stocks — VT (INR)"
IST = ZoneInfo("Asia/Kolkata")
NEW_YORK = ZoneInfo("America/New_York")



def clean_ticker(value: str) -> str:
    return "".join(str(value).strip().upper().split())


def exited_symbol_rows(
    feed: dict,
    *,
    lookback_days: int | None,
    as_of: date | None = None,
) -> list[dict]:
    """Return currently exited symbols with a dated, auditable exit.

    Active symbols always win if a future feed contains more than one lifecycle
    row for the same ticker.  This prevents a re-entered holding from appearing
    in a removal list.
    """
    reference_date = as_of or date.today()
    securities = feed.get("securities") or []
    active = {
        clean_ticker(item.get("ticker", ""))
        for item in securities
        if str(item.get("status", "active")).lower() != "removed"
    }
    rows: list[dict] = []
    seen: set[str] = set()
    for item in securities:
        if str(item.get("status", "active")).lower() != "removed":
            continue
        ticker = clean_ticker(item.get("ticker", ""))
        raw_exit = item.get("exit_date")
        if not ticker or ticker in active or ticker in seen or not raw_exit:
            continue
        try:
            exit_date = date.fromisoformat(str(raw_exit)[:10])
        except ValueError:
            continue
        days_since_exit = (reference_date - exit_date).days
        if days_since_exit < 0:
            continue
        if lookback_days is not None and days_since_exit >= lookback_days:
            continue
        rows.append({
            "Ticker": ticker,
            "Exit date": exit_date.isoformat(),
            "Days since exit": days_since_exit,
        })
        seen.add(ticker)
    return sorted(rows, key=lambda row: (row["Exit date"], row["Ticker"]), reverse=True)


def exited_symbols_text(rows: list[dict]) -> bytes:
    """Paste-ready input for 'Remove symbols from universal portfolio'."""
    return ("\n".join(row["Ticker"] for row in rows) + ("\n" if rows else "")).encode("utf-8")


def exited_symbols_csv(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows, columns=["Ticker", "Exit date", "Days since exit"]).to_csv(
        index=False
    ).encode("utf-8-sig")


def _single_ticker_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        levels = [set(map(str, result.columns.get_level_values(i)))
                  for i in range(result.columns.nlevels)]
        ticker_level = next((i for i, values in enumerate(levels)
                             if ticker in values), None)
        if ticker_level is not None:
            result = result.xs(ticker, axis=1, level=ticker_level)
        else:
            result.columns = result.columns.get_level_values(0)
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_convert(IST).tz_localize(None)
    result.index = result.index.normalize()
    return result[~result.index.duplicated(keep="last")].sort_index()


@st.cache_data(ttl=900, max_entries=64, show_spinner=False)
def load_daily_history(ticker: str, start: date, requested_end: date,
                       refresh_bucket: str = "") -> pd.DataFrame:
    # Yahoo's end date is exclusive. The extra day lets a completed current
    # session appear when the provider has published it.
    frame = yf.download(
        ticker,
        start=(start - timedelta(days=10)).isoformat(),
        end=(requested_end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    return _single_ticker_frame(frame, ticker)


@st.cache_data(ttl=60, max_entries=32, show_spinner=False)
def load_latest_ltp(ticker: str) -> tuple[float | None, str | None]:
    """Return latest Yahoo intraday LTP separately from daily history."""
    frame = yf.download(ticker, period="1d", interval="1m", auto_adjust=False,
                        actions=False, progress=False, threads=False)
    if frame.empty or "Close" not in frame:
        return None, None
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        return None, None
    observed = close.index[-1]
    if getattr(observed, "tzinfo", None) is not None:
        observed = observed.tz_convert(IST)
    return float(close.iloc[-1]), observed.strftime("%Y-%m-%d %H:%M %Z")


def completed_nse_history(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Exclude a possibly incomplete current NSE daily candle."""
    result = frame.copy()
    today = now.astimezone(IST).date()
    if now.astimezone(IST).time() < time(16, 0):
        result = result[result.index.date < today]
    else:
        result = result[result.index.date <= today]
    return result


def completed_us_history(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Exclude a possibly incomplete current US daily candle."""
    result = frame.copy()
    local_now = now.astimezone(NEW_YORK)
    if local_now.time() < time(16, 30):
        result = result[result.index.date < local_now.date()]
    else:
        result = result[result.index.date <= local_now.date()]
    return result


def price_columns(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "Close" not in frame:
        raise ValueError("Yahoo did not return a closing-price column.")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    adjusted = (
        pd.to_numeric(frame["Adj Close"], errors="coerce")
        if "Adj Close" in frame else close.copy()
    )
    valid = close.gt(0) & adjusted.gt(0) & np.isfinite(close) & np.isfinite(adjusted)
    return close.where(valid).dropna(), adjusted.where(valid).dropna()


def analyze(ticker_frame: pd.DataFrame, benchmark_frame: pd.DataFrame,
            world_frame: pd.DataFrame, fx_frame: pd.DataFrame,
            requested_start: date, *, ticker_currency: str = "INR") -> tuple[dict, pd.DataFrame]:
    start = pd.Timestamp(requested_start)
    ticker_close, ticker_adjusted = price_columns(ticker_frame)
    benchmark_close, benchmark_adjusted = price_columns(benchmark_frame)
    _, world_adjusted_usd = price_columns(world_frame)
    fx_close, _ = price_columns(fx_frame)
    ticker_close = ticker_close[ticker_close.index >= start]
    ticker_adjusted = ticker_adjusted[ticker_adjusted.index >= start]
    benchmark_close = benchmark_close[benchmark_close.index >= start]
    benchmark_adjusted = benchmark_adjusted[benchmark_adjusted.index >= start]
    world_adjusted_usd = world_adjusted_usd[world_adjusted_usd.index >= start]
    fx_close = fx_close[fx_close.index >= start]
    if ticker_close.empty or ticker_adjusted.empty:
        raise ValueError("No ticker session is available on or after the selected date.")
    if benchmark_close.empty or benchmark_adjusted.empty:
        raise ValueError("No Nifty 50 session is available on or after the selected date.")
    world_inr = pd.concat(
        [world_adjusted_usd.rename("vt_usd"), fx_close.rename("usd_inr")], axis=1
    ).dropna()
    world_inr = world_inr[world_inr.gt(0).all(axis=1)]
    if world_inr.empty:
        raise ValueError("No same-date VT and USD/INR observations are available after the selected date.")
    world_adjusted_inr = world_inr["vt_usd"] * world_inr["usd_inr"]

    normalized_currency = ticker_currency.strip().upper()
    if normalized_currency == "USD":
        ticker_inr = pd.concat(
            [ticker_adjusted.rename("ticker"), fx_close.rename("usd_inr")], axis=1
        ).dropna()
        ticker_inr = ticker_inr[ticker_inr.gt(0).all(axis=1)]
        if ticker_inr.empty:
            raise ValueError(
                "No same-date security and USD/INR observations are available after the publication date."
            )
        ticker_adjusted_for_return = ticker_inr["ticker"] * ticker_inr["usd_inr"]
    else:
        ticker_adjusted_for_return = ticker_adjusted

    ticker_entry = ticker_adjusted_for_return.index[0]
    benchmark_entry = benchmark_adjusted.index[0]
    world_entry = world_adjusted_inr.index[0]
    common_end = min(
        ticker_adjusted_for_return.index[-1], benchmark_adjusted.index[-1],
        world_adjusted_inr.index[-1],
    )
    ticker_adjusted_for_return = ticker_adjusted_for_return[
        ticker_adjusted_for_return.index <= common_end
    ]
    ticker_close = ticker_close[ticker_close.index <= common_end]
    benchmark_adjusted = benchmark_adjusted[benchmark_adjusted.index <= common_end]
    benchmark_close = benchmark_close[benchmark_close.index <= common_end]
    world_adjusted_inr = world_adjusted_inr[world_adjusted_inr.index <= common_end]

    ticker_growth = (
        ticker_adjusted_for_return / float(ticker_adjusted_for_return.iloc[0]) * 100
    )
    benchmark_growth = benchmark_adjusted / float(benchmark_adjusted.iloc[0]) * 100
    world_growth = world_adjusted_inr / float(world_adjusted_inr.iloc[0]) * 100
    combined = pd.concat(
        [
            ticker_growth.rename("Ticker"),
            benchmark_growth.rename(BENCHMARK_LABEL),
            world_growth.rename(WORLD_LABEL),
        ],
        axis=1,
    )
    ticker_return = float(ticker_growth.iloc[-1] / 100 - 1)
    benchmark_return = float(benchmark_growth.iloc[-1] / 100 - 1)
    world_return = float(world_growth.iloc[-1] / 100 - 1)
    drawdown = ticker_growth / ticker_growth.cummax() - 1
    metrics = {
        "requested_start": requested_start.isoformat(),
        "ticker_entry_date": ticker_entry.date().isoformat(),
        "benchmark_entry_date": benchmark_entry.date().isoformat(),
        "world_entry_date": world_entry.date().isoformat(),
        "as_of": common_end.date().isoformat(),
        "entry_close": float(ticker_close.loc[ticker_entry]),
        "latest_close": float(ticker_close.iloc[-1]),
        "ticker_return": ticker_return,
        "benchmark_return": benchmark_return,
        "world_return": world_return,
        "excess_return": ticker_return - benchmark_return,
        "excess_world_return": ticker_return - world_return,
        "max_drawdown": float(drawdown.min()),
        "calendar_days": int((common_end.date() - ticker_entry.date()).days),
        "ticker_sessions": int(len(ticker_adjusted_for_return)),
        "price_currency": normalized_currency,
        "price_symbol": "$" if normalized_currency == "USD" else "₹",
        "return_currency": "INR",
    }
    return metrics, combined


@st.cache_data(ttl=300, max_entries=128, show_spinner=False)
def load_security_evidence(
    ticker: str,
    entry_date: str,
    exit_date: str | None = None,
    refresh_bucket: str = "",
) -> tuple[dict, pd.DataFrame]:
    """Build publication-linked evidence for one active or exited security."""
    clean = clean_ticker(ticker)
    start = date.fromisoformat(entry_date[:10])
    requested_end = date.fromisoformat(exit_date[:10]) if exit_date else date.today()
    fetch_end = requested_end + timedelta(days=7) if exit_date else requested_end
    now = datetime.now(IST)
    is_indian = clean.endswith((".NS", ".BO"))

    security_history = load_daily_history(clean, start, fetch_end, refresh_bucket)
    security_history = (
        completed_nse_history(security_history, now)
        if is_indian
        else completed_us_history(security_history, now)
    )
    benchmark_history = completed_nse_history(
        load_daily_history(BENCHMARK_TICKER, start, fetch_end, refresh_bucket), now
    )
    world_history = completed_us_history(
        load_daily_history(WORLD_TICKER, start, fetch_end, refresh_bucket), now
    )
    fx_history = completed_us_history(
        load_daily_history(FX_TICKER, start, fetch_end, refresh_bucket), now
    )

    if exit_date:
        security_history = security_history[security_history.index.date <= requested_end]
        benchmark_history = benchmark_history[benchmark_history.index.date <= requested_end]
        world_history = world_history[world_history.index.date <= requested_end]
        fx_history = fx_history[fx_history.index.date <= requested_end]

    metrics, chart = analyze(
        security_history,
        benchmark_history,
        world_history,
        fx_history,
        start,
        ticker_currency="INR" if is_indian else "USD",
    )
    metrics["entry_source"] = "Immutable public portfolio publication"
    if exit_date:
        metrics["endpoint_label"] = "Exit close"
        metrics["endpoint_price"] = float(metrics["latest_close"])
        metrics["endpoint_as_of"] = metrics["as_of"]
    else:
        ltp, ltp_as_of = load_latest_ltp(clean)
        if ltp is None:
            metrics["endpoint_label"] = "Latest completed close"
            metrics["endpoint_price"] = float(metrics["latest_close"])
            metrics["endpoint_as_of"] = metrics["as_of"]
        else:
            metrics["endpoint_label"] = "Latest LTP"
            metrics["endpoint_price"] = float(ltp)
            metrics["endpoint_as_of"] = ltp_as_of or "timestamp unavailable"
    metrics["ltp"] = metrics["endpoint_price"]
    metrics["ltp_as_of"] = metrics["endpoint_as_of"]
    return metrics, chart


def percent(value: float) -> str:
    return f"{value:+.2%}"


def percentage_points(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def evidence_fingerprint(ticker: str, metrics: dict,
                         chart: pd.DataFrame) -> str:
    """Return a stable identifier for the exact inputs shown in an export."""
    metadata = json.dumps(
        {"ticker": ticker, **metrics}, sort_keys=True,
        separators=(",", ":"), ensure_ascii=True,
    )
    observations = chart.sort_index().to_csv(
        date_format="%Y-%m-%d", float_format="%.8f", na_rep=""
    )
    return hashlib.sha256(f"{metadata}\n{observations}".encode("utf-8")).hexdigest()


def evidence_csv(ticker: str, metrics: dict, chart: pd.DataFrame) -> bytes:
    output = StringIO()
    output.write("# Recommendation track-record evidence\n")
    for key, value in {
        "ticker": ticker,
        "india_benchmark": BENCHMARK_TICKER,
        "world_benchmark": WORLD_TICKER,
        "world_fx": FX_TICKER,
        "evidence_sha256": evidence_fingerprint(ticker, metrics, chart),
        "evidence_start_date": metrics["requested_start"],
        **metrics,
        "price_source": "Yahoo Finance daily data",
        "verification_note": metrics.get(
            "entry_source",
            "The selected date is user-entered and is not independently verified by this tool.",
        ),
    }.items():
        output.write(f"# {key}: {value}\n")
    chart.rename_axis("date").to_csv(output)
    return output.getvalue().encode("utf-8")


def share_text(ticker: str, metrics: dict, chart: pd.DataFrame) -> str:
    symbol = metrics.get("price_symbol", "₹")
    return (
        f"{ticker} — the call, in numbers\n"
        f"Published entry date: {metrics['requested_start']}\n"
        f"First tradable session: {metrics['ticker_entry_date']}\n"
        f"Data through: {metrics['as_of']}\n"
        f"{metrics.get('endpoint_label', 'Latest LTP')}: {symbol}{metrics['endpoint_price']:,.2f} ({metrics['endpoint_as_of']})\n"
        f"₹100 became ₹{100 * (1 + metrics['ticker_return']):.2f}\n"
        f"Adjusted return: {percent(metrics['ticker_return'])}\n"
        f"{BENCHMARK_LABEL} return: {percent(metrics['benchmark_return'])}\n"
        f"{WORLD_LABEL} return: {percent(metrics['world_return'])}\n"
        f"Outperformance vs Nifty: {percentage_points(metrics['excess_return'])}\n"
        f"Outperformance vs world: {percentage_points(metrics['excess_world_return'])}\n"
        f"Evidence ID: {evidence_fingerprint(ticker, metrics, chart)[:16].upper()}\n"
        "Source: immutable public publication plus Yahoo Finance market data. "
        "Historical result, not a current recommendation."
    )


def build_card_batch(feed: dict) -> bytes:
    """Generate one PNG and caption per security from a card-feed export."""
    securities = feed.get("securities") or []
    if feed.get("schema") != "public-portfolio-card-feed":
        raise ValueError("Unsupported card-feed schema.")
    if not securities:
        raise ValueError("The card feed contains no securities.")
    output = BytesIO()
    summaries = []
    now = datetime.now(IST)
    bucket = now.replace(second=0, microsecond=0)
    bucket = bucket.replace(minute=bucket.minute // 5 * 5)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in securities:
            ticker = clean_ticker(str(item.get("ticker", "")))
            start = date.fromisoformat(str(item.get("entry_date") or item["publication_date"])[:10])
            status = str(item.get("status", "active")).lower()
            requested_end = date.today()
            if status == "removed" and item.get("exit_date"):
                requested_end = date.fromisoformat(str(item["exit_date"])[:10])
            # Fetch a small cushion after an exit date so Yahoo can provide a
            # same-date benchmark/session even when the exit falls on a gap;
            # all series are trimmed back to the true endpoint below.
            fetch_end = requested_end + timedelta(days=7) if status == "removed" else requested_end
            ticker_history = load_daily_history(ticker, start, fetch_end, str(bucket))
            now_for_filter = datetime.now(IST)
            if ticker.endswith((".NS", ".BO")):
                ticker_history = completed_nse_history(ticker_history, now_for_filter)
            else:
                ticker_history = completed_us_history(ticker_history, now_for_filter)
            benchmark_history = completed_nse_history(
                load_daily_history(BENCHMARK_TICKER, start, fetch_end, str(bucket)), now_for_filter
            )
            world_history = completed_us_history(
                load_daily_history(WORLD_TICKER, start, fetch_end, str(bucket)), now_for_filter
            )
            fx_history = completed_us_history(
                load_daily_history(FX_TICKER, start, fetch_end, str(bucket)), now_for_filter
            )
            if status == "removed":
                ticker_history = ticker_history[ticker_history.index.date <= requested_end]
                benchmark_history = benchmark_history[benchmark_history.index.date <= requested_end]
                world_history = world_history[world_history.index.date <= requested_end]
                fx_history = fx_history[fx_history.index.date <= requested_end]
            try:
                metrics, chart = analyze(
                    ticker_history,
                    benchmark_history,
                    world_history,
                    fx_history,
                    start,
                    ticker_currency="INR" if ticker.endswith((".NS", ".BO")) else "USD",
                )
            except ValueError as exc:
                raise ValueError(
                    f"{ticker}: {exc} (entry {start.isoformat()}, "
                    f"endpoint {requested_end.isoformat()})"
                ) from exc
            if status == "removed":
                metrics["endpoint_label"] = "Exit close"
                metrics["endpoint_price"] = float(metrics["latest_close"])
                metrics["endpoint_as_of"] = metrics["as_of"]
            else:
                ltp, ltp_as_of = load_latest_ltp(ticker)
                if ltp is None:
                    raise ValueError(f"No intraday LTP returned for {ticker}.")
                metrics["endpoint_label"] = "Latest LTP"
                metrics["endpoint_price"] = float(ltp)
                metrics["endpoint_as_of"] = ltp_as_of or "timestamp unavailable"
            metrics["ltp"] = metrics["endpoint_price"]
            metrics["ltp_as_of"] = metrics["endpoint_as_of"]
            metrics["entry_source"] = "Immutable public portfolio publication"
            summaries.append({
                "ticker": ticker,
                "entry_date": start.isoformat(),
                "exit_date": (
                    str(item.get("exit_date"))[:10]
                    if status == "removed" and item.get("exit_date")
                    else None
                ),
                "data_through": metrics["as_of"],
                "return": metrics["ticker_return"],
                "status": status,
            })
            safe = ticker.replace("^", "").replace("/", "-")
            folder = "02-exited" if status == "removed" else "01-holdings"
            archive.writestr(
                f"{folder}/{safe}-card.png",
                whatsapp_card(ticker, metrics, chart),
            )
            archive.writestr(
                f"{folder}/{safe}-caption.txt",
                share_text(ticker, metrics, chart),
            )
        ordered = sorted(summaries, key=lambda row: row["return"])
        finished = [row for row in summaries if row["status"] == "removed"]
        active = [row for row in summaries if row["status"] != "removed"]
        finished_returns = [float(row["return"]) for row in finished]
        active_returns = [float(row["return"]) for row in active]
        recent_exits = exited_symbol_rows(feed, lookback_days=90)
        annual_exits = exited_symbol_rows(feed, lookback_days=365)
        all_dated_exits = exited_symbol_rows(feed, lookback_days=None)
        archive.writestr(
            "02-exited/remove-from-universal-last-90-days.txt",
            exited_symbols_text(recent_exits),
        )
        archive.writestr(
            "02-exited/exited-last-90-days.csv",
            exited_symbols_csv(recent_exits),
        )
        archive.writestr(
            "02-exited/exited-last-365-days-review.csv",
            exited_symbols_csv(annual_exits),
        )
        archive.writestr(
            "02-exited/exited-all-dated.csv",
            exited_symbols_csv(all_dated_exits),
        )
        archive.writestr(
            "03-analysis/exited-securities-return-summary.json",
            json.dumps({
                "definition": "Empirical realized-return summary of removed securities; not a guaranteed forecast.",
                "finished_trade_count": len(finished_returns),
                "mean_realized_return": (sum(finished_returns) / len(finished_returns)) if finished_returns else None,
                "median_realized_return": (float(np.median(finished_returns)) if finished_returns else None),
                "mean_unrealized_return": (sum(active_returns) / len(active_returns)) if active_returns else None,
                "median_unrealized_return": (float(np.median(active_returns)) if active_returns else None),
                "active_count": len(active_returns),
                "finished_trades": finished,
            }, indent=2, sort_keys=True).encode(),
        )
        archive.writestr(
            "03-analysis/outlier-summary.json",
            json.dumps({
                "highest_loss": ordered[0] if ordered else None,
                "highest_gain": ordered[-1] if ordered else None,
                "security_count": len(summaries),
            }, indent=2, sort_keys=True).encode(),
        )
        manifest = {
            "schema": "public-portfolio-card-archive",
            "schema_version": 1,
            "basket_id": feed.get("basket_id"),
            "publication_id": feed.get("publication_id"),
            "portfolio_version": feed.get("portfolio_version"),
            "generated_at": datetime.now(IST).isoformat(),
            "holdings": sorted(row["ticker"] for row in active),
            "exited": sorted(row["ticker"] for row in finished),
            "holding_count": len(active),
            "exited_count": len(finished),
        }
        archive.writestr(
            "00-summary/manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True).encode(),
        )
        archive.writestr(
            "00-summary/README.txt",
            (
                "PUBLIC PORTFOLIO TRACK-RECORD CARDS\n\n"
                "01-holdings  Current portfolio securities.\n"
                "02-exited    Securities removed from the portfolio.\n"
                "03-analysis  Aggregate return and outlier summaries.\n\n"
                "Paste 02-exited/remove-from-universal-last-90-days.txt into\n"
                "'Remove symbols from universal portfolio'. The rolling 90-day\n"
                "list is the default anti-churn cooldown. The 365-day CSV is a\n"
                "review list, not a recommended automatic blacklist.\n\n"
                "Each security has a PNG share card and matching TXT caption.\n"
                "Historical results are not investment advice or guaranteed forecasts.\n"
            ).encode(),
        )
        archive.writestr(
            "00-summary/portfolio-summary.png",
            batch_summary_card(feed, summaries),
        )
    return output.getvalue()


def batch_summary_card(feed: dict, summaries: list[dict]) -> bytes:
    """Render a human-readable summary card matching the individual-card style."""
    paper, ink, muted, accent = "#f5f0e6", "#29251f", "#6b665e", "#9f4339"
    ordered = sorted(summaries, key=lambda row: row["return"])
    loss = ordered[0] if ordered else None
    gain = ordered[-1] if ordered else None
    finished = [row["return"] for row in summaries if row["status"] == "removed"]
    active = [row["return"] for row in summaries if row["status"] != "removed"]
    mean_finished = sum(finished) / len(finished) if finished else None
    mean_active = sum(active) / len(active) if active else None
    fig = plt.figure(figsize=(8, 12), dpi=135, facecolor=paper)
    fig.patches.append(plt.Rectangle((.035,.025),.93,.95,transform=fig.transFigure,facecolor="none",edgecolor=ink,linewidth=1.2))
    fig.text(.09,.91,"PORTFOLIO",fontsize=28,fontweight="bold",family="serif",color=ink)
    fig.text(.09,.865,"CARD SUMMARY",fontsize=28,fontweight="bold",family="serif",color=ink)
    fig.text(.09,.82,f"{feed.get('portfolio_version','')} · {feed.get('publication_date','')}",fontsize=14,family="sans",color=muted)
    y=.70
    for label,row,color in [("HIGHEST GAIN",gain,accent),("HIGHEST LOSS",loss,ink)]:
        fig.text(.09,y,label,fontsize=13,fontweight="bold",color=muted)
        fig.text(.09,y-.075,row['ticker'] if row else "No data",fontsize=24,fontweight="bold",color=color)
        fig.text(.09,y-.125,f"{row['return']:+.2%}" if row else "N/A",fontsize=21,fontweight="bold",color=color)
        y-=.22
    fig.text(.09,.255,"REALIZED RETURN",fontsize=13,fontweight="bold",color=muted)
    fig.text(.09,.205,f"{mean_finished:+.2%}" if mean_finished is not None else "N/A",fontsize=24,fontweight="bold",color=accent)
    fig.text(.09,.16,f"{len(finished)} exited",fontsize=13,fontweight="bold",color=ink)
    fig.text(.55,.255,"UNREALIZED RETURN",fontsize=13,fontweight="bold",color=muted)
    fig.text(.55,.205,f"{mean_active:+.2%}" if mean_active is not None else "N/A",fontsize=24,fontweight="bold",color=accent)
    fig.text(.55,.16,f"{len(active)} active",fontsize=13,fontweight="bold",color=ink)
    fig.text(.09,.065,"Empirical realized returns · not a guaranteed forecast",fontsize=11,color=muted,style="italic")
    buf=BytesIO(); fig.savefig(buf,format="png",facecolor=paper,bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def portfolio_cover_card(
    feed: dict,
    securities: list[dict],
    *,
    scope_label: str,
) -> bytes:
    """Render a zero-network cover slide for the public evidence deck."""
    paper, ink, muted, accent, faint = (
        "#f5f0e6",
        "#29251f",
        "#625d55",
        "#913f36",
        "#d8cfbf",
    )
    serif, sans, mono = "DejaVu Serif", "DejaVu Sans", "DejaVu Sans Mono"
    current_scope = scope_label.lower().startswith("current")
    title = "CURRENT HOLDINGS" if current_scope else "EXITED SECURITIES"
    ordered = (
        sorted(
            securities,
            key=lambda item: (-float(item.get("target_weight") or 0), item["ticker"]),
        )
        if current_scope
        else sorted(
            securities,
            key=lambda item: (str(item.get("exit_date") or ""), item["ticker"]),
            reverse=True,
        )
    )
    shown = ordered[:5]

    figure = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=paper)
    figure.patches.extend(
        [
            plt.Rectangle(
                (0.035, 0.028),
                0.930,
                0.944,
                transform=figure.transFigure,
                facecolor="none",
                edgecolor=ink,
                linewidth=1.3,
            ),
            plt.Rectangle(
                (0.044, 0.037),
                0.912,
                0.926,
                transform=figure.transFigure,
                facecolor="none",
                edgecolor=faint,
                linewidth=0.8,
            ),
        ]
    )
    figure.text(0.075, 0.930, "PUBLIC PORTFOLIO", color=accent, fontsize=18,
                fontweight="bold", family=sans)
    figure.text(0.075, 0.790, "Track-record\nevidence deck", color=ink,
                fontsize=43, fontweight="bold", family=serif, linespacing=1.12)
    figure.text(
        0.075,
        0.715,
        f"{feed.get('portfolio_version', '')}  ·  published {feed.get('publication_date', '')}",
        color=muted,
        fontsize=17,
        family=sans,
    )
    figure.lines.append(
        plt.Line2D([0.075, 0.925], [0.680, 0.680], transform=figure.transFigure,
                   color=ink, linewidth=1.0)
    )
    figure.text(0.075, 0.635, title, color=muted, fontsize=14,
                fontweight="bold", family=sans)
    figure.text(0.075, 0.560, str(len(securities)), color=accent, fontsize=50,
                fontweight="bold", family=serif)
    figure.text(0.215, 0.570, "securities in this deck", color=ink, fontsize=20,
                fontweight="bold", family=sans)

    y = 0.495
    for item in shown:
        ticker = str(item.get("ticker", ""))
        if current_scope:
            detail = f"{float(item.get('target_weight') or 0):.0%} target"
        else:
            detail = f"exited {str(item.get('exit_date') or 'date unavailable')}"
        figure.text(0.095, y, ticker, color=ink, fontsize=18,
                    fontweight="bold", family=mono)
        figure.text(0.905, y, detail, color=muted, fontsize=15,
                    ha="right", family=sans)
        figure.lines.append(
            plt.Line2D([0.095, 0.905], [y - 0.018, y - 0.018],
                       transform=figure.transFigure, color=faint, linewidth=0.7)
        )
        y -= 0.055
    remaining = len(ordered) - len(shown)
    if remaining > 0:
        figure.text(0.095, y, f"+ {remaining} more", color=muted, fontsize=15,
                    family=sans, style="italic")

    figure.text(0.075, 0.125, "TURN THE PAGE", color=accent, fontsize=13,
                fontweight="bold", family=sans)
    figure.text(
        0.075,
        0.072,
        "One immutable entry date. One security per slide.\n"
        "Nifty 50 and VT world shown in INR for context.",
        color=ink,
        fontsize=15,
        family=serif,
        linespacing=1.45,
    )
    figure.text(0.925, 0.048, str(feed.get("publication_id", ""))[-16:],
                color=muted, fontsize=9, ha="right", family=mono)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=100, facecolor=paper,
                   bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return buffer.getvalue()


def whatsapp_card(ticker: str, metrics: dict, chart: pd.DataFrame) -> bytes:
    """Render a glance-readable 1080×1350 humanistic share card."""
    paper = "#f5f0e6"
    ink = "#29251f"
    muted = "#625d55"
    faint = "#d8cfbf"
    oxblood = "#913f36"
    india = "#315f78"
    world = "#7a5310"
    negative = "#a52f38"
    serif = "DejaVu Serif"
    sans = "DejaVu Sans"
    mono = "DejaVu Sans Mono"

    evidence_id = evidence_fingerprint(ticker, metrics, chart)[:16].upper()
    ending_value = 100 * (1 + metrics["ticker_return"])
    nifty_value = 100 * (1 + metrics["benchmark_return"])
    world_value = 100 * (1 + metrics["world_return"])
    result_color = oxblood if metrics["ticker_return"] >= 0 else negative
    price_symbol = metrics.get("price_symbol", "₹")
    start_label = pd.Timestamp(metrics["requested_start"]).strftime("%d %b %Y")
    end_label = pd.Timestamp(metrics["as_of"]).strftime("%d %b %Y")

    def comparison(name: str, value: float) -> str:
        if value >= 0:
            return f"Beat {name} by {value * 100:.2f} pp"
        return f"Trailed {name} by {abs(value) * 100:.2f} pp"

    figure = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=paper)
    figure.patches.extend([
        plt.Rectangle(
            (0.035, 0.028), 0.930, 0.944, transform=figure.transFigure,
            facecolor="none", edgecolor=ink, linewidth=1.3,
        ),
        plt.Rectangle(
            (0.044, 0.037), 0.912, 0.926, transform=figure.transFigure,
            facecolor="none", edgecolor=faint, linewidth=0.8,
        ),
    ])

    # The first screenful answers who, when and what happened.
    ticker_size = max(28, min(40, 49 - len(ticker) * 0.75))
    figure.text(0.075, 0.925, ticker, color=ink, fontsize=ticker_size,
                fontweight="bold", family=serif)
    figure.text(
        0.075, 0.875,
        f"{start_label}  →  {end_label}",
        color=muted, fontsize=18, family=sans,
    )
    figure.text(
        0.925, 0.875,
        f"{metrics['calendar_days']} calendar days  ·  "
        f"{metrics['ticker_sessions']} market sessions",
        color=muted, fontsize=17, ha="right", family=sans,
    )
    figure.text(
        0.075, 0.852,
        f"Daily adjusted-close series through {end_label}",
        color=muted, fontsize=11.5, family=sans,
    )
    figure.lines.extend([
        plt.Line2D([0.075, 0.925], [0.845, 0.845],
                   transform=figure.transFigure, color=ink, linewidth=1.1),
        plt.Line2D([0.075, 0.925], [0.838, 0.838],
                   transform=figure.transFigure, color=faint, linewidth=0.8),
    ])

    figure.text(0.075, 0.785, "What became of ₹100?",
                color=ink, fontsize=32, family=serif, style="italic")
    figure.text(0.075, 0.700, f"₹{ending_value:.2f}",
                color=result_color, fontsize=60, fontweight="bold",
                family=serif)
    direction = "gain" if metrics["ticker_return"] >= 0 else "loss"
    figure.text(
        0.080, 0.660,
        f"{abs(metrics['ticker_return']):.2%} {direction}",
        color=ink, fontsize=19, fontweight="bold", family=sans,
    )

    # Entry and endpoint prices are deliberately prominent. For removed
    # holdings the endpoint is the actual exit close; active holdings use LTP.
    figure.lines.append(plt.Line2D(
        [0.075, 0.925], [0.626, 0.626],
        transform=figure.transFigure, color=faint, linewidth=1.0,
    ))
    figure.text(0.080, 0.596, "ENTRY CLOSE", color=muted, fontsize=13,
                fontweight="bold", family=sans)
    figure.text(0.080, 0.555, f"{price_symbol}{metrics['entry_close']:,.2f}",
                color=ink, fontsize=26, fontweight="bold", family=serif)
    figure.text(0.555, 0.596, str(metrics.get("endpoint_label", "LATEST LTP")).upper(), color=muted,
                fontsize=13, fontweight="bold", family=sans)
    figure.text(0.555, 0.555, f"{price_symbol}{metrics['endpoint_price']:,.2f}",
                color=ink, fontsize=26, fontweight="bold", family=serif)
    figure.text(0.555, 0.525, f"{metrics['endpoint_as_of']} · price-only snapshot", color=muted,
                fontsize=10.5, family=sans)
    endpoint_price_only = 100 * metrics["endpoint_price"] / metrics["entry_close"]
    figure.text(0.555, 0.500, f"₹100 at endpoint (price-only): ₹{endpoint_price_only:.2f}",
                color=muted, fontsize=10.5, family=sans)

    # A large, direct-labelled chart can be read without a legend.
    axis = figure.add_axes([0.085, 0.285, 0.79, 0.235], facecolor=paper)
    series = [
        (ticker, chart["Ticker"].dropna(), oxblood, 3.2),
        ("Nifty 50", chart[BENCHMARK_LABEL].dropna(), india, 2.2),
        ("VT world · INR", chart[WORLD_LABEL].dropna(), world, 2.2),
    ]
    show_points = len(chart.index) <= 20
    for name, values, color, width in series:
        axis.plot(
            values.index, values, color=color, linewidth=width,
            solid_capstyle="round", marker="o" if show_points else None,
            markersize=4.2 if show_points else 0,
            markerfacecolor=paper, markeredgewidth=1.2,
        )
    axis.axhline(100, color=muted, linewidth=0.9, linestyle=(0, (3, 3)),
                 alpha=0.65)
    axis.grid(axis="y", color=faint, linewidth=0.7, alpha=0.8)
    axis.spines[:].set_visible(False)
    axis.tick_params(colors=muted, labelsize=11, length=0)
    observed_dates = pd.DatetimeIndex(chart.index.unique()).sort_values()
    tick_count = min(3, len(observed_dates))
    tick_indices = np.linspace(0, len(observed_dates) - 1,
                               tick_count, dtype=int)
    axis.set_xticks(observed_dates[tick_indices])
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))

    all_values = pd.concat([item[1] for item in series])
    y_span = max(float(all_values.max() - all_values.min()), 1.0)
    axis.set_ylim(float(all_values.min()) - y_span * 0.13,
                  float(all_values.max()) + y_span * 0.15)
    total_days = max((chart.index.max() - chart.index.min()).days, 1)
    axis.set_xlim(
        chart.index.min(),
        chart.index.max() + timedelta(days=max(int(total_days * 0.22), 6)),
    )

    labels = sorted(
        [(name, values.index[-1], float(values.iloc[-1]), color)
         for name, values, color, _ in series],
        key=lambda item: item[2],
    )
    minimum_gap = y_span * 0.085
    levels: list[float] = []
    for _, _, value, _ in labels:
        level = value
        if levels and level - levels[-1] < minimum_gap:
            level = levels[-1] + minimum_gap
        levels.append(level)
    for (name, last_date, value, color), level in zip(labels, levels):
        axis.annotate(
            f"{name}  {value:.1f}", xy=(last_date, value),
            xytext=(last_date + timedelta(days=max(int(total_days * 0.035), 1)),
                    level),
            color=color, fontsize=11.5, fontweight="bold", va="center",
            family=sans,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
            annotation_clip=False,
        )

    # Only the decision-relevant comparison remains on the share image.
    figure.text(0.075, 0.242, "THE SAME ₹100 IN THE BENCHMARKS",
                color=muted, fontsize=13, fontweight="bold", family=sans)
    figure.text(0.075, 0.202,
                f"Nifty 50  ₹{nifty_value:.2f}",
                color=india, fontsize=19, fontweight="bold", family=serif)
    figure.text(0.555, 0.202,
                f"VT world · INR  ₹{world_value:.2f}",
                color=world, fontsize=19, fontweight="bold", family=serif)
    figure.text(
        0.075, 0.160,
        f"{comparison('Nifty', metrics['excess_return'])}  ·  "
        f"{comparison('world', metrics['excess_world_return'])}",
        color=ink, fontsize=15.5, fontweight="bold", family=sans,
    )

    figure.lines.append(plt.Line2D(
        [0.075, 0.925], [0.125, 0.125],
        transform=figure.transFigure, color=faint, linewidth=0.9,
    ))
    figure.text(
        0.075, 0.096,
        "Daily adjusted closes · overseas returns and VT translated to INR · ticker LTP shown above",
        color=muted, fontsize=13, family=sans,
    )
    figure.text(
        0.075, 0.066,
        "Immutable publication date · historical result, not advice.",
        color=muted, fontsize=12.5, family=serif, style="italic",
    )
    figure.text(0.925, 0.066, evidence_id, color=muted, fontsize=9,
                ha="right", family=mono)

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=100, facecolor=paper,
                   bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return buffer.getvalue()
