"""
Wealth Machine dashboard — corrected version.

Key corrections:
- Streamlit page configuration is now the first Streamlit command.
- MongoDB credentials are read from Streamlit secrets or environment variables.
- MMI rows are validated, de-duplicated and sorted before modelling.
- The active MMI streak is treated as right-censored, not as a completed streak.
- Flip timing uses a conditional Kaplan-Meier residual-life estimate.
- Trading-session estimates are converted to business dates instead of calendar dates.
- The dashboard distinguishes sentiment-regime estimates from market-price forecasts.
- Kite previous-close values are no longer described as real-time prices.
- Profit-booking targets are calculated from average cost, not current price.
- Holding-period dates use trading sessions and avoid weekend exit dates.

This dashboard is a decision-support tool, not financial advice.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import random
from io import BytesIO
from itertools import groupby
from typing import Any, Iterable, Optional

import numpy as np
import openpyxl
import pandas as pd
import plotly.express as px
import pytz
import streamlit as st
import yfinance as yf
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from scipy import stats

# This must be the first Streamlit command in the script.
st.set_page_config(
    page_title="Wealth Machine",
    layout="wide",
    page_icon="💰",
)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
IST = pytz.timezone("Asia/Kolkata")
GROWTH_RATE = 0.04  # 4% per market day. The original value 0.456 meant 45.6%.
DEFAULT_PROFIT_BOOKING_THRESHOLD = 7.2
DEFAULT_TARGET_NET_DAILY_PCT = 0.28
DEFAULT_LAMF_CAP = 0.30  # Fraction of mutual-fund corpus; 0.30 means 30%.
MMI_NEUTRAL_LEVEL = 50.0


def now_ist() -> datetime.datetime:
    """Return the current timezone-aware IST datetime."""
    return datetime.datetime.now(IST)


def today_ist() -> datetime.date:
    """Return the current IST date."""
    return now_ist().date()


def add_business_sessions(
    from_date: datetime.date,
    sessions: int,
) -> datetime.date:
    """Add weekday business sessions to a date.

    This skips Saturdays and Sundays. Exchange holidays are not inferred, so the
    result is an approximation unless an exchange calendar is added later.
    """
    sessions = max(0, int(sessions))
    return (pd.Timestamp(from_date) + pd.offsets.BDay(sessions)).date()


def get_next_trading_day(from_date: datetime.date) -> datetime.date:
    """Return the next weekday trading date approximation."""
    return add_business_sessions(from_date, 1)


def get_market_status(now: datetime.datetime) -> str:
    """Return an approximate NSE market status based on IST time and weekday."""
    if now.tzinfo is None:
        now = IST.localize(now)
    else:
        now = now.astimezone(IST)

    if now.weekday() >= 5:
        return "market_closed"

    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if now < market_open:
        return "pre_market"
    if now >= market_close:
        return "after_market_close"
    return "market_hours"


def is_market_closed() -> bool:
    """Return whether the NSE is approximately closed based on IST schedule.

    The previous implementation inferred closure from two equal five-minute
    prices. That can classify a quiet open market as closed and can read stale
    bars on weekends.
    """
    return get_market_status(now_ist()) != "market_hours"


# -----------------------------------------------------------------------------
# OPTIONAL MONGODB CONNECTION
# -----------------------------------------------------------------------------
def _get_mongodb_uri() -> Optional[str]:
    """Read MongoDB URI from Streamlit secrets or environment variables."""
    try:
        secret_uri = st.secrets.get("MONGODB_URI")
        if secret_uri:
            return str(secret_uri)
    except Exception:
        pass
    return os.getenv("MONGODB_URI")


MONGODB_URI = _get_mongodb_uri()
client: Optional[MongoClient] = None
db = None
collection = None
mmi_collection = None
allocation_collection = None
mongo_error: Optional[str] = None

if MONGODB_URI:
    try:
        client = MongoClient(
            MONGODB_URI,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        client.admin.command("ping")
        db = client["finance_db"]
        collection = db["sell_plan_params"]
        mmi_collection = db["mmi_data"]
        allocation_collection = db["allocation_plans"]
    except Exception as exc:
        mongo_error = str(exc)
else:
    mongo_error = "MONGODB_URI is not configured. Database features are disabled."


# -----------------------------------------------------------------------------
# PORTFOLIO ALLOCATION VISUALISATION
# -----------------------------------------------------------------------------
allocation_data = {
    "Asset": [
        "Gold",
        "Silver",
        "Bitcoin",
        "Bonds, Liquid Fund",
        "Cash, Savings Account",
        "Indian Equity",
    ],
    "Allocation (%)": [3.74, 3.74, 3.74, 24.5, 5.00, 59.29],
}
allocation_visual_df = pd.DataFrame(allocation_data)

custom_colorscale = [
    [0.0, "magenta"],
    [0.5, "yellow"],
    [1.0, "green"],
]

allocation_fig = px.treemap(
    allocation_visual_df,
    path=["Asset"],
    values="Allocation (%)",
    color="Allocation (%)",
    color_continuous_scale=custom_colorscale,
    title="Portfolio Allocation Treemap",
)
allocation_fig.update_traces(
    texttemplate="<b>%{label}</b><br>%{value:.2f}%",
    textposition="middle center",
    insidetextfont=dict(size=18, color="white"),
)

with st.expander("📊 Portfolio Allocation Visualization", expanded=False):
    st.plotly_chart(allocation_fig, use_container_width=True)
    total_allocation = allocation_visual_df["Allocation (%)"].sum()
    if not np.isclose(total_allocation, 100.0, atol=0.05):
        st.warning(f"Allocation totals {total_allocation:.2f}%, not 100%.")


# -----------------------------------------------------------------------------
# STORED PARAMETERS
# -----------------------------------------------------------------------------
def get_max_roi_from_file() -> float:
    """Read the configured profit-booking threshold from max_roi.json."""
    try:
        with open("max_roi.json", "r", encoding="utf-8") as file:
            data = json.load(file)
            value = float(data.get("max_roi", 0.0))
            return value if np.isfinite(value) and value > 0 else 0.0
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        return 0.0


min_threshold = get_max_roi_from_file()
active_profit_threshold = (
    min_threshold if min_threshold > 0 else DEFAULT_PROFIT_BOOKING_THRESHOLD
)


def save_input_params(user_id: str, net_pl: float, charges: float, target_pct: float) -> None:
    """Persist the user's profit-booking inputs when MongoDB is available."""
    if collection is None:
        return
    record = {
        "user_id": user_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "net_pl": float(net_pl),
        "charges": float(charges),
        "target_pct": float(target_pct),
    }
    collection.insert_one(record)


def get_latest_input_params(user_id: str) -> dict[str, float]:
    """Load the user's latest profit-booking inputs."""
    defaults = {
        "net_pl": 0.0,
        "charges": 0.0,
        "target_pct": DEFAULT_TARGET_NET_DAILY_PCT,
    }
    if collection is None:
        return defaults
    try:
        latest = collection.find_one(
            {"user_id": user_id},
            sort=[("timestamp", -1)],
        )
        return latest if latest else defaults
    except Exception:
        return defaults


# -----------------------------------------------------------------------------
# LEVERAGE DECISION
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900)
def should_use_leverage(
    ticker: str = "^NSEI",
    days: int = 200,
    cap: float = DEFAULT_LAMF_CAP,
) -> dict[str, Any]:
    """Assess whether a trend filter permits limited leverage.

    `cap` is a fraction of the mutual-fund corpus, not a multiple. A cap of 0.30
    means the recommendation cannot exceed 30% of the corpus.
    """
    try:
        data = yf.download(
            ticker,
            period="500d",
            group_by="column",
            auto_adjust=False,
            progress=False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data is None or data.empty:
            raise ValueError(f"No price data downloaded for ticker {ticker}.")

        if "Close" not in data.columns:
            if "Adj Close" in data.columns:
                data["Close"] = data["Adj Close"]
            else:
                raise ValueError(
                    f"'Close' column not found. Available: {list(data.columns)}"
                )

        data.index = pd.to_datetime(data.index, errors="coerce")
        data = data.sort_index().dropna(subset=["Close"])
        if len(data) < days:
            raise ValueError(
                f"Insufficient rows for {days}-day MA; received {len(data)} rows."
            )

        close_series = pd.to_numeric(data["Close"], errors="coerce").dropna()
        ma_series = close_series.rolling(window=days, min_periods=days).mean()

        trend_df = pd.DataFrame({"Close": close_series, "200_MA": ma_series})
        trend_df["pct_above_ma"] = (
            trend_df["Close"] - trend_df["200_MA"]
        ) / trend_df["200_MA"]
        valid = trend_df.dropna(subset=["200_MA", "pct_above_ma"])
        if valid.empty:
            raise ValueError("No valid rows remain after calculating the moving average.")

        latest = valid.iloc[-1]
        latest_close = float(latest["Close"])
        latest_ma = float(latest["200_MA"])
        pct_above_ma = float(latest["pct_above_ma"])

        positive_distances = valid.loc[
            valid["pct_above_ma"] > 0, "pct_above_ma"
        ]
        reference_pct = (
            float(positive_distances.quantile(0.95))
            if not positive_distances.empty
            else 0.10
        )
        reference_pct = max(reference_pct, 0.01)

        return {
            "should_leverage": latest_close > latest_ma,
            "latest_close": round(latest_close, 2),
            "ma_value": round(latest_ma, 2),
            "pct_above_ma": pct_above_ma,
            # Kept for compatibility with the existing UI/function consumers.
            "max_pct_above_ma": reference_pct,
            "reference_pct_above_ma": reference_pct,
            # Kept under the existing key. It is now a reference band rather
            # than the original unsafe cap/max slope.
            "alpha": reference_pct,
            "cap": float(np.clip(cap, 0.0, 1.0)),
            "data_date": valid.index[-1].date(),
        }
    except Exception as exc:
        return {
            "should_leverage": False,
            "latest_close": None,
            "ma_value": None,
            "pct_above_ma": None,
            "max_pct_above_ma": None,
            "reference_pct_above_ma": None,
            "alpha": 0.10,
            "cap": float(np.clip(cap, 0.0, 1.0)),
            "error": str(exc),
        }


def compute_lamf_pct(
    pct_above_ma: float,
    mmi: float,
    alpha: float,
    cap: float = DEFAULT_LAMF_CAP,
) -> float:
    """Calculate a capped, risk-aware LAMF fraction.

    Leverage is zero below the moving average. Above it, the recommendation is
    scaled by trend strength, reduced as MMI moves into greed, and penalised when
    price is unusually far above the moving average.
    """
    if pct_above_ma is None or not np.isfinite(pct_above_ma) or pct_above_ma <= 0:
        return 0.0

    mmi = float(np.clip(mmi, 0.0, 100.0))
    cap = float(np.clip(cap, 0.0, 1.0))
    reference_band = max(float(alpha), 0.01)

    trend_strength = float(np.clip(pct_above_ma / reference_band, 0.0, 1.0))
    sentiment_factor = float(np.clip((60.0 - mmi) / 60.0, 0.0, 1.0))

    # Reduce leverage when the index is more than 10% above its 200-DMA.
    overextension = max(pct_above_ma - 0.10, 0.0)
    overextension_penalty = float(np.exp(-overextension / 0.05))

    lamf_pct = cap * trend_strength * sentiment_factor * overextension_penalty
    return float(np.clip(lamf_pct, 0.0, cap))


with st.expander("⚖️ Leverage Decision Based on NIFTY 200-Day MA", expanded=False):
    leverage_result = should_use_leverage()

    if leverage_result.get("error"):
        st.error(f"⚠️ Error fetching data: {leverage_result['error']}")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("📈 NIFTY Close", f"{leverage_result['latest_close']:,.2f}")
        col_b.metric("📊 200-Day MA", f"{leverage_result['ma_value']:,.2f}")
        col_c.metric(
            "📏 95th percentile above MA",
            f"{leverage_result['reference_pct_above_ma'] * 100:.2f}%",
        )

        if leverage_result["should_leverage"]:
            st.success("✅ NIFTY is above its 200-day MA; limited leverage may be considered.")
            leverage_mmi = st.number_input(
                "📊 Market Mood Index (MMI)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                key="leverage_mmi",
            )
            mf_corpus = st.number_input(
                "💼 Enter Mutual Fund Corpus (₹)",
                min_value=0.0,
                value=18_00_000.0,
                step=10_000.0,
            )
            lamf_pct = compute_lamf_pct(
                leverage_result["pct_above_ma"],
                leverage_mmi,
                leverage_result["alpha"],
                cap=leverage_result["cap"],
            )
            lamf_amount = mf_corpus * lamf_pct
            metric_a, metric_b = st.columns(2)
            metric_a.metric("📌 LAMF % Recommended", f"{lamf_pct * 100:.2f}%")
            metric_b.metric("💸 Maximum LAMF Amount", f"₹{lamf_amount:,.0f}")
            st.caption(
                "Trend and sentiment are filters, not guarantees. Confirm lender margin, "
                "interest cost, drawdown tolerance and liquidation risk before borrowing."
            )
        else:
            st.warning("🛑 NIFTY is below its 200-day MA; this model recommends no leverage.")


st.subheader("📊 Stock Holdings Analysis & Market Mood Dashboard")


# -----------------------------------------------------------------------------
# MARKET MOOD ANALYSIS
# -----------------------------------------------------------------------------
class MarketMoodAnalyzer:
    """Analyse binary Fear/Greed MMI streaks and estimate regime duration.

    The model estimates when MMI may cross the binary threshold of 50. It does
    not forecast NIFTY direction, returns, volatility or a market correction.
    """

    def __init__(self, mmi_data: Any):
        if isinstance(mmi_data, bytes):
            self.df = self._prepare_mmi_data_from_bytes(mmi_data)
        elif isinstance(mmi_data, pd.DataFrame):
            self.df = self._prepare_mmi_data_from_df(mmi_data)
        elif hasattr(mmi_data, "read"):
            self.df = self._prepare_mmi_data_from_bytes(mmi_data.read())
        else:
            raise ValueError(f"Unsupported MMI input type: {type(mmi_data)}")

        self.df = self.df.sort_values("Date").reset_index(drop=True)
        if self.df.empty:
            raise ValueError("MMI dataset contains no valid rows.")

        self.mmi_last_date = self.df["Date"].iloc[-1].date()
        self.today_date = today_ist()
        self.current_mmi = float(self.df["MMI"].iloc[-1])
        self.current_mood = "Fear" if self.current_mmi <= MMI_NEUTRAL_LEVEL else "Greed"
        self.current_streak = self._get_current_streak_length()
        self.streak_records = self._build_streak_records()
        self.run_lengths = self._identify_mood_streaks()

    def _prepare_mmi_data_from_bytes(self, mmi_bytes: bytes) -> pd.DataFrame:
        if not mmi_bytes:
            raise ValueError("Empty MMI file provided.")
        dataframe = pd.read_csv(BytesIO(mmi_bytes))
        return self._process_dataframe(dataframe)

    def _prepare_mmi_data_from_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return self._process_dataframe(dataframe.copy())

    def _process_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalise, validate, sort and de-duplicate MMI data."""
        if dataframe is None or dataframe.empty:
            raise ValueError("MMI dataset is empty.")

        dataframe = dataframe.copy()
        dataframe.columns = [str(column).strip() for column in dataframe.columns]
        lower_map = {column.lower(): column for column in dataframe.columns}

        date_column = next(
            (lower_map[key] for key in lower_map if "date" in key),
            dataframe.columns[0] if len(dataframe.columns) >= 1 else None,
        )
        mmi_column = next(
            (
                lower_map[key]
                for key in lower_map
                if "market mood" in key or key == "mmi" or key.startswith("mmi ")
            ),
            dataframe.columns[1] if len(dataframe.columns) >= 2 else None,
        )
        nifty_column = next(
            (lower_map[key] for key in lower_map if "nifty" in key),
            dataframe.columns[2] if len(dataframe.columns) >= 3 else None,
        )

        if date_column is None or mmi_column is None:
            raise ValueError("Could not identify Date and MMI columns.")

        selected = pd.DataFrame(
            {
                "Date": dataframe[date_column],
                "MMI": dataframe[mmi_column],
                "Nifty": dataframe[nifty_column] if nifty_column else np.nan,
            }
        )
        selected["Date"] = pd.to_datetime(
            selected["Date"],
            dayfirst=True,
            errors="coerce",
        )
        selected["MMI"] = pd.to_numeric(selected["MMI"], errors="coerce")
        selected["Nifty"] = pd.to_numeric(selected["Nifty"], errors="coerce")
        selected = selected.dropna(subset=["Date", "MMI"])
        selected = selected[selected["MMI"].between(0.0, 100.0, inclusive="both")]
        selected["Date"] = selected["Date"].dt.normalize()
        selected = selected.sort_values("Date")
        selected = selected.drop_duplicates(subset=["Date"], keep="last")
        selected["Mood"] = np.where(
            selected["MMI"] <= MMI_NEUTRAL_LEVEL,
            "Fear",
            "Greed",
        )
        selected.reset_index(drop=True, inplace=True)

        if len(selected) < 10:
            raise ValueError("At least 10 valid MMI observations are required.")
        return selected

    def _count_trading_days(
        self,
        start_date: datetime.date,
        calendar_days: int,
    ) -> int:
        """Count weekday sessions in a calendar interval."""
        end_date = start_date + datetime.timedelta(days=calendar_days)
        sessions = pd.date_range(
            start=start_date + datetime.timedelta(days=1),
            end=end_date,
            freq="B",
        )
        return len(sessions)

    def _build_streak_records(self) -> list[dict[str, Any]]:
        """Create completed/censored run records.

        The last run is active and therefore right-censored. Treating it as a
        completed event biases the survival curve toward shorter durations.
        """
        records: list[dict[str, Any]] = []
        moods = self.df["Mood"].tolist()

        for mood, grouped_indexes in groupby(range(len(moods)), key=lambda i: moods[i]):
            indexes = list(grouped_indexes)
            start_index = indexes[0]
            end_index = indexes[-1]
            records.append(
                {
                    "mood": mood,
                    "duration": len(indexes),
                    "start_date": self.df.loc[start_index, "Date"].date(),
                    "end_date": self.df.loc[end_index, "Date"].date(),
                    "event_observed": 0 if end_index == len(self.df) - 1 else 1,
                }
            )
        return records

    def _identify_mood_streaks(self) -> dict[str, list[int]]:
        """Return completed historical run lengths by mood."""
        run_lengths: dict[str, list[int]] = {"Fear": [], "Greed": []}
        for record in self.streak_records:
            if record["event_observed"] == 1:
                run_lengths[record["mood"]].append(int(record["duration"]))
        return run_lengths

    def _get_current_streak_length(self) -> int:
        current_streak = 1
        for index in range(len(self.df) - 2, -1, -1):
            if self.df["Mood"].iloc[index] == self.current_mood:
                current_streak += 1
            else:
                break
        return current_streak

    @staticmethod
    def _empirical_survival_hazard(
        data: Iterable[int],
        event_observed: Optional[Iterable[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a Kaplan-Meier survival curve without an external dependency."""
        durations = np.asarray(list(data), dtype=int)
        if durations.size == 0:
            return np.array([0], dtype=int), np.array([1.0], dtype=float)
        if np.any(durations <= 0):
            raise ValueError("Streak durations must be positive integers.")

        events = (
            np.ones_like(durations, dtype=int)
            if event_observed is None
            else np.asarray(list(event_observed), dtype=int)
        )
        if events.shape != durations.shape:
            raise ValueError("event_observed must match the duration array.")

        timeline = [0]
        survival_probability = [1.0]
        survival = 1.0

        for time_value in sorted(np.unique(durations)):
            at_risk = int(np.sum(durations >= time_value))
            observed_events = int(
                np.sum((durations == time_value) & (events == 1))
            )
            if at_risk > 0 and observed_events > 0:
                survival *= 1.0 - (observed_events / at_risk)
            timeline.append(int(time_value))
            survival_probability.append(float(survival))

        return np.asarray(timeline), np.asarray(survival_probability)

    @staticmethod
    def _survival_at(
        time_value: int,
        survival_days: np.ndarray,
        survival_prob: np.ndarray,
    ) -> float:
        valid_indexes = np.where(survival_days <= time_value)[0]
        if len(valid_indexes) == 0:
            return 1.0
        return float(survival_prob[valid_indexes[-1]])

    def _analyze_mood(self, mood: str) -> dict[str, Any]:
        records = [record for record in self.streak_records if record["mood"] == mood]
        durations = np.asarray([record["duration"] for record in records], dtype=int)
        events = np.asarray([record["event_observed"] for record in records], dtype=int)
        days, survival = self._empirical_survival_hazard(durations, events)
        completed_runs = np.asarray(
            [record["duration"] for record in records if record["event_observed"] == 1],
            dtype=int,
        )
        return {
            "runs": completed_runs,
            "all_durations": durations,
            "event_observed": events,
            "survival_days": days,
            "survival_prob": survival,
        }

    def _get_confidence_flip_date(
        self,
        survival_days: np.ndarray,
        survival_prob: np.ndarray,
        confidence: float = 0.05,
    ) -> Optional[int]:
        """Return total streak duration at an unconditional survival threshold.

        Kept for backward compatibility. The displayed estimate uses the
        conditional residual-life method below because the current streak has
        already survived for `current_streak` sessions.
        """
        confidence = float(np.clip(confidence, 0.001, 0.999))
        for duration, survival in zip(survival_days, survival_prob):
            if survival <= confidence:
                return int(duration)
        return None

    def _conditional_quantile_remaining(
        self,
        mood: str,
        tail_probability: float,
    ) -> Optional[int]:
        """Return remaining sessions at a conditional survival threshold."""
        result = self._analyze_mood(mood)
        durations = result["all_durations"]
        if durations.size == 0:
            return None

        current = self.current_streak if mood == self.current_mood else 0
        survival_at_current = self._survival_at(
            current,
            result["survival_days"],
            result["survival_prob"],
        )
        if survival_at_current <= 0:
            return 1

        tail_probability = float(np.clip(tail_probability, 0.001, 0.999))
        maximum_duration = int(np.max(durations))
        for total_duration in range(current + 1, maximum_duration + 1):
            conditional_survival = self._survival_at(
                total_duration,
                result["survival_days"],
                result["survival_prob"],
            ) / survival_at_current
            if conditional_survival <= tail_probability:
                return max(1, total_duration - current)
        return None

    def _expected_remaining_sessions(self, mood: Optional[str] = None) -> Optional[int]:
        """Estimate conditional restricted mean remaining streak sessions."""
        mood = mood or self.current_mood
        result = self._analyze_mood(mood)
        durations = result["all_durations"]
        if durations.size == 0:
            return None

        current = self.current_streak if mood == self.current_mood else 0
        survival_at_current = self._survival_at(
            current,
            result["survival_days"],
            result["survival_prob"],
        )
        if survival_at_current <= 0:
            return 1

        maximum_duration = int(np.max(durations))
        residual_mean = 0.0
        for total_duration in range(current, maximum_duration):
            residual_mean += self._survival_at(
                total_duration,
                result["survival_days"],
                result["survival_prob"],
            ) / survival_at_current

        return max(1, int(round(residual_mean)))

    def _get_days_until_confidence_flip(self, confidence: float = 0.05) -> Optional[int]:
        """Return conditional remaining trading sessions at the requested tail."""
        return self._conditional_quantile_remaining(self.current_mood, confidence)

    def _get_forecast_horizon(self, confidence: float = 0.50) -> int:
        """Return a conditional KM horizon, replacing the uncensored Weibull fit."""
        remaining = self._conditional_quantile_remaining(
            self.current_mood,
            confidence,
        )
        return remaining if remaining is not None else 5

    def get_flip_estimate(self) -> dict[str, Any]:
        """Return expected and uncertainty dates for the current mood flip."""
        expected_remaining = self._expected_remaining_sessions() or 5
        median_remaining = self._conditional_quantile_remaining(
            self.current_mood,
            0.50,
        ) or expected_remaining
        upper_90_remaining = self._conditional_quantile_remaining(
            self.current_mood,
            0.10,
        )

        expected_date = add_business_sessions(self.mmi_last_date, expected_remaining)
        median_date = add_business_sessions(self.mmi_last_date, median_remaining)
        upper_90_date = (
            add_business_sessions(self.mmi_last_date, upper_90_remaining)
            if upper_90_remaining is not None
            else None
        )
        return {
            "expected_remaining_sessions": expected_remaining,
            "median_remaining_sessions": median_remaining,
            "upper_90_remaining_sessions": upper_90_remaining,
            "expected_date": expected_date,
            "median_date": median_date,
            "upper_90_date": upper_90_date,
        }

    def _generate_survival_based_forecast(
        self,
        forecast_days: int = 30,
        confidence: float = 0.50,
    ) -> pd.DataFrame:
        """Generate a reproducible illustrative MMI path.

        This is a scenario visualisation, not a price or MMI forecasting model.
        """
        remaining_sessions = self._get_forecast_horizon(confidence)
        dates = pd.bdate_range(
            start=pd.Timestamp(self.mmi_last_date) + pd.offsets.BDay(1),
            periods=max(1, int(forecast_days)),
        )
        seed = int(self.mmi_last_date.strftime("%Y%m%d")) + int(round(self.current_mmi * 100))
        rng = np.random.default_rng(seed)

        forecast = []
        current_mmi = self.current_mmi
        original_mood = self.current_mood
        flip_triggered = False

        for session_number, forecast_date in enumerate(dates, start=1):
            if session_number <= remaining_sessions:
                current_mmi += float(rng.uniform(-2.0, 2.0))
            elif not flip_triggered:
                flip_triggered = True
                current_mmi = 49.0 if original_mood == "Greed" else 51.0
            else:
                current_mmi += float(rng.uniform(-3.0, 3.0))

            current_mmi = float(np.clip(current_mmi, 0.0, 100.0))
            forecast.append((forecast_date.date(), current_mmi))

        return pd.DataFrame(forecast, columns=["Date", "Forecasted_MMI"])

    def generate_allocation_plan(
        self,
        investable_amount: float,
    ) -> tuple[pd.DataFrame, datetime.date]:
        """Generate the existing MMI-aware staggered allocation plan."""
        investable_amount = float(investable_amount)
        if investable_amount <= 0:
            raise ValueError("Investable amount must be greater than zero.")

        expected_remaining = self._expected_remaining_sessions() or 15
        expected_remaining = int(np.clip(expected_remaining, 5, 60))

        start_anchor = max(self.today_date, self.mmi_last_date)
        all_days = pd.bdate_range(
            start=pd.Timestamp(start_anchor) + pd.offsets.BDay(1),
            periods=expected_remaining,
        )

        # Preserve the original lower-frequency plan for long horizons.
        if expected_remaining >= 14:
            monday_days = [date for date in all_days if date.weekday() == 0]
            if all_days[-1] not in monday_days:
                monday_days.append(all_days[-1])
            all_days = pd.DatetimeIndex(monday_days)

        if len(all_days) < 5:
            all_days = pd.bdate_range(
                start=pd.Timestamp(start_anchor) + pd.offsets.BDay(1),
                periods=5,
            )

        mmi_step = (MMI_NEUTRAL_LEVEL - self.current_mmi) / max(1, len(all_days) - 1)
        estimated_mmi_values = [
            self.current_mmi + (mmi_step * index)
            for index in range(len(all_days))
        ]

        raw_weights = np.asarray(
            [max(0.01, MMI_NEUTRAL_LEVEL - value) for value in estimated_mmi_values],
            dtype=float,
        )
        normalised_weights = raw_weights / raw_weights.sum()

        allocation_rows = []
        allocations = investable_amount * normalised_weights
        allocations[-1] += investable_amount - float(allocations.sum())

        for index, (date, estimated_mmi, weight, amount) in enumerate(
            zip(all_days, estimated_mmi_values, normalised_weights, allocations),
            start=1,
        ):
            gap = max(0.0, MMI_NEUTRAL_LEVEL - estimated_mmi)
            allocation_rows.append(
                {
                    "Day": index,
                    "Date": date.strftime("%a, %d %b %Y"),
                    "Est. MMI": round(float(estimated_mmi), 2),
                    "MMI Gap": round(float(gap), 2),
                    "Weight": round(float(weight), 6),
                    "Allocation (%)": f"{weight * 100:.2f}%",
                    "Allocation (₹)": f"₹{amount:.2f}",
                }
            )

        confidence_date = all_days[-1].date()
        return pd.DataFrame(allocation_rows), confidence_date

    @staticmethod
    def _mean_confidence_interval(data: np.ndarray) -> tuple[Optional[float], Optional[tuple[float, float]]]:
        """Return a t-based 95% confidence interval for the historical mean."""
        if len(data) < 2:
            return None, None
        mean = float(np.mean(data))
        standard_error = float(stats.sem(data))
        margin = float(stats.t.ppf(0.975, len(data) - 1) * standard_error)
        return mean, (mean - margin, mean + margin)

    def display_mood_analysis(self) -> None:
        fear_result = self._analyze_mood("Fear")
        greed_result = self._analyze_mood("Greed")
        flip_estimate = self.get_flip_estimate()

        st.subheader("📈 Current Market Mood Analysis")
        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Current MMI", f"{self.current_mmi:.2f}", self.current_mood)
        metric_two.metric("Current Streak", f"{self.current_streak} trading sessions")

        days_from_today = (flip_estimate["expected_date"] - self.today_date).days
        status_text = (
            "today"
            if days_from_today == 0
            else f"in {days_from_today} calendar days"
            if days_from_today > 0
            else f"data estimate passed by {abs(days_from_today)} days"
        )
        metric_three.metric(
            "Expected Flip Date",
            flip_estimate["expected_date"].strftime("%d %b %Y"),
            status_text,
        )

        if self.mmi_last_date < self.today_date:
            st.warning(
                f"Latest MMI observation is {self.mmi_last_date:%d %b %Y}; "
                "the forecast starts from that observation, not from an assumed value for today."
            )

        uncertainty_text = (
            f"Median estimate: {flip_estimate['median_date']:%d %b %Y}. "
            f"About 90% of comparable historical streaks ended by "
            f"{flip_estimate['upper_90_date']:%d %b %Y}."
            if flip_estimate["upper_90_date"] is not None
            else f"Median estimate: {flip_estimate['median_date']:%d %b %Y}."
        )
        st.caption(
            "Conditional Kaplan-Meier estimate with the active streak treated as "
            f"right-censored. {uncertainty_text} Business dates skip weekends but not NSE holidays. "
            "This estimates an MMI threshold crossing, not a market correction."
        )

        with st.expander("📊 Show Historical Streak Patterns", expanded=False):
            fear_runs = fear_result["runs"]
            greed_runs = greed_result["runs"]
            fear_mean, fear_ci = self._mean_confidence_interval(fear_runs)
            greed_mean, greed_ci = self._mean_confidence_interval(greed_runs)

            st.markdown("**📐 95% Confidence Interval for the Historical Mean**")
            confidence_table = pd.DataFrame(
                {
                    "Mood": ["Fear", "Greed"],
                    "Mean Streak (sessions)": [
                        f"{fear_mean:.1f}" if fear_mean is not None else "N/A",
                        f"{greed_mean:.1f}" if greed_mean is not None else "N/A",
                    ],
                    "95% CI of Mean": [
                        f"{fear_ci[0]:.1f} – {fear_ci[1]:.1f}" if fear_ci else "N/A",
                        f"{greed_ci[0]:.1f} – {greed_ci[1]:.1f}" if greed_ci else "N/A",
                    ],
                }
            )
            st.table(confidence_table)

            def safe_mode(values: np.ndarray) -> Optional[int]:
                if len(values) == 0:
                    return None
                return int(stats.mode(values, keepdims=False).mode)

            summary_table = pd.DataFrame(
                {
                    "Mood": ["Fear", "Greed"],
                    "Completed Runs": [len(fear_runs), len(greed_runs)],
                    "Min": [int(np.min(fear_runs)), int(np.min(greed_runs))],
                    "Median": [float(np.median(fear_runs)), float(np.median(greed_runs))],
                    "Mean": [round(float(np.mean(fear_runs)), 1), round(float(np.mean(greed_runs)), 1)],
                    "Mode": [safe_mode(fear_runs), safe_mode(greed_runs)],
                    "75th Percentile": [
                        float(np.quantile(fear_runs, 0.75)),
                        float(np.quantile(greed_runs, 0.75)),
                    ],
                    "Max": [int(np.max(fear_runs)), int(np.max(greed_runs))],
                }
            )
            st.markdown("**📘 Completed Historical Streak Statistics (Trading Sessions)**")
            st.table(summary_table)

        st.info("### 💰 Capital Allocation")
        if self.current_mmi < MMI_NEUTRAL_LEVEL:
            invest_pct = float(np.clip((MMI_NEUTRAL_LEVEL - self.current_mmi) * 2, 0, 100))
            st.info(
                f"😊 **Fear regime, MMI = {self.current_mmi:.2f}**  \n"
                f"Rule-based deployment suggestion: invest up to **{invest_pct:.1f}%** "
                "of deployable cash gradually, subject to asset quality and personal risk limits."
            )
        elif self.current_mmi > MMI_NEUTRAL_LEVEL:
            liquid_hold_pct = float(np.clip((self.current_mmi - MMI_NEUTRAL_LEVEL) * 2, 0, 100))
            st.info(
                f"😬 **Greed regime, MMI = {self.current_mmi:.2f}**  \n"
                f"Rule-based reserve suggestion: keep at least **{liquid_hold_pct:.1f}%** "
                "of total capital in liquid or lower-volatility instruments."
            )
        else:
            st.info("⚖️ **MMI is neutral at 50.** Consider a balanced allocation.")

        current_completed_runs = (
            greed_runs if self.current_mood == "Greed" else fear_runs
        )
        percentile_rank = (
            float(np.mean(current_completed_runs <= self.current_streak))
            if len(current_completed_runs)
            else 0.5
        )

        if self.current_mood == "Greed":
            if percentile_rank < 0.50:
                st.warning(
                    f"📉 **Greed regime — historically early-to-middle stage.**  \n"
                    f"Current duration is around the {percentile_rank * 100:.0f}th percentile "
                    "of completed Greed streaks. Review concentrated winners and rebalance only "
                    f"where your plan supports it. A configurable profit-review threshold is "
                    f"**{active_profit_threshold:.1f}%**."
                )
            else:
                st.warning(
                    f"🛑 **Greed regime — historically extended.**  \n"
                    f"Current duration is around the {percentile_rank * 100:.0f}th percentile "
                    "of completed Greed streaks. This raises sentiment-regime risk but does not "
                    "prove that a price correction is imminent. Consider partial rebalancing, "
                    f"liquidity and position limits when net returns exceed **{active_profit_threshold:.1f}%**."
                )
        else:
            if percentile_rank < 0.50:
                st.success(
                    "🟢 **Fear regime — historically early-to-middle stage.** Accumulate only "
                    "high-quality assets gradually and preserve diversification."
                )
            else:
                st.success(
                    "📘 **Fear regime — historically extended.** Continue selective deployment, "
                    "but reassess fundamentals before averaging down."
                )


# -----------------------------------------------------------------------------
# ALLOCATION PLAN STORAGE
# -----------------------------------------------------------------------------
def save_allocation_plan(
    user_id: str,
    plan_df: pd.DataFrame,
    total_amount: float,
    days: int,
    mmi_snapshot: dict[str, Any],
) -> None:
    if allocation_collection is None:
        return
    allocation_collection.insert_one(
        {
            "user_id": user_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "investable_amount": float(total_amount),
            "total_days": int(days),
            "mmi_snapshot": mmi_snapshot,
            "confidence_date": mmi_snapshot.get("confidence_date"),
            "plan": plan_df.to_dict(orient="records"),
        }
    )


def get_latest_allocation_plan(user_id: str) -> Optional[pd.DataFrame]:
    if allocation_collection is None:
        return None
    try:
        record = allocation_collection.find_one(
            {"user_id": user_id},
            sort=[("timestamp", -1)],
        )
        return pd.DataFrame(record["plan"]) if record else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# HOLDINGS ANALYSIS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_equity_mapping() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/KPranaydeep/Finance/refs/heads/main/EQUITY_L.csv"
    mapping = pd.read_csv(url)
    mapping.columns = mapping.columns.str.strip()
    required = ["ISIN NUMBER", "SYMBOL", "NAME OF COMPANY"]
    missing = [column for column in required if column not in mapping.columns]
    if missing:
        raise ValueError(f"Equity mapping is missing columns: {missing}")
    return mapping[required].rename(
        columns={
            "ISIN NUMBER": "ISIN",
            "SYMBOL": "Symbol",
            "NAME OF COMPANY": "Company Name",
        }
    )


def analyze_holdings(uploaded_holdings: Any) -> Optional[pd.DataFrame]:
    """Analyse stock holdings from Groww or Kite XLSX exports."""
    try:
        uploaded_holdings.seek(0)
        workbook = pd.ExcelFile(uploaded_holdings)
        sheet_names = workbook.sheet_names
        uploaded_holdings.seek(0)

        if "Equity" in sheet_names:
            holdings = pd.read_excel(uploaded_holdings, sheet_name="Equity", skiprows=22)
            holdings.columns = holdings.columns.astype(str).str.strip()
            holdings = holdings.dropna(subset=["Symbol"])
            holdings = holdings.rename(
                columns={
                    "Quantity Available": "Quantity",
                    "Average Price": "Average Price",
                    "Unrealized P&L": "P&L",
                    "Unrealized P&L Pct.": "P&L (%)",
                    "Previous Closing Price": "Live LTP",
                }
            )
            holdings["Company Name"] = holdings["Symbol"]
            holdings["Price Source"] = "Previous close from Kite export"
        else:
            holdings = pd.read_excel(uploaded_holdings, sheet_name="Sheet1", skiprows=9)
            if not holdings.empty and holdings.iloc[0].astype(str).str.contains(
                "Stock Name", case=False
            ).any():
                holdings = holdings.iloc[1:]
            holdings = holdings.rename(
                columns={
                    "Unnamed: 0": "Stock Name",
                    "Unnamed: 1": "ISIN",
                    "Unnamed: 2": "Quantity",
                    "Unnamed: 3": "Average Price",
                    "Unnamed: 4": "Buy Value",
                    "Unnamed: 5": "LTP",
                    "Unnamed: 6": "Current Value",
                    "Unnamed: 7": "P&L",
                }
            )
            holdings = holdings.dropna(subset=["Stock Name", "ISIN"])
            holdings = holdings.merge(load_equity_mapping(), on="ISIN", how="left")
            holdings = holdings.dropna(subset=["Symbol"])
            holdings["Company Name"] = holdings["Company Name"].fillna(
                holdings["Stock Name"]
            )
            holdings["Live LTP"] = holdings["LTP"]
            holdings["Price Source"] = "LTP snapshot from Groww export"

        required_columns = ["Symbol", "Quantity", "Average Price", "Live LTP"]
        missing = [column for column in required_columns if column not in holdings.columns]
        if missing:
            raise ValueError(f"Holdings file is missing required fields: {missing}")

        for column in ["Quantity", "Average Price", "Live LTP"]:
            holdings[column] = pd.to_numeric(holdings[column], errors="coerce")

        holdings = holdings.dropna(subset=required_columns)
        holdings = holdings[
            (holdings["Quantity"] > 0)
            & (holdings["Average Price"] > 0)
            & (holdings["Live LTP"] > 0)
        ].copy()

        holdings["Invested Amount"] = holdings["Quantity"] * holdings["Average Price"]
        holdings["Current Value"] = holdings["Quantity"] * holdings["Live LTP"]
        holdings["Profit/Loss"] = holdings["Current Value"] - holdings["Invested Amount"]
        holdings["Profit/Loss (%)"] = np.where(
            holdings["Invested Amount"] > 0,
            holdings["Profit/Loss"] / holdings["Invested Amount"] * 100,
            np.nan,
        )

        return holdings[
            [
                "Symbol",
                "Company Name",
                "Quantity",
                "Average Price",
                "Live LTP",
                "Price Source",
                "Invested Amount",
                "Current Value",
                "Profit/Loss",
                "Profit/Loss (%)",
            ]
        ]
    except Exception as exc:
        st.error(f"❌ Could not process XLSX file: {exc}")
        return None


# -----------------------------------------------------------------------------
# MMI DATABASE HELPERS AND UI
# -----------------------------------------------------------------------------
def read_mmi_from_mongodb() -> pd.DataFrame:
    if mmi_collection is None:
        return pd.DataFrame()
    try:
        records = list(mmi_collection.find({}, {"_id": 0}))
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception as exc:
        st.error(f"❌ Failed to read MMI data from MongoDB: {exc}")
        return pd.DataFrame()


with st.expander("📂 Upload Full MMI Dataset", expanded=False):
    uploaded_mmi_csv = st.file_uploader(
        "Upload full MMI dataset (CSV format)",
        type=["csv"],
        key="upload_mmi_db",
    )

    uploaded_bytes: Optional[bytes] = None
    if uploaded_mmi_csv is not None and uploaded_mmi_csv.size > 0:
        uploaded_bytes = uploaded_mmi_csv.getvalue()
        try:
            validated_analyzer = MarketMoodAnalyzer(uploaded_bytes)
            validated_mmi_df = validated_analyzer.df[["Date", "MMI", "Nifty"]].copy()

            if mmi_collection is not None:
                documents = []
                for row in validated_mmi_df.to_dict(orient="records"):
                    documents.append(
                        {
                            "Date": pd.Timestamp(row["Date"]).to_pydatetime(),
                            "MMI": float(row["MMI"]),
                            "Nifty": None if pd.isna(row["Nifty"]) else float(row["Nifty"]),
                        }
                    )
                # Validate everything before replacing the full collection.
                mmi_collection.delete_many({})
                if documents:
                    mmi_collection.insert_many(documents, ordered=False)
                st.success("✅ Validated MMI data replaced the full MongoDB dataset.")
            else:
                st.success("✅ MMI data validated. MongoDB is disabled, so it is used for this session only.")
        except Exception as exc:
            st.error(f"❌ Error processing MMI CSV: {exc}")
            uploaded_bytes = None


try:
    if uploaded_bytes:
        st.info("📄 Using uploaded MMI CSV file")
        analyzer: Optional[MarketMoodAnalyzer] = MarketMoodAnalyzer(uploaded_bytes)
    else:
        database_mmi_df = read_mmi_from_mongodb()
        if not database_mmi_df.empty:
            st.info("☁️ Using MMI data from MongoDB")
            analyzer = MarketMoodAnalyzer(database_mmi_df)
        else:
            analyzer = None
            st.warning("⚠️ No valid MMI data found. Upload the full dataset or configure MongoDB.")
except Exception as exc:
    analyzer = None
    st.error(f"❌ Error loading MMI data: {exc}")

st.markdown("📊 [Visit Tickertape Market Mood Index](https://www.tickertape.in/market-mood-index)")

with st.expander("📝 Add Today's MMI", expanded=False):
    with st.form("add_today_mmi"):
        today_mmi = st.number_input(
            "Today's MMI",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
        )
        submitted = st.form_submit_button("📥 Save / Update in MongoDB")

    if submitted:
        if mmi_collection is None:
            st.error("MongoDB is not configured. Set MONGODB_URI before saving daily MMI values.")
        else:
            try:
                current_date = today_ist()
                nifty_history = yf.Ticker("^NSEI").history(period="5d")
                if nifty_history.empty:
                    raise ValueError("Could not fetch a recent NIFTY close.")
                nifty_value = float(nifty_history["Close"].dropna().iloc[-1])
                date_datetime = datetime.datetime.combine(
                    current_date,
                    datetime.time.min,
                )
                mmi_collection.update_one(
                    {"Date": date_datetime},
                    {
                        "$set": {
                            "Date": date_datetime,
                            "MMI": float(today_mmi),
                            "Nifty": nifty_value,
                        }
                    },
                    upsert=True,
                )
                st.success(f"✅ Saved MMI: {today_mmi:.1f} | Recent NIFTY close: {nifty_value:.2f}")

                refreshed_mmi_df = read_mmi_from_mongodb()
                analyzer = (
                    MarketMoodAnalyzer(refreshed_mmi_df)
                    if not refreshed_mmi_df.empty
                    else None
                )
            except Exception as exc:
                st.error(f"❌ Could not save today's MMI: {exc}")


if analyzer:
    analyzer.display_mood_analysis()

    if analyzer.current_mood == "Fear":
        st.success("🟢 MMI indicates Fear — gradual allocation planning is enabled.")
        st.markdown(
            "The plan staggers deployable capital across estimated business sessions. "
            "It reduces timing concentration but does not evaluate stock quality or valuation."
        )

        with st.form("allocation_plan_form"):
            investable_amount = st.number_input(
                "Investable Amount (₹)",
                min_value=100.0,
                step=1000.0,
            )
            submit_allocation = st.form_submit_button("Generate Allocation Plan")

        if submit_allocation:
            allocation_plan_df, confidence_date = analyzer.generate_allocation_plan(
                investable_amount
            )
            st.dataframe(allocation_plan_df, use_container_width=True)
            save_allocation_plan(
                "default_user",
                allocation_plan_df,
                investable_amount,
                len(allocation_plan_df),
                {
                    "mmi": analyzer.current_mmi,
                    "mood": analyzer.current_mood,
                    "streak": analyzer.current_streak,
                    "date": analyzer.mmi_last_date.strftime("%Y-%m-%d"),
                    "confidence_date": confidence_date.strftime("%Y-%m-%d"),
                },
            )
            st.download_button(
                "Download Allocation CSV",
                allocation_plan_df.to_csv(index=False),
                file_name="allocation_plan.csv",
            )

        with st.expander("🗂 View Last Saved Allocation Plan"):
            previous_plan = get_latest_allocation_plan("default_user")
            if previous_plan is not None:
                st.dataframe(previous_plan, use_container_width=True)
            else:
                st.info("No saved plan is available.")


uploaded_holdings = None
if analyzer and analyzer.current_mood == "Greed":
    st.header("📤 Upload Your Holdings")
    uploaded_holdings = st.file_uploader(
        "Upload your stock holdings file (.xlsx format only — Groww or Kite)",
        type=["xlsx"],
        key="upload_holdings_greed",
    )


# -----------------------------------------------------------------------------
# STOCK RECOMMENDATIONS FROM GOOGLE SHEET
# -----------------------------------------------------------------------------
st.markdown("## 📌 Recommended Stocks to Explore")
sheet_id = "1f1N_2T9xvifzf4BjeiwVgpAcak8_AVaEEbae_NXua8c"
sheet_name = "Sheet1"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit?usp=sharing"
st.markdown(f"🔗 [Open Google Sheet]({sheet_url})")

try:
    recommendation_raw_df = pd.read_csv(csv_url, header=None)
    recommendation_df = recommendation_raw_df.iloc[2:, [0, 1]].copy()
    recommendation_df.columns = ["Stock", "Score"]
    recommendation_df = recommendation_df.dropna(subset=["Stock"])
    recommendation_df["Stock"] = recommendation_df["Stock"].astype(str).str.strip()
    recommendation_df = recommendation_df[recommendation_df["Stock"] != ""]
    recommendation_df["Score"] = pd.to_numeric(
        recommendation_df["Score"],
        errors="coerce",
    ).fillna(0.0)
    recommendation_df = recommendation_df.drop_duplicates(subset=["Stock"], keep="first")

    if not recommendation_df.empty:
        sorted_recommendations = recommendation_df.sort_values("Score", ascending=False)
        top_ten = sorted_recommendations.head(10)["Stock"].tolist()
        remaining_stocks = sorted_recommendations.iloc[10:]["Stock"].tolist()
        random_ten = (
            random.sample(remaining_stocks, 10)
            if len(remaining_stocks) >= 10
            else remaining_stocks
        )
        combined_pool = list(dict.fromkeys(top_ten + random_ten))

        if "display_selection" not in st.session_state:
            st.session_state.display_selection = []

        if st.button("🎯 Pick 10 to Display"):
            st.session_state.display_selection = (
                random.sample(combined_pool, 10)
                if len(combined_pool) >= 10
                else combined_pool
            )

        if st.session_state.display_selection:
            st.markdown("### Selected Stocks")
            for stock in st.session_state.display_selection:
                st.write(f"- {stock}")
        st.caption("Community or score-based ideas require independent fundamental and risk review.")
except Exception as exc:
    st.error("❌ Failed to load Google Sheet data.")
    st.code(str(exc), language="text")


# -----------------------------------------------------------------------------
# PROFIT-BOOKING HELPERS
# -----------------------------------------------------------------------------
def get_april_first_current_year() -> str:
    current_date = today_ist()
    financial_year_start_year = (
        current_date.year if current_date.month >= 4 else current_date.year - 1
    )
    return datetime.date(financial_year_start_year, 4, 1).strftime("%Y-%m-%d")


def trading_days_elapsed(start_date_str: str, end_date_str: Optional[str] = None) -> int:
    end_date = (
        today_ist()
        if end_date_str is None
        else datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    )
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    if end_date < start_date:
        return 0
    return len(pd.bdate_range(start=start_date, end=end_date))


def calculate_dynamic_sell_limit(
    net_pl: float,
    charges: float,
    target_net_daily_pct: float = DEFAULT_TARGET_NET_DAILY_PCT,
) -> float:
    """Calculate the gross return required to retain a target net return.

    Charges are modelled as their historical share of gross trading profit:
    gross P&L = net P&L + charges. The previous formula divided that percentage
    by elapsed days, mixing incompatible units.
    """
    net_pl = float(net_pl)
    charges = max(0.0, float(charges))
    target_net_daily_pct = max(0.0, float(target_net_daily_pct))

    gross_pl = net_pl + charges
    if net_pl <= 0 or gross_pl <= 0:
        raise ValueError("Positive net P&L is required to estimate charge drag.")

    cost_fraction = float(np.clip(charges / gross_pl, 0.0, 0.95))
    required_gross_return_pct = target_net_daily_pct / max(1.0 - cost_fraction, 0.05)
    return round(1.0 + required_gross_return_pct / 100.0, 6)


def extract_net_pl_and_charges(file: Any) -> tuple[float, float]:
    file.seek(0)
    workbook = openpyxl.load_workbook(file, data_only=True, read_only=True)
    worksheet = workbook.active
    net_pl = worksheet["B9"].value or 0.0
    charges = worksheet["B25"].value or 0.0
    return float(net_pl), float(charges)


# -----------------------------------------------------------------------------
# PROFIT BOOKING UI
# -----------------------------------------------------------------------------
if uploaded_holdings is not None:
    st.header("💼 Your Portfolio Analysis")
    merged_df = analyze_holdings(uploaded_holdings)

    if merged_df is not None and not merged_df.empty:
        st.subheader("🧾 Current Holdings")
        st.dataframe(
            merged_df[
                [
                    "Symbol",
                    "Company Name",
                    "Quantity",
                    "Average Price",
                    "Live LTP",
                    "Price Source",
                    "Current Value",
                    "Profit/Loss",
                    "Profit/Loss (%)",
                ]
            ],
            use_container_width=True,
        )
        if merged_df["Price Source"].str.contains("Previous close").any():
            st.caption(
                "Kite's exported 'Previous Closing Price' is a reference price, not a live quote."
            )

        total_invested = float(merged_df["Invested Amount"].sum())
        total_current_value = float(merged_df["Current Value"].sum())
        total_profit_loss = float(merged_df["Profit/Loss"].sum())
        overall_return_pct = (
            total_profit_loss / total_invested * 100 if total_invested > 0 else 0.0
        )

        st.subheader("📊 Portfolio Summary")
        summary_one, summary_two, summary_three = st.columns(3)
        summary_one.metric("💰 Total Invested", f"₹{total_invested:,.2f}")
        summary_two.metric("📈 Reference Value", f"₹{total_current_value:,.2f}")
        summary_three.metric(
            "📊 Overall P&L",
            f"₹{total_profit_loss:,.2f}",
            delta=f"{overall_return_pct:.2f}%",
        )

        user_id = "default_user"
        uploaded_report = st.file_uploader(
            "📄 Upload your P&L Report (B9 = Net P&L, B25 = Charges)",
            type=["xlsx"],
        )
        latest_params = get_latest_input_params(user_id)
        net_pl_default = float(latest_params.get("net_pl", 0.0))
        charges_default = float(latest_params.get("charges", 0.0))

        if uploaded_report is not None:
            try:
                net_pl_default, charges_default = extract_net_pl_and_charges(
                    uploaded_report
                )
                st.success("✅ Auto-filled Net P&L and charges from the uploaded report.")
            except Exception as exc:
                st.error(f"⚠️ Failed to extract report values: {exc}")

        st.subheader("🎯 Profit Booking Strategy")
        with st.form("profit_booking_parameters"):
            net_pl = st.number_input(
                "Enter net P&L (INR)",
                value=max(0.0, net_pl_default),
                min_value=0.0,
                step=1000.0,
            )
            charges_input = st.number_input(
                "Enter charges (INR)",
                value=max(0.0, charges_default),
                min_value=0.0,
                step=100.0,
            )
            target_net_daily_pct = st.number_input(
                "Target net daily return (%)",
                value=float(latest_params.get("target_pct", DEFAULT_TARGET_NET_DAILY_PCT)),
                min_value=0.01,
                max_value=5.0,
                step=0.01,
            )
            rotation_option = st.radio(
                "Select rotation strategy for calculating target profit:",
                ["Daily", "Weekly", "Monthly"],
                index=0,
                horizontal=True,
            )
            target_rupees_input = st.number_input(
                "Target profit (₹); enter 0 to use the automatically calculated target",
                value=0.0,
                min_value=0.0,
                step=100.0,
            )
            calculate_plan = st.form_submit_button("Save Parameters and Calculate")

        if calculate_plan:
            save_input_params(user_id, net_pl, charges_input, target_net_daily_pct)

            if net_pl <= 0:
                st.error("❌ A positive net P&L is required to calculate charge drag.")
            else:
                try:
                    sell_limit_multiplier = calculate_dynamic_sell_limit(
                        net_pl,
                        charges_input,
                        target_net_daily_pct,
                    )
                    required_return_pct = (sell_limit_multiplier - 1.0) * 100.0
                    st.markdown(
                        f"💡 Required gross return is approximately **{required_return_pct:.4f}%** "
                        "above average cost to retain the selected net target under the historical charge ratio."
                    )

                    period_sessions = {"Daily": 1, "Weekly": 5, "Monthly": 21}
                    period_target_pct = (
                        (1.0 + required_return_pct / 100.0)
                        ** period_sessions[rotation_option]
                        - 1.0
                    )
                    default_target = round(total_invested * period_target_pct, 2)
                    target_rupees = (
                        float(target_rupees_input)
                        if target_rupees_input > 0
                        else max(0.0, default_target)
                    )
                    st.caption(f"Profit target used: ₹{target_rupees:,.2f}")

                    profitable_df = merged_df[merged_df["Profit/Loss"] > 0].copy()
                    profitable_df = profitable_df.sort_values(
                        "Profit/Loss (%)",
                        ascending=False,
                    )
                    profitable_df["Target Price (₹)"] = (
                        profitable_df["Average Price"] * sell_limit_multiplier
                    ).round(2)
                    profitable_df["Execution Estimate (₹)"] = profitable_df[
                        ["Live LTP", "Target Price (₹)"]
                    ].max(axis=1)

                    cumulative_profit = 0.0
                    sell_plan_rows = []

                    for _, row in profitable_df.iterrows():
                        if cumulative_profit >= target_rupees:
                            break

                        per_share_profit = (
                            float(row["Execution Estimate (₹)"])
                            - float(row["Average Price"])
                        )
                        if per_share_profit <= 0:
                            continue

                        maximum_shares = int(math.floor(float(row["Quantity"])))
                        if maximum_shares <= 0:
                            continue

                        needed_profit = max(0.0, target_rupees - cumulative_profit)
                        shares_to_sell = min(
                            maximum_shares,
                            max(1, int(math.ceil(needed_profit / per_share_profit))),
                        )
                        actual_profit = shares_to_sell * per_share_profit
                        cumulative_profit += actual_profit

                        sell_plan_rows.append(
                            {
                                "Symbol": row["Symbol"],
                                "Company Name": row["Company Name"],
                                "Quantity to Sell": shares_to_sell,
                                "Average Price": float(row["Average Price"]),
                                "Reference Price": float(row["Live LTP"]),
                                "Target Price (₹)": float(row["Target Price (₹)"]),
                                "Execution Estimate (₹)": float(row["Execution Estimate (₹)"]),
                                "Expected Gross Profit": actual_profit,
                                "Profit (%)": per_share_profit / float(row["Average Price"]) * 100,
                            }
                        )

                    if sell_plan_rows:
                        sell_plan_df = pd.DataFrame(sell_plan_rows)
                        st.success(
                            f"✅ Suggested sell plan estimates ₹{cumulative_profit:,.2f} gross profit."
                        )
                        st.dataframe(sell_plan_df, use_container_width=True)
                        st.caption(
                            "Execution price, taxes, slippage and live brokerage can differ. "
                            "Use a current quote before placing an order."
                        )
                        st.download_button(
                            "📥 Download Sell Plan",
                            sell_plan_df.to_csv(index=False).encode("utf-8"),
                            "sell_plan.csv",
                            "text/csv",
                            key="download-sell-plan",
                        )
                    else:
                        st.warning("📉 No profitable holdings can currently support the target.")
                except Exception as exc:
                    st.error(f"Could not calculate the sell plan: {exc}")


# -----------------------------------------------------------------------------
# SELL OR HOLD STRATEGY
# -----------------------------------------------------------------------------
target_profit_percent = 7.2
assumed_daily_net_return_pct = 0.278

holding_sessions = int(
    math.ceil(
        math.log1p(target_profit_percent / 100.0)
        / math.log1p(assumed_daily_net_return_pct / 100.0)
    )
)
strategy_today = today_ist()
strategy_exit_date = add_business_sessions(strategy_today, holding_sessions)
holding_calendar_days = (strategy_exit_date - strategy_today).days

st.markdown(
    f"""
### 📌 Sell or Hold Strategy

- ✅ **Illustrative target period: {holding_sessions} trading sessions**  
  From **{strategy_today.strftime('%b %d, %Y')}**, the weekday-based target date is **{strategy_exit_date.strftime('%b %d, %Y')}** ({holding_calendar_days} calendar days).

- 📉 If a holding is in loss on that date, **reassess the business thesis, valuation, opportunity cost and risk**. Do not hold indefinitely only to reach breakeven.

- 💰 Review for partial or full exit when estimated **net profit ≥ {target_profit_percent:.1f}%**, using current prices and brokerage-inclusive estimates:
  - [Groww Brokerage Calculator](https://groww.in/calculators/brokerage-calculator)
  - [Zerodha Brokerage Calculator](https://zerodha.com/brokerage-calculator/#tab-equities)

The date is derived from an assumed compounded daily net return of **{assumed_daily_net_return_pct:.3f}%**. It is a planning assumption, not an expected-return guarantee.
"""
)

if mongo_error:
    st.caption(f"Database status: {mongo_error}")
