from __future__ import annotations

from datetime import date, timedelta


MARKET = "NSE_EQ"
CALENDAR_YEAR = 2026

SOURCE_NAME = "NSE India - Market Timings & Holidays"
SOURCE_REF = "https://www.nseindia.com/resources/exchange-communication-holidays"
SOURCE_AS_OF = "2026-09-03"


# ---------------------------------------------------------------------
# OFFICIAL NSE EQUITY TRADING HOLIDAYS - 2026
# ---------------------------------------------------------------------
#
# Keep this calendar version-controlled.
#
# If NSE changes the calendar later, do not silently rewrite historical
# database snapshots. Store a new calendar snapshot instead.
# ---------------------------------------------------------------------

NSE_EQUITY_HOLIDAYS_2026 = {
    date(2026, 1, 15): "Municipal Corporation Election in Maharashtra",
    date(2026, 1, 26): "Republic Day",
    date(2026, 2, 19): "Chhatrapati Shivaji Maharaj Jayanti",
    date(2026, 3, 3): "Holi (Second Day)",
    date(2026, 3, 19): "Gudhi Padwa",
    date(2026, 3, 26): "Ram Navami",
    date(2026, 3, 31): "Mahavir Jayanti",
    date(2026, 4, 1): "Annual Bank Closing",
    date(2026, 4, 3): "Good Friday",
    date(2026, 4, 14): "Dr. Babasaheb Ambedkar Jayanti",
    date(2026, 5, 1): "Maharashtra Din / Buddha Pournima",
    date(2026, 5, 28): "Bakri ID (Id-Uz-Zuha)",
    date(2026, 6, 26): "Muharram",
    date(2026, 8, 26): "Id-E-Milad",
    date(2026, 9, 14): "Ganesh Chaturthi",
    date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 10): "Diwali (Bali Pratipada)",
    date(2026, 11, 24): "Guru Nanak Jayanti",
    date(2026, 12, 25): "Christmas",
}


# Special NSE open session.
NSE_SPECIAL_OPEN_SESSIONS_2026 = {
    date(2026, 11, 8): (
        "Muhurat Trading - special NSE session; "
        "timings notified separately by the exchange"
    ),
}


def week_start(reference_date: date) -> date:
    """
    Return Monday of the calendar week containing reference_date.
    """
    return reference_date - timedelta(days=reference_date.weekday())


def week_end(reference_date: date) -> date:
    """
    Return Sunday of the calendar week containing reference_date.
    """
    return week_start(reference_date) + timedelta(days=6)


def build_nse_equity_sessions_2026() -> list[dict]:
    """
    Return one explicit NSE-equity session record for every
    calendar date in 2026.

    session_type values:

    NORMAL
    SPECIAL
    WEEKEND
    EXCHANGE_HOLIDAY
    """

    rows: list[dict] = []

    current = date(2026, 1, 1)
    end = date(2026, 12, 31)

    while current <= end:

        if current in NSE_SPECIAL_OPEN_SESSIONS_2026:

            rows.append(
                {
                    "session_date": current.isoformat(),
                    "is_open": True,
                    "session_type": "SPECIAL",
                    "notes": NSE_SPECIAL_OPEN_SESSIONS_2026[current],
                }
            )

        elif current.weekday() >= 5:

            rows.append(
                {
                    "session_date": current.isoformat(),
                    "is_open": False,
                    "session_type": "WEEKEND",
                    "notes": None,
                }
            )

        elif current in NSE_EQUITY_HOLIDAYS_2026:

            rows.append(
                {
                    "session_date": current.isoformat(),
                    "is_open": False,
                    "session_type": "EXCHANGE_HOLIDAY",
                    "notes": NSE_EQUITY_HOLIDAYS_2026[current],
                }
            )

        else:

            rows.append(
                {
                    "session_date": current.isoformat(),
                    "is_open": True,
                    "session_type": "NORMAL",
                    "notes": None,
                }
            )

        current += timedelta(days=1)

    return rows


def get_nse_equity_session(target_date: date) -> dict:
    """
    Return the NSE equity session record for target_date.
    """
    if target_date.year != CALENDAR_YEAR:
        raise ValueError(
            f"No verified NSE equity calendar snapshot is bundled "
            f"for {target_date.year}."
        )

    session_map = {
        row["session_date"]: row
        for row in build_nse_equity_sessions_2026()
    }

    return session_map[target_date.isoformat()]
        

def nse_session_for_date(target_date: date) -> dict:
    """
    Return NSE session information for one date.

    Unsupported years fail closed instead of guessing.
    """

    if target_date.year != CALENDAR_YEAR:
        raise ValueError(
            f"No verified NSE equity calendar snapshot is bundled "
            f"for {target_date.year}."
        )

    session_map = {
        row["session_date"]: row
        for row in build_nse_equity_sessions_2026()
    }

    return session_map[target_date.isoformat()]


def first_nse_trading_day_of_week(
    reference_date: date,
) -> date | None:
    """
    Return the first OPEN NSE equity session in the Monday-Sunday
    calendar week containing reference_date.

    Examples:

    Monday open:
        -> Monday

    Monday holiday, Tuesday open:
        -> Tuesday

    Monday and Tuesday closed, Wednesday open:
        -> Wednesday

    Returns None only if the entire week has no open session.
    """

    if reference_date.year != CALENDAR_YEAR:
        raise ValueError(
            f"No verified NSE equity calendar snapshot is bundled "
            f"for {reference_date.year}."
        )

    monday = week_start(reference_date)

    session_map = {
        row["session_date"]: row
        for row in build_nse_equity_sessions_2026()
    }

    for offset in range(7):

        candidate = monday + timedelta(days=offset)

        row = session_map.get(candidate.isoformat())

        if row and row["is_open"]:
            return candidate

    return None


def calendar_snapshot_payload_2026() -> dict:
    """
    Return the complete deterministic calendar payload for
    hashing and storage in the public-basket ledger.
    """

    return {
        "market": MARKET,
        "range_start": "2026-01-01",
        "range_end": "2026-12-31",
        "source": SOURCE_NAME,
        "source_ref": SOURCE_REF,
        "source_as_of": SOURCE_AS_OF,
        "sessions": build_nse_equity_sessions_2026(),
    }
