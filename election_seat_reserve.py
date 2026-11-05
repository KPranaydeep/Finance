"""User-selected seat-exposure reserve policy, distinct from empirical loss risk."""
import math
from urllib.parse import urlparse

import pandas as pd

from election_watch import SEED, LOK_SABHA_ELECTED_SEATS, RAJYA_SABHA_ELECTED_SEATS

WINDOW_DAYS = 20
DENOMINATORS = {"assembly": sum(row[2] for row in SEED),
                "ls": LOK_SABHA_ELECTED_SEATS, "rs": RAJYA_SABHA_ELECTED_SEATS}
LABELS = {"assembly": "Assembly", "ls": "Lok Sabha", "rs": "Rajya Sabha"}


def official_schedule(event):
    source = urlparse(str(event.get("source") or ""))
    return (event.get("verified") is True and source.scheme == "https"
            and source.hostname in {"eci.gov.in", "www.eci.gov.in", "pib.gov.in", "www.pib.gov.in"})


def rolling_seat_reserve(events, today, inr_weight, *, assembly_snapshot=None):
    """[today, today+19]: sum actual contests per house, then take the maximum.

    No term dates, state LS/RS allocations, or polls repeated across phases.
    Missing counts remain explicit; a known subset is never called complete.
    """
    share = float(inr_weight)
    if not math.isfinite(share) or not 0 <= share <= 1.001:
        raise ValueError("Complete INR exposure is required")
    day = pd.Timestamp(today).normalize().tz_localize(None)
    end = day + pd.Timedelta(days=WINDOW_DAYS - 1)
    denominators = DENOMINATORS.copy()
    denominator_errors = []
    if assembly_snapshot is not None:
        rows = assembly_snapshot.get('rows') or []
        try:
            if len(rows) < 31 or len({row['State'] for row in rows}) != len(rows):
                raise ValueError('Partial assembly denominator')
            counts = [int(row['Assembly seats']) for row in rows]
            if any(n <= 0 for n in counts):
                raise ValueError('Invalid assembly seats')
            denominators['assembly'] = sum(counts)
        except (KeyError, TypeError, ValueError):
            denominators['assembly'] = None
            denominator_errors.append('National assembly denominator unavailable')
    totals = dict.fromkeys(DENOMINATORS, 0)
    used, unknown, seen = [], denominator_errors, {}
    for event in events:
        counting = pd.to_datetime(event.get("counting_date"), errors="coerce", utc=True)
        if pd.isna(counting) or not official_schedule(event):
            continue
        counting = counting.tz_convert(None).normalize()
        if not day <= counting <= end or event.get("cancelled") or event.get("superseded"):
            continue
        identity = event.get("election_id")
        counts = event.get("contested_seats")
        if not identity or not isinstance(counts, dict) or set(counts) != set(DENOMINATORS):
            unknown.append(event.get("name", "Unnamed election"))
            continue
        try:
            values = {house: int(counts[house]) for house in DENOMINATORS}
            if any(isinstance(counts[h], bool) or float(counts[h]) != values[h]
                   or not 0 <= values[h] <= (denominators[h] or float('inf')) for h in values):
                raise ValueError("Invalid contest count")
        except (TypeError, ValueError, OverflowError):
            unknown.append(event.get("name", "Unnamed election"))
            continue
        # Same election repeated by ECI and PIB counts once. Conflicting dates
        # or counts are withheld rather than choosing whichever was read last.
        signature = (str(counting.date()), tuple(values.items()))
        if identity in seen:
            if signature != seen[identity][0]:
                unknown.append(str(identity) + " (conflicting schedules)")
                seen[identity][1] = None
            continue
        seen[identity] = [signature, (event, values)]
    for signature, record in seen.values():
        if record is None:
            continue
        event, values = record
        for house in totals:
            totals[house] += values[house]
        used.append(event)
    if any(denominators[h] is not None and totals[h] > denominators[h] for h in totals):
        unknown.append("Overlapping contest counts exceed a house total")
        totals = dict.fromkeys(DENOMINATORS, 0)
    fractions = {house: totals[house] / denominators[house] if denominators[house] else None for house in totals}
    usable = {house:value for house,value in fractions.items() if value is not None}
    maximum = max(usable.values())
    driver = max(usable, key=usable.get) if maximum else None
    return dict(window_start=str(day.date()), window_end=str(end.date()), window_days=WINDOW_DAYS,
                totals=totals, denominators=denominators, fractions=fractions,
                denominator_verified_at=(assembly_snapshot or {}).get('verified_at', '2026-10-30'),
                seat_fraction=maximum, inr_weight=share, reserve_fraction=share * maximum,
                driver=LABELS.get(driver), events=used, missing_counts=unknown,
                coverage_complete=not unknown, basis="user-selected seat-exposure policy; not a loss forecast")
