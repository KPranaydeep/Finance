"""Source-dated schedules, retained through provider failures; no term-date guesses."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
import requests
from election_watch import _atomic_save

INDEX = "https://www.eci.gov.in/general-election-to-state-legislative-assembly"
PIB_INDEX = "https://www.pib.gov.in/allRel.aspx"
SEED_EVENTS = [
    dict(name="Delhi 2025", poll_dates=["2025-02-05"], counting_date="2025-02-08", verified=True,
         source="https://www.pib.gov.in/PressReleasePage.aspx?PRID=2090923"),
    dict(name="Bihar 2025", poll_dates=["2025-11-06", "2025-11-11"], counting_date="2025-11-14", verified=True,
         source="https://www.pib.gov.in/PressReleasePage.aspx?PRID=2189717"),
    dict(name="West Bengal 2026", poll_dates=["2026-04-23", "2026-04-29"], counting_date="2026-05-04", verified=True,
         source="https://www.eci.gov.in/election-details/2026/S25/3"),
]


def parse_schedule(html, source):
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    if not re.search(r"General Election.*?(Legislative Assembly|Legislative Assemblies)", text, re.I):
        raise ValueError("Not an assembly general-election schedule")
    def dates(label):
        return sorted({str(pd.to_datetime(v, dayfirst=True).date()) for v in
                       re.findall(label + r"\s*:?\s*(\d{2}[-./]\d{2}[-./]\d{4})", text, re.I)})
    counts, polls = dates(r"Date of Counting"), dates(r"Date of Poll")
    if not counts or not polls or max(polls) > min(counts):
        raise ValueError("No unambiguous polling/counting dates in source")
    return [dict(name="Assembly results " + counting, counting_date=counting, poll_dates=polls,
                 source=source, verified=True) for counting in counts]


def monthly_schedule(path, today, force=False, fetch=None):
    path = Path(path)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {"events": SEED_EVENTS, "baseline_verified_at": "2026-10-30"}
    if not force and saved.get("attempted_month") == str(today)[:7] and saved.get("discovery_version") == 2:
        return saved
    get = fetch or requests.get
    updated = dict(saved, discovery_version=2, attempted_month=str(today)[:7], attempted_at=datetime.now(timezone.utc).isoformat())
    try:
        response = get(INDEX, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = sorted({urljoin(INDEX, a["href"]) for a in soup.select("a[href]")
                        if re.fullmatch(r"/election-details/\d{4}/[A-Z]\d+/3", a["href"])})
        if not links:
            raise ValueError("ECI index has no readable schedule links; retained dated baseline")
        events = {(e["source"], e["counting_date"]): e for e in saved["events"]}
        errors = []
        for link in links[-12:]:
            try:
                page = get(link, timeout=8)
                page.raise_for_status()
                parsed = parse_schedule(page.text, link)
                # A revised official schedule replaces earlier dates for that
                # source; do not keep a superseded counting day active.
                events = {key: value for key, value in events.items() if key[0] != link}
                for event in parsed:
                    events[(event["source"], event["counting_date"])] = event
            except Exception as exc:
                errors.append(str(exc)[:120])
        updated.update(events=list(events.values()), checked_at=str(today))
        if errors:
            updated["refresh_error"] = "; ".join(errors[:2])
        else:
            updated.pop("refresh_error", None)
    except Exception as exc:
        updated["refresh_error"] = str(exc)[:180]
        # PIB republishes Election Commission announcements. Use only official
        # release links and explicit schedules, not search snippets or term ends.
        try:
            listing = get(PIB_INDEX, timeout=8)
            listing.raise_for_status()
            soup = BeautifulSoup(listing.text, "html.parser")
            links = set()
            for anchor in soup.select("a[href]"):
                title = anchor.get_text(" ", strip=True)
                link = urljoin(PIB_INDEX, anchor["href"])
                if (urlparse(link).hostname in {"pib.gov.in", "www.pib.gov.in"}
                        and re.search(r"PRID=\d+", link, re.I)
                        and re.search(r"schedul", title, re.I)
                        and re.search(r"Legislative Assembl", title, re.I)):
                    links.add(link)
            events = {(e["source"], e["counting_date"]): e for e in saved["events"]}
            parsed_count, failures = 0, []
            for link in sorted(links)[:6]:
                try:
                    page = get(link, timeout=8)
                    page.raise_for_status()
                    parsed = parse_schedule(page.text, link)
                    events = {key: value for key, value in events.items() if key[0] != link}
                    for event in parsed:
                        events[(link, event["counting_date"])] = event
                        parsed_count += 1
                except Exception as error:
                    failures.append(str(error)[:100])
            updated.update(events=list(events.values()), fallback_checked_at=str(today),
                           fallback_source=PIB_INDEX, fallback_schedules=parsed_count)
            updated["refresh_error"] += "; PIB latest releases checked: " + str(parsed_count) + " readable schedules"
            if failures:
                updated["refresh_error"] += "; " + failures[0]
        except Exception as fallback_error:
            updated["refresh_error"] += "; PIB: " + str(fallback_error)[:100]
    _atomic_save(path, updated)
    return updated
