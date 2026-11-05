"""Source-dated schedules, retained through provider failures; no term-date guesses."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
import requests
from election_watch import _atomic_save
from election_watch import SEED

INDEX = "https://www.eci.gov.in/general-election-to-state-legislative-assembly"
PIB_INDEX = "https://www.pib.gov.in/allRel.aspx?lang=1&reg=3"
SEED_EVENTS = [
    dict(name="Delhi 2025", election_id="assembly:NCT Delhi:2025", contested_seats={"assembly":70,"ls":0,"rs":0}, poll_dates=["2025-02-05"], counting_date="2025-02-08", verified=True,
         source="https://www.pib.gov.in/PressReleasePage.aspx?PRID=2090923"),
    dict(name="Bihar 2025", election_id="assembly:Bihar:2025", contested_seats={"assembly":243,"ls":0,"rs":0}, poll_dates=["2025-11-06", "2025-11-11"], counting_date="2025-11-14", verified=True,
         source="https://www.pib.gov.in/PressReleasePage.aspx?PRID=2189717"),
    dict(name="West Bengal 2026", election_id="assembly:West Bengal:2026", contested_seats={"assembly":294,"ls":0,"rs":0}, poll_dates=["2026-04-23", "2026-04-29"], counting_date="2026-05-04", verified=True,
         source="https://www.eci.gov.in/election-details/2026/S25/3"),
]


def parse_schedule(html, source, *, assembly_snapshot=None):
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    assembly = bool(re.search(r"General Election.*?(Legislative Assembly|Legislative Assemblies)", text, re.I))
    ls = bool(re.search(r"(?:General|Bye)[ -]?Election.*?(Lok Sabha|House of the People)", text, re.I))
    rs = bool(re.search(r"(?:Biennial|Bye)[ -]?Election.*?(Rajya Sabha|Council of States)", text, re.I))
    if sum((assembly, ls, rs)) != 1:
        raise ValueError("Not an unambiguous assembly/LS/RS election schedule")
    def dates(label):
        return sorted({str(pd.to_datetime(v, dayfirst=True).date()) for v in
                       re.findall(label + r"\s*:?\s*(\d{2}[-./]\d{2}[-./]\d{4})", text, re.I)})
    counts, polls = dates(r"Date of Counting"), dates(r"Date of Poll")
    if not counts or not polls or max(polls) > min(counts):
        raise ValueError("No unambiguous polling/counting dates in source")
    events = []
    for counting in counts:
        event = dict(name=("Assembly" if assembly else "Lok Sabha" if ls else "Rajya Sabha") + " results " + counting,
                     counting_date=counting, poll_dates=polls, source=source, verified=True)
        # Assign seat counts only when a single result date unambiguously covers
        # the named full assemblies, or an explicit parliamentary seat count.
        if len(counts) == 1:
            heading = re.split(r"Date of (?:Poll|Counting)", text, maxsplit=1, flags=re.I)[0][:1000]
            state_rows = SEED if assembly_snapshot is None else [
                (row['State'], '', row['Assembly seats']) for row in assembly_snapshot.get('rows', [])]
            states = [row for row in state_rows if re.search(r"\b" + re.escape(row[0].replace("NCT Delhi", "Delhi")) + r"\b", heading, re.I)]
            if assembly and states:
                for state in states:
                    events.append(dict(event, name=state[0] + " results " + counting,
                                       election_id="assembly:" + state[0] + ":" + counting[:4],
                                       contested_seats={"assembly":state[2],"ls":0,"rs":0}))
                continue
            elif ls or rs:
                chamber = "ls" if ls else "rs"
                label = r"(?:Lok Sabha|House of the People)" if ls else r"(?:Rajya Sabha|Council of States)"
                numbers = {int(n) for n in re.findall(r"(\d+)\s+seats?\s+(?:in|to|of)\s+(?:the\s+)?" + label, text, re.I)}
                if len(numbers) == 1:
                    event.update(election_id=chamber + ":" + ",".join(sorted(row[0] for row in states)) + ":" + str(next(iter(numbers))) + ":" + counting,
                                 contested_seats={"assembly":0,"ls":next(iter(numbers)) if ls else 0,"rs":next(iter(numbers)) if rs else 0})
        events.append(event)
    return events


def monthly_schedule(path, today, force=False, fetch=None, *, assembly_snapshot=None):
    path = Path(path)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {"events": SEED_EVENTS, "baseline_verified_at": "2026-10-30"}
    # Migrate counts for the dated baseline without pretending a live refresh
    # happened. Never overwrite later official revisions.
    seeded = {(e["source"],e["counting_date"]):e for e in SEED_EVENTS}
    saved["events"] = [dict(seeded.get((e.get("source"),e.get("counting_date")),{}), **e) for e in saved["events"]]
    near_event = any(str(today) <= e.get("counting_date", "") <= str((pd.Timestamp(today)+pd.Timedelta(days=19)).date()) for e in saved["events"])
    daily_due = near_event and saved.get("attempted_date") != str(today)
    if not force and not daily_due and saved.get("attempted_month") == str(today)[:7] and saved.get("discovery_version") == 3:
        return saved
    get = fetch or requests.get
    deadline = time.monotonic() + 24
    def read(url):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Schedule refresh time budget reached; previous verified dates retained")
        response = get(url, timeout=min(8, remaining), headers={"User-Agent":"Mozilla/5.0 portfolio election watch"})
        response.raise_for_status()
        return response
    def key(event):
        return (event['source'],event.get('election_id') or event['name'],event['counting_date'])
    updated = dict(saved, discovery_version=3, attempted_month=str(today)[:7], attempted_date=str(today), attempted_at=datetime.now(timezone.utc).isoformat(),
                   coverage_complete=False, coverage="Retained baseline and discovered official notices; nationwide assembly/LS/RS schedule completeness unverified")
    events = {key(e): e for e in saved['events']}
    errors = []
    try:
        response = read(INDEX)
        soup = BeautifulSoup(response.text, "html.parser")
        links = sorted({urljoin(INDEX, a["href"]) for a in soup.select("a[href]")
                        if re.fullmatch(r"/election-details/\d{4}/[A-Z]\d+/3", a["href"])})
        if not links:
            raise ValueError("ECI index has no readable schedule links; retained dated baseline")
        for link in links[-8:]:
            try:
                page = read(link)
                parsed = parse_schedule(page.text, link, assembly_snapshot=assembly_snapshot)
                # A revised official schedule replaces earlier dates for that
                # source; do not keep a superseded counting day active.
                events = {key: value for key, value in events.items() if key[0] != link}
                for event in parsed:
                    events[key(event)] = event
            except Exception as exc:
                errors.append(str(exc)[:120])
        updated.update(events=list(events.values()), checked_at=str(today))
    except Exception as exc:
        errors.append("ECI: " + str(exc)[:180])
    # Always inspect the English official releases for LS/RS notices as well;
    # successful assembly discovery says nothing about parliamentary coverage.
    try:
        listing = read(PIB_INDEX)
        soup = BeautifulSoup(listing.text, "html.parser")
        links = set()
        for anchor in soup.select("a[href]"):
            title = anchor.get_text(" ", strip=True)
            link = urljoin(PIB_INDEX, anchor["href"])
            if (urlparse(link).hostname in {"pib.gov.in", "www.pib.gov.in"}
                    and re.search(r"PRID=\d+", link, re.I)
                    and re.search(r"schedul", title, re.I)
                    and re.search(r"Legislative Assembl|Lok Sabha|Rajya Sabha|Council of States|House of the People", title, re.I)):
                links.add(link)
        parsed_count = 0
        for link in sorted(links, key=lambda url:int(re.search(r"PRID=(\d+)",url,re.I)[1]), reverse=True)[:4]:
            try:
                page = read(link)
                parsed = parse_schedule(page.text, link, assembly_snapshot=assembly_snapshot)
                events = {identity: value for identity, value in events.items() if identity[0] != link}
                for event in parsed:
                    events[key(event)] = event
                    parsed_count += 1
            except Exception as error:
                errors.append("PIB: " + str(error)[:100])
        updated.update(fallback_checked_at=str(today), fallback_source=PIB_INDEX, fallback_schedules=parsed_count)
    except Exception as fallback_error:
        errors.append("PIB: " + str(fallback_error)[:100])
    updated['events'] = list(events.values())
    if errors:
        updated['refresh_error'] = '; '.join(errors[:3])
    else:
        updated.pop('refresh_error',None)
    _atomic_save(path, updated)
    return updated
