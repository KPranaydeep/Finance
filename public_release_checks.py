"""Fail-closed checks for material intentionally exposed by the public app."""

from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEYS=re.compile(r"(password|passwd|secret|token|api.?key|broker|account.?number|database.?url|credential|private.?key)",re.I)
SENSITIVE_VALUES=(re.compile(r"postgres(?:ql)?://",re.I),re.compile(r"(?:[A-Z]:\\Users\\|/home/|/Users/)",re.I),re.compile(r"-----BEGIN .*PRIVATE KEY-----"))
TEST_MARKERS=re.compile(r"(^|[-_ ])(test|fixture|synthetic|fake|development|dev)([-_ ]|$)",re.I)


def inspect_public_data(value: Any, *, production: bool = True, path: str = "$") -> list[str]:
    findings=[]
    if isinstance(value,dict):
        for key,item in value.items():
            child=f"{path}.{key}"
            if SENSITIVE_KEYS.search(str(key)): findings.append(f"Sensitive field name at {child}")
            findings.extend(inspect_public_data(item,production=production,path=child))
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value): findings.extend(inspect_public_data(item,production=production,path=f"{path}[{index}]"))
    elif isinstance(value,str):
        if any(pattern.search(value) for pattern in SENSITIVE_VALUES): findings.append(f"Sensitive value pattern at {path}")
        if production and TEST_MARKERS.search(value): findings.append(f"Non-production marker at {path}")
    return sorted(set(findings))


def prepare_evidence_export(value: dict) -> tuple[dict, list[str]]:
    """Inspect a labelled research export without relaxing production checks.

    The exception is limited to known forecast provenance fields and requires
    actual backfill flags in the loaded NAV. Original records are not changed.
    """
    research=any(isinstance(row,dict) and row.get("is_backfill") is True
                 for row in value.get("nav",[]))
    evidence={**value,"evidence_metadata":{
        "classification":"RESEARCH_SIMULATION" if research else "POST_PUBLICATION_MODEL",
        "contains_backfilled_nav":research,
        "description":(
            "Includes retrospectively simulated history; it is not a live investment track record."
            if research else "Model evidence based on recorded post-publication NAV."
        ),
    }}
    findings=inspect_public_data(evidence,production=True)
    if not research:
        return evidence,findings
    allowed=set()
    for collection in ("forecasts","active_forecasts"):
        for index,row in enumerate(evidence.get(collection,[])):
            payload=row.get("forecast_json") if isinstance(row,dict) else None
            if isinstance(payload,dict) and payload.get("history_source") == "DEVELOPMENT_BACKFILL":
                allowed.add(f"Non-production marker at $.{collection}[{index}].forecast_json.history_source")
    return evidence,[finding for finding in findings if finding not in allowed]
