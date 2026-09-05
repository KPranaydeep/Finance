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
