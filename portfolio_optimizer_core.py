from __future__ import annotations

"""
UI-free access to the existing portfolio optimizer implementation.

This compatibility module deliberately does not copy or rewrite the optimizer
formulas. It reads ``portfolio_rebalancer_database.py`` and loads only its imports,
constants, helper functions, and classes that appear before the file's UI section.
Top-level Streamlit page statements are never executed.

This is an interim extraction boundary. Once characterization tests prove parity,
the selected functions can be moved here physically without changing their bodies.
"""

import ast
from pathlib import Path
from types import CodeType
from typing import Final


SOURCE_FILENAME: Final = "portfolio_rebalancer_database.py"
UI_MARKER: Final = "# UI"

PUBLIC_CORE_FUNCTIONS: Final = (
    "get_latest_price_map",
    "rebalance_plan_multi",
    "run_portfolio_analysis_multi",
)


def _source_path() -> Path:
    path = Path(__file__).resolve().with_name(SOURCE_FILENAME)
    if not path.is_file():
        raise RuntimeError(
            f"{SOURCE_FILENAME} must be beside portfolio_optimizer_core.py"
        )
    return path


def _ui_boundary(source: str) -> int:
    """Return the line where the interactive page starts."""
    lines = source.splitlines()
    for index, line in enumerate(lines, start=1):
        if UI_MARKER in line and line.lstrip().startswith("#"):
            return index
    raise RuntimeError(
        f"Could not find the '# UI' boundary in {SOURCE_FILENAME}; refusing "
        "to risk executing interactive page code."
    )


def _is_safe_module_node(node: ast.AST, boundary: int) -> bool:
    """Select declarations needed by functions, excluding page-side effects."""
    if getattr(node, "lineno", boundary) >= boundary:
        return False

    return isinstance(
        node,
        (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Assign,
            ast.AnnAssign,
        ),
    )


def _compiled_core(source: str, filename: str) -> CodeType:
    parsed = ast.parse(source, filename=filename)
    boundary = _ui_boundary(source)
    selected = [
        node for node in parsed.body if _is_safe_module_node(node, boundary)
    ]

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, filename=filename, mode="exec")


def _load_core_namespace() -> dict:
    path = _source_path()
    source = path.read_text(encoding="utf-8")
    namespace = {
        "__builtins__": __builtins__,
        "__file__": str(path),
        "__name__": "_portfolio_optimizer_legacy_core",
        "__package__": None,
    }
    exec(_compiled_core(source, str(path)), namespace, namespace)

    missing = [name for name in PUBLIC_CORE_FUNCTIONS if name not in namespace]
    if missing:
        raise RuntimeError(
            "Required optimizer functions were not found: " + ", ".join(missing)
        )
    return namespace


_CORE = _load_core_namespace()

# These are the stable public entry points used by the scheduled adapter.
run_portfolio_analysis_multi = _CORE["run_portfolio_analysis_multi"]
rebalance_plan_multi = _CORE["rebalance_plan_multi"]
get_latest_price_map = _CORE["get_latest_price_map"]

# Export the lower-level calculation helpers needed for characterization tests and
# for the later physical extraction. Private UI/rendering helpers stay unexported.
for _name, _value in _CORE.items():
    if callable(_value) and not _name.startswith("__"):
        globals().setdefault(_name, _value)


__all__ = sorted(
    name
    for name, value in globals().items()
    if callable(value) and not name.startswith("_")
)
