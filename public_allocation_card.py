"""Landscape share image for the public target allocation."""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PUBLIC_PORTFOLIO_URL = "https://theportfolio.streamlit.app/#target-allocation"


def listing_descriptor(ticker: str, source_currency: str | None) -> str:
    """Return an unambiguous listing market and native quote currency."""
    symbol = str(ticker or "").strip().upper()
    currency = str(source_currency or "").strip().upper() or "—"
    if symbol == "CASH":
        return "Cash · INR"
    if symbol.endswith((".NS", ".BO")):
        return "India · INR"
    if currency == "USD":
        return "U.S. · USD"
    return f"Overseas · {currency}"


def _price_text(value: float | None) -> str:
    if value is None:
        return "—"
    return f"₹{float(value):,.2f}"


def _change_summary(
    changes: tuple[tuple[str, float], ...], *, empty_text: str
) -> tuple[str, str]:
    if not changes:
        return empty_text, "—"
    total = sum(weight for _, weight in changes)
    headline = f"{len(changes)} securities · {total:.0%} total"
    # Keep the publication-change summary consistent with the public page.
    visible = changes[:3]
    details = "  ·  ".join(f"{ticker} {weight:.0%}" for ticker, weight in visible)
    if len(changes) > len(visible):
        details += f"  ·  +{len(changes) - len(visible)} more"
    return headline, details


def render_allocation_card(
    portfolio_version: str,
    publication_date: str,
    rows: tuple[tuple[str, float, float | None, str], ...],
    changes: tuple[
        str,
        tuple[tuple[str, float], ...],
        tuple[tuple[str, float], ...],
    ] | None = None,
    decision_strip: tuple[tuple[str, str, str], ...] = (),
    *,
    public_url: str = PUBLIC_PORTFOLIO_URL,
) -> bytes:
    """Render every holding, in source order, as an exact 2378×1080 PNG.

    Holdings flow down the left table, then down the right table. Keeping the
    review and change bands above both tables leaves room for readable rows.
    """
    if not rows:
        raise ValueError("ALLOCATION_CARD_REQUIRES_ROWS")

    paper, ink, muted = "#f5f0e6", "#29251f", "#6b665e"
    accent, rule, alternate_row = "#9f4339", "#d8cfbf", "#f0ebE1"
    figure = plt.figure(figsize=(23.78, 10.8), dpi=100, facecolor=paper)

    def text(x, y, value, size=14, *, color=ink, bold=False,
             align="left", family="sans-serif", **kwargs):
        return figure.text(
            x, y, value, fontsize=size, color=color,
            fontweight="bold" if bold else "normal", family=family,
            ha=align, va="center", **kwargs,
        )

    def line(left, right, y, *, color=rule, width=0.8):
        figure.lines.append(plt.Line2D(
            (left, right), (y, y), transform=figure.transFigure,
            color=color, linewidth=width,
        ))

    left, right = 0.035, 0.965
    text(left, 0.949, "PUBLIC PORTFOLIO", 14, color=accent, bold=True)
    text(left, 0.899, "TARGET ALLOCATION", 26, bold=True, family="serif")
    text(right, 0.909,
         f"{portfolio_version}  ·  {publication_date}  ·  {len(rows)} securities",
         16, color=muted, align="right")
    line(left, right, 0.857, color=ink, width=1.1)

    band_top = 0.823
    if decision_strip:
        column_width = (right - left) / len(decision_strip)
        for index, (label, value, note) in enumerate(decision_strip):
            x = left + index * column_width
            text(x, band_top, label, 12, color=muted, bold=True)
            text(x, band_top - 0.042, value, 22, bold=True,
                 color=accent if index == 0 else ink)
            text(x, band_top - 0.080, note, 12, color=muted)
        band_top -= 0.128

    if changes is not None:
        previous_version, entries, exits = changes
        text(left, band_top, f"CHANGES SINCE {previous_version}", 12,
             color=muted, bold=True)
        for x, label, values, empty in (
            (0.255, "ENTRIES", entries, "No new entries"),
            (0.620, "EXITS", exits, "No exits"),
        ):
            headline, details = _change_summary(values, empty_text=empty)
            text(x, band_top, label, 12, color=accent if label == "ENTRIES" else muted,
                 bold=True)
            text(x + 0.060, band_top, headline, 13, bold=True)
            text(x, band_top - 0.033, details, 12, color=muted)
        band_top -= 0.086
    else:
        band_top -= 0.012

    # Two balanced table columns preserve source order without compressing
    # all 21 holdings into the height of one portrait-style table.
    split = (len(rows) + 1) // 2
    tables = (rows[:split], rows[split:]) if len(rows) > 1 else (rows,)
    table_width = 0.445 if len(tables) == 2 else right - left
    first_y = band_top - 0.056
    last_y = 0.165
    step = min(0.043, (first_y - last_y) / max(split - 1, 1))
    row_font = min(17, step * 1080 * 0.60 * 72 / 100)
    for table_index, table_rows in enumerate(tables):
        table_left = left + table_index * 0.485
        table_right = table_left + table_width
        columns = (
            table_left + 0.006,
            table_left + table_width * 0.51,
            table_left + table_width * 0.75,
            table_left + table_width * 0.79,
        )
        for x, label, align in zip(
            columns, ("SECURITY", "TARGET", "INR CLOSE", "LISTING"),
            ("left", "right", "right", "left"),
        ):
            text(x, band_top, label, 12, color=muted, bold=True, align=align)
        line(table_left, table_right, band_top - 0.023, color=ink, width=0.9)
        for index, (ticker, weight, price, listing) in enumerate(table_rows):
            y = first_y - index * step
            if index % 2:
                figure.patches.append(Rectangle(
                    (table_left, y - step / 2), table_width, step,
                    transform=figure.transFigure, facecolor=alternate_row,
                    edgecolor="none", linewidth=0,
                ))
            text(columns[0], y, ticker, row_font, bold=True, family="monospace")
            text(columns[1], y, f"{weight:.0%}", row_font,
                 color=accent, align="right")
            text(columns[2], y, _price_text(price), row_font, align="right")
            text(columns[3], y, listing, row_font - 2, color=muted)

    line(left, right, 0.125)
    total_weight = sum(row[1] for row in rows)
    text(left, 0.096, f"Total target  {total_weight:.0%}", 14, bold=True)
    text(right, 0.096, public_url, 12, color=accent, align="right")
    text(left, 0.062,
         "Review dates request reassessment—not an automatic trade. 28-day median is a separate statistical horizon.",
         11.5, color=muted, style="italic")
    text(left, 0.036,
         "Entries/exits compare active publications—not executed trades",
         11.5, color=muted)

    output = BytesIO()
    # Do not allow an ambient savefig.bbox='tight' to crop the fixed canvas.
    with plt.rc_context({"savefig.bbox": None}):
        figure.savefig(output, format="png", dpi=100, facecolor=paper,
                       bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return output.getvalue()
