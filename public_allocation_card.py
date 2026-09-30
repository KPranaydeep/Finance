"""Portrait share image for the public target allocation."""

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
    details = "  ·  ".join(f"{ticker} {weight:.0%}" for ticker, weight in changes)
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
    """Render a mobile-first, exact 1080×1350 allocation share card.

    Holdings flow through two aligned reading columns. This preserves all
    securities and all publication changes without turning a WhatsApp preview
    into an illegibly long screenshot.
    """
    if not rows:
        raise ValueError("ALLOCATION_CARD_REQUIRES_ROWS")

    paper, ink, muted = "#f5f0e6", "#29251f", "#6b665e"
    accent, entry_green = "#9f4339", "#3f6b55"
    exit_red, rule, alternate_row = "#9f4339", "#d8cfbf", "#f0ebe1"
    figure = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=paper)

    def text(x, y, value, size=14, *, color=ink, bold=False,
             align="left", family="sans-serif", valign="center", **kwargs):
        return figure.text(
            x, y, value, fontsize=size, color=color,
            fontweight="bold" if bold else "normal", family=family,
            ha=align, va=valign, **kwargs,
        )

    def line(left, right, y, *, color=rule, width=0.8):
        figure.lines.append(plt.Line2D(
            (left, right), (y, y), transform=figure.transFigure,
            color=color, linewidth=width,
        ))

    left, right = 0.06, 0.94
    text(left, 0.951, "PUBLIC PORTFOLIO", 15, color=accent, bold=True)
    text(left, 0.914, "TARGET ALLOCATION", 25, bold=True, family="serif")
    text(left, 0.881,
         f"{portfolio_version}  ·  {publication_date}  ·  {len(rows)} securities",
         12.5, color=muted)
    line(left, right, 0.858, color=ink, width=1.1)

    band_top = 0.832
    if decision_strip:
        column_width = (right - left) / len(decision_strip)
        for index, (label, value, note) in enumerate(decision_strip):
            x = left + index * column_width
            text(x, band_top, label, 9.5, color=muted, bold=True)
            text(x, band_top - 0.027, value, 18, bold=True,
                 color=accent if index == 0 else ink)
            text(x, band_top - 0.051, note, 9.5, color=muted)
        band_top -= 0.087

    if changes is not None:
        previous_version, entries, exits = changes
        text(left, band_top, f"CHANGES SINCE {previous_version}", 10,
             color=muted, bold=True)
        detail_line_counts = []
        for x, label, values, empty in (
            (left, "ENTRIES", entries, "No new entries"),
            (0.53, "EXITS", exits, "No exits"),
        ):
            change_color = entry_green if label == "ENTRIES" else exit_red
            headline, details = _change_summary(values, empty_text=empty)
            text(x, band_top - 0.025, label, 10.5,
                 color=change_color, bold=True)
            text(x, band_top - 0.048, headline, 11.5, bold=True)
            # Character-aware wrapping preserves every item while allowing
            # short symbols to share more of the available half-width.
            parts = details.split("  ·  ")
            detail_lines, current = [], ""
            for part in parts:
                candidate = part if not current else current + "  ·  " + part
                if current and len(candidate) > 52:
                    detail_lines.append(current)
                    current = part
                else:
                    current = candidate
            if current:
                detail_lines.append(current)
            detail_line_counts.append(max(len(detail_lines), 1))
            text(x, band_top - 0.075, "\n".join(detail_lines), 8.8,
                 color=change_color, linespacing=1.45, valign="top")
        band_top -= 0.105 + max(0, max(detail_line_counts) - 1) * 0.015
    else:
        band_top -= 0.012

    panel_lefts = (left, 0.525)
    panel_rights = (0.475, right)
    split = (len(rows) + 1) // 2
    panels = (rows[:split], rows[split:])
    header_y = band_top
    for panel_left, panel_right, panel_rows in zip(
            panel_lefts, panel_rights, panels):
        if not panel_rows:
            continue
        width = panel_right - panel_left
        columns = (
            panel_left + 0.003,
            panel_left + width * 0.51,
            panel_left + width * 0.78,
            panel_left + width * 0.82,
        )
        for x, label, align in zip(
            columns, ("SECURITY", "TARGET", "INR CLOSE", "LISTING"),
            ("left", "right", "right", "left"),
        ):
            text(x, header_y, label, 8.7, color=muted, bold=True, align=align)
        line(panel_left, panel_right, header_y - 0.014, color=ink, width=0.9)

    first_y, last_y = header_y - 0.038, 0.155
    maximum_panel_rows = max(len(panel) for panel in panels)
    step = min(0.035, (first_y - last_y) /
               max(maximum_panel_rows - 1, 1))
    row_font = min(13.5, step * 1350 * 0.58 * 72 / 100)
    for panel_index, panel_rows in enumerate(panels):
        panel_left, panel_right = panel_lefts[panel_index], panel_rights[panel_index]
        width = panel_right - panel_left
        columns = (
            panel_left + 0.003,
            panel_left + width * 0.51,
            panel_left + width * 0.78,
            panel_left + width * 0.82,
        )
        for index, (ticker, weight, price, listing) in enumerate(panel_rows):
            y = first_y - index * step
            if index % 2:
                figure.patches.append(Rectangle(
                    (panel_left, y - step / 2), width, step,
                    transform=figure.transFigure, facecolor=alternate_row,
                    edgecolor="none", linewidth=0,
                ))
            text(columns[0], y, ticker, row_font, bold=True, family="monospace")
            text(columns[1], y, f"{weight:.0%}", row_font,
                 color=accent, align="right")
            text(columns[2], y, _price_text(price), row_font, align="right")
            text(columns[3], y, listing, max(row_font - 3.5, 7.5), color=muted)

    line(left, right, 0.126)
    total_weight = sum(row[1] for row in rows)
    text(left, 0.106, f"Total target  {total_weight:.0%}", 12.5, bold=True)
    text(left, 0.080,
         "Review dates request reassessment—not an automatic trade. "
         "28-day median is a separate statistical horizon.",
         9.2, color=muted, style="italic")
    text(left, 0.055,
         "Entries/exits compare active publications—not executed trades",
         9.2, color=muted)
    text(left, 0.031, public_url, 9.2, color=accent)

    output = BytesIO()
    # Do not allow an ambient savefig.bbox='tight' to crop the fixed canvas.
    with plt.rc_context({"savefig.bbox": None}):
        figure.savefig(output, format="png", dpi=100, facecolor=paper,
                       bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return output.getvalue()
