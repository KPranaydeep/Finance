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
    """Render all holdings in a continuous table as an exact 1080×2378 PNG.

    The taller canvas gives the review strip and publication changes their own
    space above the holdings, with comfortable row spacing and a fixed footer.
    """
    if not rows:
        raise ValueError("ALLOCATION_CARD_REQUIRES_ROWS")

    paper, ink, muted = "#f5f0e6", "#29251f", "#6b665e"
    accent, rule, alternate_row = "#9f4339", "#d8cfbf", "#f0ebe1"
    figure = plt.figure(figsize=(10.8, 23.78), dpi=100, facecolor=paper)

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

    left, right = 0.06, 0.94
    text(left, 0.957, "PUBLIC PORTFOLIO", 16, color=accent, bold=True)
    text(left, 0.926, "TARGET ALLOCATION", 27, bold=True, family="serif")
    text(left, 0.901,
         f"{portfolio_version}  ·  {publication_date}  ·  {len(rows)} securities",
         14, color=muted)
    line(left, right, 0.880, color=ink, width=1.1)

    band_top = 0.857
    if decision_strip:
        column_width = (right - left) / len(decision_strip)
        for index, (label, value, note) in enumerate(decision_strip):
            x = left + index * column_width
            text(x, band_top, label, 11, color=muted, bold=True)
            text(x, band_top - 0.023, value, 21, bold=True,
                 color=accent if index == 0 else ink)
            text(x, band_top - 0.043, note, 11, color=muted)
        band_top -= 0.075

    if changes is not None:
        previous_version, entries, exits = changes
        text(left, band_top, f"CHANGES SINCE {previous_version}", 11,
             color=muted, bold=True)
        for x, label, values, empty in (
            (left, "ENTRIES", entries, "No new entries"),
            (0.53, "EXITS", exits, "No exits"),
        ):
            headline, details = _change_summary(values, empty_text=empty)
            text(x, band_top - 0.022, label, 12,
                 color=accent if label == "ENTRIES" else muted, bold=True)
            text(x, band_top - 0.041, headline, 13, bold=True)
            # Wrap at separators to preserve every existing summary item.
            # Half-width columns fit two typical tickers per line.
            parts = details.split("  ·  ")
            detail_lines = ["  ·  ".join(parts[i:i + 2])
                            for i in range(0, len(parts), 2)]
            text(x, band_top - 0.064, "\n".join(detail_lines), 10.5,
                 color=muted, linespacing=1.6)
        band_top -= 0.109
    else:
        band_top -= 0.012

    columns = (left + 0.008, 0.475, 0.710, 0.755)
    for x, label, align in zip(
        columns, ("SECURITY", "TARGET", "INR CLOSE", "LISTING"),
        ("left", "right", "right", "left"),
    ):
        text(x, band_top, label, 11.5, color=muted, bold=True, align=align)
    line(left, right, band_top - 0.013, color=ink, width=0.9)

    first_y, last_y = band_top - 0.033, 0.145
    step = min(0.030, (first_y - last_y) / max(len(rows) - 1, 1))
    row_font = min(17, step * 2378 * 0.60 * 72 / 100)
    for index, (ticker, weight, price, listing) in enumerate(rows):
        y = first_y - index * step
        if index % 2:
            figure.patches.append(Rectangle(
                (left, y - step / 2), right - left, step,
                transform=figure.transFigure, facecolor=alternate_row,
                edgecolor="none", linewidth=0,
            ))
        text(columns[0], y, ticker, row_font, bold=True, family="monospace")
        text(columns[1], y, f"{weight:.0%}", row_font,
             color=accent, align="right")
        text(columns[2], y, _price_text(price), row_font, align="right")
        text(columns[3], y, listing, row_font - 3, color=muted)

    line(left, right, 0.113)
    total_weight = sum(row[1] for row in rows)
    text(left, 0.094, f"Total target  {total_weight:.0%}", 14, bold=True)
    text(left, 0.071,
         "Review dates request reassessment—not an automatic trade.\n"
         "28-day median is a separate statistical horizon.",
         10.5, color=muted, style="italic", linespacing=1.6)
    text(left, 0.049,
         "Entries/exits compare active publications—not executed trades",
         10.5, color=muted)
    text(left, 0.031, public_url, 10.5, color=accent)

    output = BytesIO()
    # Do not allow an ambient savefig.bbox='tight' to crop the fixed canvas.
    with plt.rc_context({"savefig.bbox": None}):
        figure.savefig(output, format="png", dpi=100, facecolor=paper,
                       bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return output.getvalue()
