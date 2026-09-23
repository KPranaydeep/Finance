"""Mobile-readable share image for the public target allocation."""

from __future__ import annotations

from functools import lru_cache
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


@lru_cache(maxsize=32)
def render_allocation_card(
    portfolio_version: str,
    publication_date: str,
    rows: tuple[tuple[str, float, float | None, str], ...],
    *,
    public_url: str = PUBLIC_PORTFOLIO_URL,
) -> bytes:
    """Render the complete target allocation as a 1080×1350 PNG."""
    if not rows:
        raise ValueError("ALLOCATION_CARD_REQUIRES_ROWS")

    paper = "#f5f0e6"
    ink = "#29251f"
    muted = "#6b665e"
    accent = "#9f4339"
    rule = "#d8cfbf"
    figure = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=paper)
    figure.patches.extend(
        [
            Rectangle(
                (0.035, 0.025), 0.93, 0.95,
                transform=figure.transFigure,
                facecolor="none", edgecolor=ink, linewidth=1.2,
            ),
            Rectangle(
                (0.044, 0.034), 0.912, 0.932,
                transform=figure.transFigure,
                facecolor="none", edgecolor=rule, linewidth=0.8,
            ),
        ]
    )

    figure.text(0.075, 0.925, "PUBLIC PORTFOLIO", fontsize=18,
                fontweight="bold", color=accent, family="sans-serif")
    figure.text(0.075, 0.875, "TARGET ALLOCATION", fontsize=29,
                fontweight="bold", color=ink, family="serif")
    figure.text(
        0.075, 0.835,
        f"{portfolio_version}  ·  {publication_date}  ·  {len(rows)} securities",
        fontsize=13.5, color=muted, family="sans-serif",
    )

    columns = (0.075, 0.455, 0.625, 0.790)
    for x, label in zip(columns, ("SECURITY", "TARGET", "INR CLOSE", "LISTING")):
        figure.text(x, 0.785, label, fontsize=11.5, fontweight="bold",
                    color=muted, family="sans-serif")
    figure.lines.append(
        plt.Line2D((0.075, 0.925), (0.766, 0.766), transform=figure.transFigure,
                   color=ink, linewidth=1.1)
    )

    first_y, last_y = 0.738, 0.140
    step = (first_y - last_y) / max(len(rows) - 1, 1)
    row_font = 12.5 if len(rows) > 22 else 14
    for index, (ticker, weight, price, listing) in enumerate(rows):
        y = first_y - index * step
        figure.text(columns[0], y, ticker, fontsize=row_font, fontweight="bold",
                    color=ink, family="monospace", va="center")
        figure.text(columns[1], y, f"{weight:.0%}", fontsize=row_font,
                    color=accent, family="sans-serif", va="center")
        figure.text(columns[2], y, _price_text(price), fontsize=row_font,
                    color=ink, family="sans-serif", va="center")
        figure.text(columns[3], y, listing, fontsize=row_font - 1,
                    color=muted, family="sans-serif", va="center")
        if index < len(rows) - 1:
            line_y = y - step * 0.50
            figure.lines.append(
                plt.Line2D((0.075, 0.925), (line_y, line_y),
                           transform=figure.transFigure, color=rule, linewidth=0.55)
            )

    total_weight = sum(row[1] for row in rows)
    figure.text(0.075, 0.095, f"Total target  {total_weight:.0%}", fontsize=13,
                fontweight="bold", color=ink, family="sans-serif")
    figure.text(
        0.075, 0.064,
        "Reference closes are shown in INR · model allocation, not trade instructions",
        fontsize=10.5, color=muted, family="sans-serif", style="italic",
    )
    figure.text(0.925, 0.064, public_url, fontsize=9.5, color=accent,
                family="sans-serif", ha="right")

    output = BytesIO()
    figure.savefig(output, format="png", dpi=100, facecolor=paper,
                   bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return output.getvalue()
