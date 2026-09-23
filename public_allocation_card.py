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


def _change_summary(
    changes: tuple[tuple[str, float], ...], *, empty_text: str
) -> tuple[str, str]:
    if not changes:
        return empty_text, "—"
    total = sum(weight for _, weight in changes)
    headline = f"{len(changes)} securities · {total:.0%} total"
    # Three sized names remain legible within one half-width mobile panel.
    visible = changes[:3]
    details = "  ·  ".join(f"{ticker} {weight:.0%}" for ticker, weight in visible)
    if len(changes) > len(visible):
        details += f"  ·  +{len(changes) - len(visible)} more"
    return headline, details


@lru_cache(maxsize=32)
def render_allocation_card(
    portfolio_version: str,
    publication_date: str,
    rows: tuple[tuple[str, float, float | None, str], ...],
    changes: tuple[
        str,
        tuple[tuple[str, float], ...],
        tuple[tuple[str, float], ...],
    ] | None = None,
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

    table_header_y = 0.785
    first_y, last_y = 0.738, 0.140
    if changes is not None:
        previous_version, entries, exits = changes
        entry_headline, entry_details = _change_summary(
            entries, empty_text="No new entries"
        )
        exit_headline, exit_details = _change_summary(
            exits, empty_text="No exits"
        )
        figure.text(0.075, 0.792, f"CHANGES SINCE {previous_version}",
                    fontsize=10.5, fontweight="bold", color=muted,
                    family="sans-serif")
        figure.text(0.075, 0.758, "ENTRIES", fontsize=11.5,
                    fontweight="bold", color=accent, family="sans-serif")
        figure.text(0.075, 0.730, entry_headline, fontsize=12.5,
                    fontweight="bold", color=ink, family="sans-serif")
        figure.text(0.075, 0.704, entry_details, fontsize=9.5,
                    color=muted, family="sans-serif")
        figure.text(0.535, 0.758, "EXITS", fontsize=11.5,
                    fontweight="bold", color=ink, family="sans-serif")
        figure.text(0.535, 0.730, exit_headline, fontsize=12.5,
                    fontweight="bold", color=ink, family="sans-serif")
        figure.text(0.535, 0.704, exit_details, fontsize=9.5,
                    color=muted, family="sans-serif")
        table_header_y = 0.655
        first_y, last_y = 0.615, 0.140

    columns = (0.075, 0.455, 0.625, 0.790)
    for x, label in zip(columns, ("SECURITY", "TARGET", "INR CLOSE", "LISTING")):
        figure.text(x, table_header_y, label, fontsize=11.5, fontweight="bold",
                    color=muted, family="sans-serif")
    figure.lines.append(
        plt.Line2D(
            (0.075, 0.925), (table_header_y - 0.019, table_header_y - 0.019),
            transform=figure.transFigure,
                   color=ink, linewidth=1.1)
    )

    step = (first_y - last_y) / max(len(rows) - 1, 1)
    row_font = 11.5 if changes is not None and len(rows) > 22 else (
        12.5 if len(rows) > 22 else 14
    )
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
        "Entries/exits compare active publications · not executed trades",
        fontsize=10.5, color=muted, family="sans-serif", style="italic",
    )
    figure.text(0.925, 0.064, public_url, fontsize=9.5, color=accent,
                family="sans-serif", ha="right")

    output = BytesIO()
    figure.savefig(output, format="png", dpi=100, facecolor=paper,
                   bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return output.getvalue()
