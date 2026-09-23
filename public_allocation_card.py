"""Mobile-readable share image for the public target allocation."""

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
    # Three sized names remain legible within one half-width mobile panel.
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
    if decision_strip:
        count = len(decision_strip)
        left, right = 0.075, 0.925
        column_width = (right - left) / count
        for index, (label, value, note) in enumerate(decision_strip):
            x = left + index * column_width
            figure.text(x, 0.800, label, fontsize=9.5, fontweight="bold",
                        color=muted, family="sans-serif")
            figure.text(x, 0.770, value, fontsize=15.5, fontweight="bold",
                        color=accent if index == 0 else ink, family="sans-serif")
            figure.text(x, 0.746, note, fontsize=9.5,
                        color=muted, family="sans-serif")
            if index:
                divider_x = x - 0.018
                figure.lines.append(
                    plt.Line2D((divider_x, divider_x), (0.742, 0.812),
                               transform=figure.transFigure, color=rule,
                               linewidth=0.8)
                )
        table_header_y = 0.695
        first_y = 0.655
    if changes is not None:
        previous_version, entries, exits = changes
        entry_headline, entry_details = _change_summary(
            entries, empty_text="No new entries"
        )
        exit_headline, exit_details = _change_summary(
            exits, empty_text="No exits"
        )
        change_top = 0.690 if decision_strip else 0.792
        figure.text(0.075, change_top, f"CHANGES SINCE {previous_version}",
                    fontsize=10.5, fontweight="bold", color=muted,
                    family="sans-serif")
        figure.text(0.075, change_top - 0.034, "ENTRIES", fontsize=11.5,
                    fontweight="bold", color=accent, family="sans-serif")
        figure.text(0.075, change_top - 0.062, entry_headline, fontsize=12.5,
                    fontweight="bold", color=ink, family="sans-serif")
        figure.text(0.075, change_top - 0.088, entry_details, fontsize=9.5,
                    color=muted, family="sans-serif")
        figure.text(0.535, change_top - 0.034, "EXITS", fontsize=11.5,
                    fontweight="bold", color=ink, family="sans-serif")
        figure.text(0.535, change_top - 0.062, exit_headline, fontsize=12.5,
                    fontweight="bold", color=ink, family="sans-serif")
        figure.text(0.535, change_top - 0.088, exit_details, fontsize=9.5,
                    color=muted, family="sans-serif")
        table_header_y = 0.555 if decision_strip else 0.655
        first_y, last_y = (0.518 if decision_strip else 0.615), 0.140

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
    row_font = 11 if decision_strip and changes is not None and len(rows) > 20 else (
        11.5 if changes is not None and len(rows) > 22 else (
        12.5 if len(rows) > 22 else 14
    ))
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
        0.075, 0.068,
        "Review dates request reassessment—not an automatic trade. 28-day median is a separate statistical horizon.",
        fontsize=8.8, color=muted, family="sans-serif", style="italic",
    )
    figure.text(0.075, 0.047,
                "Entries/exits compare active publications—not executed trades",
                fontsize=8.5, color=muted, family="sans-serif")
    figure.text(0.925, 0.047, public_url, fontsize=8.5, color=accent,
                family="sans-serif", ha="right")

    output = BytesIO()
    figure.savefig(output, format="png", dpi=100, facecolor=paper,
                   bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return output.getvalue()
