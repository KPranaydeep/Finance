from io import BytesIO

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from public_allocation_card import (
    _wrapped_ticker_lines,
    listing_descriptor,
    render_allocation_card,
    render_buy_plan_card,
    render_changes_card,
)


def test_change_paragraph_wraps_tickers_without_repeating_weights():
    lines = _wrapped_ticker_lines(
        (("FIRST.NS", .10), ("SECOND.NS", .07), ("THIRD", .03)), width=20
    )

    assert lines == ("FIRST.NS · SECOND.NS", "THIRD")
    rendered = " ".join(lines)
    assert all(ticker in rendered for ticker in ("FIRST.NS", "SECOND.NS", "THIRD"))
    assert "%" not in rendered


def test_listing_descriptor_distinguishes_indian_and_us_quotes():
    assert listing_descriptor("SBC.NS", "INR") == "India · INR"
    assert listing_descriptor("AXTI", "USD") == "U.S. · USD"
    assert listing_descriptor("CASH", "INR") == "Cash · INR"
    assert listing_descriptor("UNKNOWN", None) == "Overseas · —"


@pytest.mark.parametrize(
    ("count", "expected_height"),
    ((1, 2000), (21, 2000), (31, 2316)),
)
def test_allocation_card_is_exact_portrait_png(count, expected_height):
    rows = tuple(
        (f"SECURITY{i}.NS", 0.04, 100.0 + i, "India · INR")
        for i in range(count)
    )
    image = render_allocation_card(
        "P008",
        "2026-09-15",
        rows,
        (
            "P007",
            (("ENTRY.NS", .04), ("USENTRY", .03)),
            (("EXIT.NS", .05),),
        ),
        (
            ("ALLOCATION REVIEW", "02 OCT", "Planning estimate"),
            ("NET SINCE ENTRY", "+1.84%", "After modeled costs"),
            ("28-DAY MEDIAN", "+2.40%", "Through 21 Oct"),
        ),
    )
    pixels = mpimg.imread(BytesIO(image), format="png")

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert pixels.shape[:2] == (expected_height, 1272)
    assert len(image) > 20_000


@pytest.mark.parametrize("with_changes", [False, True])
@pytest.mark.parametrize("metric_count", [0, 1, 2, 3])
def test_portrait_content_fits_and_numeric_columns_align(monkeypatch, with_changes, metric_count):
    rows = tuple(
        (f"SECURITY{i:02}.NS", 1 / 21, None if i == 0 else 123456.78, "India · INR")
        for i in range(21)
    )
    metrics = (
        ("ALLOCATION REVIEW", "02 OCT", "Planning estimate"),
        ("NET SINCE ENTRY", "+1.84%", "After modeled costs"),
        ("28-DAY MEDIAN", "+2.40%", "Through 21 Oct"),
    )[:metric_count]
    savefig = Figure.savefig

    def inspect(figure, *args, **kwargs):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        texts = figure.texts
        # Check actual glyph extents, including the final row and disclosures.
        boxes = [item.get_window_extent(renderer) for item in texts]
        for box in boxes:
            assert figure.bbox.contains(box.x0, box.y0)
            assert figure.bbox.contains(box.x1, box.y1)
        for index, box in enumerate(boxes):
            assert not any(box.overlaps(other) for other in boxes[index + 1:])
        names = [item for item in texts if item.get_text().startswith("SECURITY")
                 and item.get_text() != "SECURITY"]
        assert [item.get_text() for item in names] == [row[0] for row in rows]
        assert all(item.get_ha() == "left" for item in names)
        assert len({item.get_position()[0] for item in names}) == 1
        weights = [
            item for item in texts
            if item.get_text() == "5%" and item.get_color() == "#9f4339"
        ]
        prices = [item for item in texts if item.get_text() in ("₹123,456.78", "—")]
        # The empty change details may also contain an em dash.
        prices = [item for item in prices if item.get_ha() == "right"]
        assert len(weights) == len(prices) == 21
        assert all(item.get_ha() == "right" for item in weights + prices)
        assert len({item.get_position()[0] for item in weights}) == 1
        assert len({item.get_position()[0] for item in prices}) == 1
        assert all(item.get_fontsize() >= 18 for item in names)
        assert len(figure.patches) == 10
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    # Image dimensions must also survive global Matplotlib tight-crop settings.
    with plt.rc_context({"savefig.bbox": "tight"}):
        image = render_allocation_card(
            "P009", "2026-09-23", rows,
            ("P008", (("ENTRY.NS", .05),), ()) if with_changes else None,
            metrics,
        )
    assert mpimg.imread(BytesIO(image), format="png").shape[:2] == (2000, 1272)


def test_empty_allocation_is_rejected():
    with pytest.raises(ValueError, match="ALLOCATION_CARD_REQUIRES_ROWS"):
        render_allocation_card("P009", "2026-09-23", ())


def test_allocation_clusters_india_and_overseas_in_one_image(monkeypatch):
    captured = {}
    savefig = Figure.savefig

    def inspect(figure, *args, **kwargs):
        captured["text"] = [item.get_text() for item in figure.texts]
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    render_allocation_card(
        "P010", "2026-09-30",
        (
            ("INDIA.NS", .60, 100.0, "India · INR"),
            ("VT", .40, 9_300.0, "U.S. · USD"),
        ),
    )
    assert "INDIA · INR" in captured["text"]
    assert "OVERSEAS LISTINGS" in captured["text"]
    assert "INDIA.NS" in captured["text"]
    assert "VT" in captured["text"]


@pytest.mark.parametrize(
    ("count", "expected_height"),
    ((1, 900), (4, 900), (12, 1372), (31, 2000)),
)
def test_buy_plan_card_has_fixed_width_and_content_driven_height(count, expected_height):
    orders = tuple(
        (
            f"BUY{i:02}.NS", i + 1, 100.0 + i,
            (i + 1) * (100.0 + i), "India · INR",
        )
        for i in range(count)
    )
    image = render_buy_plan_card(
        "P010", "2026-09-30", 100_000, 98_500, 1_500, orders,
        mode_label="Target-weight allocation",
    )
    pixels = mpimg.imread(BytesIO(image), format="png")
    assert pixels.shape[:2] == (expected_height, 1272)


def test_empty_buy_plan_card_is_rejected():
    with pytest.raises(ValueError, match="BUY_PLAN_CARD_REQUIRES_ORDERS"):
        render_buy_plan_card(
            "P010", "2026-09-30", 1_000, 0, 1_000, (),
            mode_label="Starter allocation",
        )


def test_buy_plan_clusters_india_and_overseas_in_one_image(monkeypatch):
    captured = {}
    savefig = Figure.savefig

    def inspect(figure, *args, **kwargs):
        captured["text"] = [item.get_text() for item in figure.texts]
        captured["positions"] = {
            item.get_text(): item.get_position() for item in figure.texts
        }
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    render_buy_plan_card(
        "P010", "2026-09-30", 10_000, 9_500, 500,
        (
            ("INDIA.NS", 2, 100.0, 200.0, "India · INR"),
            ("VT", 1, 9_300.0, 9_300.0, "U.S. · USD"),
        ),
        mode_label="Target-weight allocation",
    )
    assert "INDIA · INR" in captured["text"]
    assert "OVERSEAS LISTINGS" in captured["text"]
    assert "INDIA.NS" in captured["text"]
    assert "VT" in captured["text"]
    assert "PRICE" in captured["text"]
    assert "PLANNED VALUE" in captured["text"]
    # Full-width market sections are vertically stacked, not squeezed into
    # two half-width panels.
    assert captured["positions"]["INDIA · INR"][1] > captured["positions"]["OVERSEAS LISTINGS"][1]


def test_all_exits_are_rendered_without_more_truncation(monkeypatch):
    exits = tuple((f"EXIT{i}.NS", .01) for i in range(9))
    captured = {}
    savefig = Figure.savefig

    def inspect(figure, *args, **kwargs):
        captured["text"] = "\n".join(item.get_text() for item in figure.texts)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    render_allocation_card(
        "P010", "2026-09-30",
        tuple((f"KEEP{i}.NS", .05, 100., "India · INR") for i in range(20)),
        ("P009", (), exits),
    )
    for ticker, _ in exits:
        assert ticker in captured["text"]
    assert "+6 more" not in captured["text"]


def test_entry_and_exit_changes_use_restrained_semantic_colors(monkeypatch):
    captured = {}
    savefig = Figure.savefig

    def inspect(figure, *args, **kwargs):
        captured.update({item.get_text(): item.get_color() for item in figure.texts})
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    render_allocation_card(
        "P010", "2026-09-30",
        (("KEEP.NS", 1., 100., "India · INR"),),
        ("P009", (("ENTRY.NS", .05),), (("EXIT.NS", .04),)),
    )

    assert captured["ENTRIES"] == "#3f6b55"
    assert captured["ENTRY.NS"] == "#3f6b55"
    assert captured["EXITS"] == "#9f4339"
    assert captured["EXIT.NS"] == "#9f4339"


def test_large_publication_contains_every_change_and_caps_height(monkeypatch):
    rows = tuple(
        (f"SECURITY{i:02}.NS", 1 / 31, 1234.56 + i, "India · INR")
        for i in range(31)
    )
    entries = tuple((f"ENTRY{i}.NS", .01) for i in range(18))
    exits = tuple((f"EXIT{i}.NS", .01) for i in range(14))
    savefig = Figure.savefig

    def inspect_allocation(figure, *args, **kwargs):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        texts = figure.texts
        boxes = [item.get_window_extent(renderer) for item in texts]
        assert all(figure.bbox.contains(box.x0, box.y0)
                   and figure.bbox.contains(box.x1, box.y1) for box in boxes)
        security_rows = [item for item in texts
                         if item.get_text().startswith("SECURITY")
                         and item.get_text() != "SECURITY"]
        assert len(security_rows) == 31
        assert min(item.get_fontsize() for item in security_rows) >= 18
        rendered = "\n".join(item.get_text() for item in texts)
        assert "Full symbol list on companion Changes card" not in rendered
        assert all(ticker in rendered for ticker, _ in entries + exits)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_allocation)
    image = render_allocation_card(
        "P010", "2026-09-30", rows, ("P009", entries, exits),
        (("ALLOCATION REVIEW", "08 OCT", "Planning estimate"),
         ("NET SINCE ENTRY", "-0.49%", "After modeled costs"),
         ("28-DAY MEDIAN", "+4.99%", "Through 28 Oct")),
    )
    assert mpimg.imread(BytesIO(image), format="png").shape[:2] == (2481, 1272)

    def inspect_changes(figure, *args, **kwargs):
        figure.canvas.draw()
        rendered = "\n".join(item.get_text() for item in figure.texts)
        assert all(ticker in rendered for ticker, _ in entries + exits)
        rows_rendered = [
            item for item in figure.texts
            if item.get_text().startswith(("ENTRY", "EXIT"))
            and item.get_text() not in ("ENTRIES", "EXITS")
        ]
        assert min(item.get_fontsize() for item in rows_rendered) >= 18
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_changes)
    changes_image = render_changes_card(
        "P010", "2026-09-30", "P009", entries, exits,
    )
    assert mpimg.imread(BytesIO(changes_image), format="png").shape[:2] == (2000, 1272)
