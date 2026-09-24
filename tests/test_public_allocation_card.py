from io import BytesIO

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from public_allocation_card import listing_descriptor, render_allocation_card


def test_listing_descriptor_distinguishes_indian_and_us_quotes():
    assert listing_descriptor("SBC.NS", "INR") == "India · INR"
    assert listing_descriptor("AXTI", "USD") == "U.S. · USD"
    assert listing_descriptor("CASH", "INR") == "Cash · INR"
    assert listing_descriptor("UNKNOWN", None) == "Overseas · —"


@pytest.mark.parametrize("count", [1, 21, 25])
def test_allocation_card_is_exact_landscape_png(count):
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
    assert pixels.shape[:2] == (1080, 2378)
    assert len(image) > 20_000


@pytest.mark.parametrize("with_changes", [False, True])
@pytest.mark.parametrize("metric_count", [0, 1, 2, 3])
def test_landscape_content_fits_and_numeric_columns_align(monkeypatch, with_changes, metric_count):
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
        for table in (names[:11], names[11:]):
            assert len({item.get_position()[0] for item in table}) == 1
        weights = [item for item in texts if item.get_text() == "5%"]
        prices = [item for item in texts if item.get_text() in ("₹123,456.78", "—")]
        # The empty change details may also contain an em dash.
        prices = [item for item in prices if item.get_ha() == "right"]
        assert len(weights) == len(prices) == 21
        assert all(item.get_ha() == "right" for item in weights + prices)
        assert len({item.get_position()[0] for item in weights}) == 2
        assert len({item.get_position()[0] for item in prices}) == 2
        assert all(item.get_fontsize() >= 15 for item in names)
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
    assert mpimg.imread(BytesIO(image), format="png").shape[:2] == (1080, 2378)


def test_empty_allocation_is_rejected():
    with pytest.raises(ValueError, match="ALLOCATION_CARD_REQUIRES_ROWS"):
        render_allocation_card("P009", "2026-09-23", ())
