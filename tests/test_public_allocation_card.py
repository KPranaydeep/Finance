from io import BytesIO

import matplotlib.image as mpimg

from public_allocation_card import listing_descriptor, render_allocation_card


def test_listing_descriptor_distinguishes_indian_and_us_quotes():
    assert listing_descriptor("SBC.NS", "INR") == "India · INR"
    assert listing_descriptor("AXTI", "USD") == "U.S. · USD"
    assert listing_descriptor("CASH", "INR") == "Cash · INR"
    assert listing_descriptor("UNKNOWN", None) == "Overseas · —"


def test_allocation_card_is_mobile_portrait_png():
    rows = tuple(
        (f"SECURITY{i}.NS", 0.04, 100.0 + i, "India · INR")
        for i in range(25)
    )
    image = render_allocation_card("P008", "2026-09-15", rows)
    pixels = mpimg.imread(BytesIO(image), format="png")

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert pixels.shape[:2] == (1350, 1080)
    assert len(image) > 20_000
