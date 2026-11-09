from pathlib import Path
import tomllib

from streamlit.testing.v1 import AppTest


def test_native_public_summary_preserves_values_and_pending_explanation():
    app = AppTest.from_file(str(Path(__file__).with_name('portfolio_ui_fixture.py'))).run()
    assert not app.exception
    assert app.title[0].value == 'A risk-aware model portfolio.'
    assert [metric.value for metric in app.metric] == ['27', '4.95%', '06 Nov 2026']
    assert any('Immutable publication · P012' in item.value for item in app.markdown)
    app.checkbox[0].check().run()
    assert not app.exception
    assert app.metric[1].value == 'Unavailable'
    assert any('no outcome is estimated' in item.value for item in app.caption)


def test_private_summary_uses_same_components_and_keeps_diagnostics_optional():
    app = AppTest.from_file(str(Path(__file__).with_name('portfolio_ui_fixture.py'))).run()
    app.segmented_control[0].set_value('Private').run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ['15', '₹385,179.17', '7']
    assert not app.expander[0].proto.expanded
    assert len(app.button) == 3


def test_shared_theme_text_and_primary_button_contrast():
    theme = tomllib.loads((Path(__file__).parents[1] / '.streamlit/config.toml').read_text())['theme']

    def luminance(color):
        rgb = [int(color.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in rgb]
        return sum(value * weight for value, weight in zip(linear, [.2126, .7152, .0722]))

    def contrast(first, second):
        a, b = sorted([luminance(first), luminance(second)])
        return (b + .05) / (a + .05)

    assert contrast(theme['primaryColor'], '#FFFFFF') >= 4.5
    for mode in ('light', 'dark'):
        assert contrast(theme[mode]['textColor'], theme[mode]['backgroundColor']) >= 4.5
        assert contrast(theme[mode]['textColor'], theme[mode]['secondaryBackgroundColor']) >= 4.5
