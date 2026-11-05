from pathlib import Path


def test_outlook_only_skips_nav_but_keeps_checks_and_verified_snapshot():
    workflow=Path('.github/workflows/public_portfolio_daily.yml').read_text()
    trigger=workflow.split('permissions:')[0]
    assert 'default: full' in trigger
    assert '- full\n' in trigger and '- outlook_only\n' in trigger
    steps=workflow.split('      - ')
    nav=next(step for step in steps if 'run: python update_public_nav.py' in step)
    assert "github.event_name != 'workflow_dispatch'" in nav
    assert "inputs.refresh_scope != 'outlook_only'" in nav
    assert workflow.index('python update_public_forecasts.py')<workflow.index('python production_smoke_test.py')
    assert workflow.index('python production_smoke_test.py')<workflow.index('python export_public_record_snapshot.py')
    for command in ['python update_public_forecasts.py','python production_smoke_test.py',
                    'python export_public_record_snapshot.py','bash scripts/publish_public_snapshot.sh public_record_snapshot.json']:
        assert '\n        if:' not in next(step for step in steps if 'run: '+command in step)
    assert 'publish_approved_portfolio' not in workflow
