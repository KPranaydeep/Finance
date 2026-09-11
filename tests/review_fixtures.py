from public_review.core import freeze


def policy():
    """Fresh, deterministic test data; never read the owner's production policy."""
    return {
        "policy_version": "test-policy-v2",
        "policy_approved": True,
        "tax_profile": "resident_individual_regular_delivery",
        "tariff_verified_on": "2026-09-09",
        "tariff_max_age_days": 90,
        "calendar_verified_through": "2026-12-31",
        "capital_inr": None,
        "entry_wait_after_open_minutes": 60,
        "entry_quote_interval": "1m",
        "entry_max_quote_delay_minutes": 330,
        "assessment_wait_after_close_minutes": 30,
        "slab_rate": .30,
        "surcharge_rate": 0.,
        "slippage_bps": 10.,
        "target_xirr": 1.,
        "drawdown_limit": .15,
        "concentration_limit": .50,
        "drift_limit": .05,
        "min_annual_improvement": .06,
        "crossing_probability": .20,
        "max_review_sessions": 20,
        "simulation_paths": 1000,
        "block_length": 5,
        "seed": 19473,
        "validation_train": 252,
        "validation_horizon": 20,
        "validation_min_folds": 20,
        "validation_max_brier": .20,
        "history_years": 3,
        "instrument_kinds": {"A.NS": "equity", "B.NS": "listed_non_equity_etf"},
    }


def baseline():
    p = policy()
    return freeze({"publication_id": "PUB-TEST", "basket_id": "TEST", "portfolio_version": 1,
                   "published_at": "2026-05-01T10:00:00+00:00"},
                  {"A.NS": .5, "B.NS": .5}, {"A.NS": 100., "B.NS": 50.},
                  "2026-05-04", 10000., p["instrument_kinds"], p, "2026-05-04T12:00:00+00:00")
