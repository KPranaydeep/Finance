"""Validated configuration shared by production trust-layer jobs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class PublicPortfolioConfig:
    environment: str
    basket_id: str
    timezone: str
    performance_calculation_version: str
    forecast_calculation_version: str
    forecast_methodology_version: str
    cache_ttl_seconds: int
    refresh_policy: str
    model_backfill_trading_days: int


def load_public_portfolio_config(*, require_production: bool = False) -> PublicPortfolioConfig:
    environment = os.getenv("PUBLIC_PORTFOLIO_ENV", "TEST").strip().upper()
    if environment not in {"TEST", "PRODUCTION"}:
        raise RuntimeError("PUBLIC_PORTFOLIO_ENV must be TEST or PRODUCTION")
    if require_production and environment != "PRODUCTION":
        raise RuntimeError("This command requires PUBLIC_PORTFOLIO_ENV=PRODUCTION")
    config = PublicPortfolioConfig(
        environment=environment,
        basket_id=os.getenv("PUBLIC_BASKET_ID", "PUBLIC-01").strip(),
        timezone=os.getenv("PUBLIC_PORTFOLIO_TIMEZONE", "Asia/Kolkata").strip(),
        performance_calculation_version=os.getenv("PUBLIC_PERFORMANCE_VERSION", "performance-v1").strip(),
        forecast_calculation_version=os.getenv("PUBLIC_FORECAST_VERSION", "forecast-v1").strip(),
        forecast_methodology_version=os.getenv("PUBLIC_FORECAST_METHOD", "historical-bootstrap-14d-v1").strip(),
        cache_ttl_seconds=int(os.getenv("PUBLIC_CACHE_TTL_SECONDS", "300")),
        refresh_policy=os.getenv("PUBLIC_REFRESH_POLICY", "weekdays-after-market-close").strip(),
        model_backfill_trading_days=int(os.getenv("PUBLIC_MODEL_BACKFILL_TRADING_DAYS", "0")),
    )
    if not config.basket_id or not config.performance_calculation_version or not config.forecast_calculation_version:
        raise RuntimeError("Public portfolio configuration values must not be blank")
    ZoneInfo(config.timezone)
    if config.cache_ttl_seconds < 0:
        raise RuntimeError("PUBLIC_CACHE_TTL_SECONDS must be non-negative")
    if not 0 <= config.model_backfill_trading_days <= 1260:
        raise RuntimeError("PUBLIC_MODEL_BACKFILL_TRADING_DAYS must be between 0 and 1260")
    return config
