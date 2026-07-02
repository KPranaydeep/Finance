"""Application configuration and benchmark catalogue."""

INDEX_CATALOG = {
    "NIFTY 50": {
        "symbol": "^NSEI",
        "nse_name": "NIFTY 50",
        "kind": "Index",
        "description": "Large-cap diversified Indian equities; suitable for portfolios dominated by major companies.",
    },
    "NIFTY 500": {
        "symbol": "^CRSLDX",
        "nse_name": "NIFTY 500",
        "kind": "Index",
        "description": "Broad-market proxy spanning large, mid and small companies. Availability is validated at runtime.",
    },
    "NIFTY Midcap 100": {
        "symbol": "^CNXMidcap",
        "nse_name": "NIFTY MIDCAP 100",
        "kind": "Index",
        "description": "Mid-cap benchmark; useful when the portfolio has a substantial mid-cap allocation.",
    },
    "NIFTY Next 50 (ETF proxy)": {
        "symbol": "JUNIORBEES.NS",
        "nse_name": "NIFTY NEXT 50",
        "kind": "ETF proxy",
        "description": "Tradable proxy for companies immediately below the NIFTY 50 universe.",
    },
    "NIFTY Midcap 150 (ETF proxy)": {
        "symbol": "MID150BEES.NS",
        "nse_name": "NIFTY MIDCAP 150",
        "kind": "ETF proxy",
        "description": "Tradable proxy for a diversified mid-cap portfolio.",
    },
    "NIFTY Smallcap 250 (ETF proxy)": {
        "symbol": "SMALLCAP.NS",
        "nse_name": "NIFTY SMALLCAP 250",
        "kind": "ETF proxy",
        "description": "Tradable proxy for diversified small-cap exposure; validate symbol/data availability.",
    },
    "NIFTY Bank": {
        "symbol": "^NSEBANK",
        "nse_name": "NIFTY BANK",
        "kind": "Index",
        "description": "For portfolios concentrated in large and liquid banking stocks.",
    },
    "NIFTY IT": {
        "symbol": "^CNXIT",
        "nse_name": "NIFTY IT",
        "kind": "Index",
        "description": "For portfolios concentrated in Indian information-technology companies.",
    },
    "NIFTY Auto": {
        "symbol": "^CNXAUTO",
        "nse_name": "NIFTY AUTO",
        "kind": "Index",
        "description": "For portfolios concentrated in automobile and related companies.",
    },
    "NIFTY FMCG": {
        "symbol": "^CNXFMCG",
        "nse_name": "NIFTY FMCG",
        "kind": "Index",
        "description": "For portfolios concentrated in fast-moving consumer-goods companies.",
    },
    "NIFTY Pharma": {
        "symbol": "^CNXPHARMA",
        "nse_name": "NIFTY PHARMA",
        "kind": "Index",
        "description": "For portfolios concentrated in pharmaceutical companies.",
    },
    "NIFTY Metal": {
        "symbol": "^CNXMETAL",
        "nse_name": "NIFTY METAL",
        "kind": "Index",
        "description": "For portfolios concentrated in metals and mining companies.",
    },
    "NIFTY Realty": {
        "symbol": "^CNXREALTY",
        "nse_name": "NIFTY REALTY",
        "kind": "Index",
        "description": "For portfolios concentrated in listed real-estate companies.",
    },
    "NIFTY PSU Bank": {
        "symbol": "^CNXPSUBANK",
        "nse_name": "NIFTY PSU BANK",
        "kind": "Index",
        "description": "For portfolios concentrated in public-sector banks.",
    },
    "NIFTY Energy": {
        "symbol": "^CNXENERGY",
        "nse_name": "NIFTY ENERGY",
        "kind": "Index",
        "description": "For portfolios concentrated in energy-sector companies.",
    },
    "NIFTY Infrastructure": {
        "symbol": "^CNXINFRA",
        "nse_name": "NIFTY INFRASTRUCTURE",
        "kind": "Index",
        "description": "For portfolios concentrated in infrastructure-related companies.",
    },
}

POLICY_FLOORS = {
    "Aggressive — 5% minimum reserve": 0.05,
    "Balanced — 10% minimum reserve": 0.10,
    "Defensive — 20% minimum reserve": 0.20,
    "Formula only — no policy floor": 0.00,
}

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
