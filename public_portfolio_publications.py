"""Immutable, basket-scoped publication and forecast persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb

from public_portfolio_trust import canonical_json, fingerprint

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS public_portfolio_versions (
        publication_id TEXT PRIMARY KEY, basket_id TEXT NOT NULL REFERENCES public_baskets(basket_id),
        run_id TEXT NOT NULL, portfolio_version INTEGER NOT NULL, as_of TIMESTAMPTZ NOT NULL,
        calculation_version TEXT NOT NULL, strategy_version TEXT NOT NULL,
        cash_weight DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (cash_weight >= 0 AND cash_weight <= 1),
        published_at TIMESTAMPTZ NOT NULL, publication_status TEXT NOT NULL CHECK (publication_status = 'PUBLISHED'),
        portfolio_fingerprint TEXT NOT NULL, payload_json JSONB NOT NULL,
        UNIQUE (basket_id, portfolio_version), UNIQUE (basket_id, run_id),
        UNIQUE (basket_id, portfolio_fingerprint)
    )""",
    """CREATE TABLE IF NOT EXISTS public_portfolio_positions (
        publication_id TEXT NOT NULL REFERENCES public_portfolio_versions(publication_id),
        ticker TEXT NOT NULL, target_weight DOUBLE PRECISION NOT NULL CHECK (target_weight >= 0 AND target_weight <= 1),
        PRIMARY KEY (publication_id, ticker)
    )""",
    """CREATE TABLE IF NOT EXISTS public_forecasts (
        forecast_id TEXT PRIMARY KEY, basket_id TEXT NOT NULL REFERENCES public_baskets(basket_id),
        publication_id TEXT NOT NULL REFERENCES public_portfolio_versions(publication_id),
        forecast_timestamp TIMESTAMPTZ NOT NULL, forecast_date DATE NOT NULL, horizon_days INTEGER NOT NULL CHECK (horizon_days > 0),
        calculation_version TEXT NOT NULL, sample_start DATE NOT NULL, sample_end DATE NOT NULL,
        observation_count INTEGER NOT NULL, methodology TEXT NOT NULL,
        median_return DOUBLE PRECISION NOT NULL, lower_50 DOUBLE PRECISION NOT NULL, upper_50 DOUBLE PRECISION NOT NULL,
        lower_90 DOUBLE PRECISION NOT NULL, upper_90 DOUBLE PRECISION NOT NULL,
        probability_positive DOUBLE PRECISION NOT NULL, probability_negative DOUBLE PRECISION NOT NULL,
        probability_loss_gt_threshold DOUBLE PRECISION NOT NULL, loss_threshold DOUBLE PRECISION NOT NULL,
        forecast_json JSONB NOT NULL,
        payload_sha256 TEXT NOT NULL, UNIQUE (basket_id, publication_id, forecast_date, horizon_days, calculation_version)
    )""",
    """CREATE TABLE IF NOT EXISTS public_forecast_realizations (
        realization_id TEXT PRIMARY KEY, forecast_id TEXT NOT NULL UNIQUE REFERENCES public_forecasts(forecast_id),
        actual_start_value DOUBLE PRECISION NOT NULL, actual_end_value DOUBLE PRECISION NOT NULL,
        actual_return DOUBLE PRECISION NOT NULL, realization_date DATE NOT NULL,
        comparison_status TEXT NOT NULL CHECK (comparison_status IN ('COMPLETE','INVALID')),
        created_at TIMESTAMPTZ NOT NULL, payload_sha256 TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS portfolio_trust_audit (
        audit_id BIGSERIAL PRIMARY KEY, basket_id TEXT NOT NULL REFERENCES public_baskets(basket_id),
        sequence_number BIGINT NOT NULL, event_at TIMESTAMPTZ NOT NULL, entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL, event_type TEXT NOT NULL, payload_json JSONB NOT NULL,
        previous_hash TEXT, event_hash TEXT NOT NULL, UNIQUE (basket_id, sequence_number),
        UNIQUE (basket_id, event_hash)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_publications_current ON public_portfolio_versions (basket_id, portfolio_version DESC)",
    "CREATE INDEX IF NOT EXISTS idx_forecasts_basket_date ON public_forecasts (basket_id, forecast_date DESC)",
    """DO $$ BEGIN
       IF to_regclass('public.portfolio_publications') IS NOT NULL THEN
         INSERT INTO public_portfolio_versions
           (publication_id,basket_id,run_id,portfolio_version,as_of,calculation_version,strategy_version,
            cash_weight,published_at,publication_status,portfolio_fingerprint,payload_json)
         SELECT publication_id,basket_id,run_id,portfolio_version,as_of,calculation_version,strategy_version,
                cash_weight,published_at,publication_status,portfolio_fingerprint,payload_json
         FROM portfolio_publications ON CONFLICT DO NOTHING;
       END IF;
       IF to_regclass('public.portfolio_constituents') IS NOT NULL THEN
         INSERT INTO public_portfolio_positions (publication_id,ticker,target_weight)
         SELECT publication_id,ticker,target_weight FROM portfolio_constituents ON CONFLICT DO NOTHING;
       END IF;
       END $$""",
    """CREATE OR REPLACE FUNCTION reject_trust_record_mutation() RETURNS TRIGGER AS $$
       BEGIN RAISE EXCEPTION 'Trust record % is immutable', TG_TABLE_NAME; END; $$ LANGUAGE plpgsql""",
    """DO $$ DECLARE t TEXT; BEGIN FOREACH t IN ARRAY ARRAY['public_portfolio_versions','public_portfolio_positions','public_forecasts','public_forecast_realizations','portfolio_trust_audit'] LOOP
       IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='immutable_trust_'||t) THEN
       EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reject_trust_record_mutation()', 'immutable_trust_'||t,t);
       END IF; END LOOP; END $$""",
)


def init_trust_schema(conn: Any) -> None:
    for statement in SCHEMA:
        conn.execute(statement)


def _audit(conn: Any, basket_id: str, entity_type: str, entity_id: str, event_type: str, payload: dict) -> str:
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"trust:{basket_id}",))
    prior = conn.execute(
        "SELECT sequence_number, event_hash FROM portfolio_trust_audit WHERE basket_id=%s ORDER BY sequence_number DESC LIMIT 1",
        (basket_id,),
    ).fetchone()
    sequence = int(prior["sequence_number"]) + 1 if prior else 1
    previous = prior["event_hash"] if prior else ""
    envelope = {"basket_id": basket_id, "sequence_number": sequence, "entity_type": entity_type,
                "entity_id": entity_id, "event_type": event_type, "payload": payload}
    event_hash = fingerprint({"previous_hash": previous, "event": envelope})
    conn.execute(
        """INSERT INTO portfolio_trust_audit
        (basket_id, sequence_number, event_at, entity_type, entity_id, event_type, payload_json, previous_hash, event_hash)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (basket_id, sequence, datetime.now(timezone.utc), entity_type, entity_id, event_type,
         Jsonb(envelope), previous or None, event_hash),
    )
    return event_hash


def validate_constituents(constituents: list[dict], cash_weight: float) -> list[dict]:
    if not constituents:
        raise ValueError("An approved portfolio must contain constituents")
    normalized = sorted(
        ({"ticker": str(row["ticker"]).strip().upper(), "target_weight": float(row["target_weight"])} for row in constituents),
        key=lambda row: row["ticker"],
    )
    if any(not row["ticker"] or row["target_weight"] < 0 or row["target_weight"] > 1 for row in normalized):
        raise ValueError("Invalid ticker or target weight")
    if len({row["ticker"] for row in normalized}) != len(normalized):
        raise ValueError("Duplicate constituent ticker")
    total = sum(row["target_weight"] for row in normalized) + float(cash_weight)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Constituent and cash weights must sum to 1.0; received {total:.10f}")
    return normalized


def publish_approved_portfolio(conn: Any, *, basket_id: str, run_id: str, as_of: datetime,
                               calculation_version: str, strategy_version: str,
                               constituents: list[dict], cash_weight: float = 0.0) -> dict:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    normalized = validate_constituents(constituents, cash_weight)
    material = {"basket_id": basket_id, "run_id": run_id, "as_of": as_of.astimezone(timezone.utc).isoformat(),
                "calculation_version": calculation_version, "strategy_version": strategy_version,
                "constituents": normalized, "cash_weight": float(cash_weight)}
    digest = fingerprint(material)
    publication_id = f"PUB-{digest[:24].upper()}"
    with conn.transaction():
        init_trust_schema(conn)
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"publication:{basket_id}",))
        existing = conn.execute("SELECT * FROM public_portfolio_versions WHERE basket_id=%s AND run_id=%s", (basket_id, run_id)).fetchone()
        if existing:
            if existing["portfolio_fingerprint"] != digest:
                raise ValueError("run_id already exists with different portfolio content")
            return dict(existing)
        row = conn.execute("SELECT COALESCE(MAX(portfolio_version),0)+1 AS next_version FROM public_portfolio_versions WHERE basket_id=%s", (basket_id,)).fetchone()
        version = int(row["next_version"])
        now = datetime.now(timezone.utc)
        conn.execute(
            """INSERT INTO public_portfolio_versions
            (publication_id,basket_id,run_id,portfolio_version,as_of,calculation_version,strategy_version,
             cash_weight,published_at,publication_status,portfolio_fingerprint,payload_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'PUBLISHED',%s,%s)""",
            (publication_id,basket_id,run_id,version,as_of,calculation_version,strategy_version,
             cash_weight,now,digest,Jsonb(material)),
        )
        for item in normalized:
            conn.execute("INSERT INTO public_portfolio_positions (publication_id,ticker,target_weight) VALUES (%s,%s,%s)",
                         (publication_id,item["ticker"],item["target_weight"]))
        _audit(conn,basket_id,"portfolio_publication",publication_id,"PORTFOLIO_PUBLISHED",
               {"portfolio_version":version,"portfolio_fingerprint":digest,"run_id":run_id})
    return {"publication_id":publication_id,"basket_id":basket_id,"run_id":run_id,
            "portfolio_version":version,"strategy_version":strategy_version,
            "calculation_version":calculation_version,"as_of":as_of,
            "constituents":normalized,"cash_weight":float(cash_weight),
            "publication_fingerprint":digest,"portfolio_fingerprint":digest,
            "published_at":now,"publication_status":"PUBLISHED"}


def load_trust_records(conn: Any, basket_id: str) -> dict:
    init_trust_schema(conn)
    publications = conn.execute("SELECT * FROM public_portfolio_versions WHERE basket_id=%s ORDER BY portfolio_version DESC", (basket_id,)).fetchall()
    current = dict(publications[0]) if publications else None
    constituents = [] if not current else conn.execute(
        "SELECT ticker,target_weight FROM public_portfolio_positions WHERE publication_id=%s ORDER BY target_weight DESC,ticker",
        (current["publication_id"],),
    ).fetchall()
    forecasts = conn.execute("SELECT * FROM public_forecasts WHERE basket_id=%s ORDER BY forecast_date DESC", (basket_id,)).fetchall()
    realizations = conn.execute("""SELECT r.* FROM public_forecast_realizations r JOIN public_forecasts f ON f.forecast_id=r.forecast_id
        WHERE f.basket_id=%s ORDER BY r.realization_date DESC""", (basket_id,)).fetchall()
    audit = conn.execute("SELECT * FROM portfolio_trust_audit WHERE basket_id=%s ORDER BY sequence_number", (basket_id,)).fetchall()
    cash = conn.execute("SELECT event_at,event_type,amount_inr FROM cash_ledger WHERE basket_id=%s ORDER BY event_at", (basket_id,)).fetchall()
    return {"publications":[dict(r) for r in publications],"current":current,
            "constituents":[dict(r) for r in constituents],"forecasts":[dict(r) for r in forecasts],
            "forecast_realizations":[dict(r) for r in realizations],
            "audit":[dict(r) for r in audit],"cash_flows":[dict(r) for r in cash]}


def record_forecast(conn: Any, *, basket_id: str, publication_id: str, forecast_date,
                    calculation_version: str, forecast: dict) -> dict:
    material={"basket_id":basket_id,"publication_id":publication_id,"forecast_date":str(forecast_date),
              "calculation_version":calculation_version,"forecast":forecast}
    digest=fingerprint(material); forecast_id=f"FCST-{digest[:24].upper()}"
    with conn.transaction():
        init_trust_schema(conn)
        existing=conn.execute("""SELECT * FROM public_forecasts WHERE basket_id=%s AND publication_id=%s
            AND forecast_date=%s AND horizon_days=%s AND calculation_version=%s""",
            (basket_id,publication_id,forecast_date,forecast["horizon_days"],calculation_version)).fetchone()
        if existing: return dict(existing)
        now=datetime.now(timezone.utc)
        conn.execute("""INSERT INTO public_forecasts
            (forecast_id,basket_id,publication_id,forecast_timestamp,forecast_date,horizon_days,calculation_version,
             sample_start,sample_end,observation_count,methodology,median_return,lower_50,upper_50,lower_90,upper_90,
             probability_positive,probability_negative,probability_loss_gt_threshold,loss_threshold,forecast_json,payload_sha256)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (forecast_id,basket_id,publication_id,now,forecast_date,forecast["horizon_days"],calculation_version,
             forecast["sample_start"],forecast["sample_end"],forecast["observation_count"],forecast["method"],
             forecast["median_return"],forecast["lower_50"],forecast["upper_50"],forecast["lower_90"],forecast["upper_90"],
             forecast["probability_positive"],forecast["probability_negative"],forecast["probability_loss_gt_threshold"],
             forecast["loss_threshold"],Jsonb(forecast),digest))
        _audit(conn,basket_id,"forecast",forecast_id,"FORECAST_RECORDED",{"payload_sha256":digest,"forecast_date":str(forecast_date)})
    return {"forecast_id":forecast_id,"payload_sha256":digest}


def evaluate_due_forecasts(conn: Any, basket_id: str) -> int:
    """Attach outcomes after the requested number of subsequent NAV observations."""
    pending=conn.execute("""SELECT f.* FROM public_forecasts f LEFT JOIN public_forecast_realizations r ON r.forecast_id=f.forecast_id
        WHERE f.basket_id=%s AND r.forecast_id IS NULL""",(basket_id,)).fetchall()
    updated=0
    for row in pending:
        observations=conn.execute("""SELECT nav_date,nav FROM daily_nav WHERE basket_id=%s AND nav_date >= %s
            ORDER BY nav_date,calculation_version DESC""",(basket_id,row["forecast_date"])).fetchall()
        by_day={item["nav_date"]:float(item["nav"]) for item in observations}
        ordered=sorted(by_day.items())
        if len(ordered)<=int(row["horizon_days"]): continue
        start,end=ordered[0],ordered[int(row["horizon_days"])]
        actual=end[1]/start[1]-1
        realization_material={"forecast_id":row["forecast_id"],"actual_start_value":start[1],"actual_end_value":end[1],
                              "actual_return":actual,"realization_date":str(end[0]),"comparison_status":"COMPLETE"}
        realization_id=f"REAL-{fingerprint(realization_material)[:24].upper()}"
        with conn.transaction():
            conn.execute("""INSERT INTO public_forecast_realizations
                (realization_id,forecast_id,actual_start_value,actual_end_value,actual_return,realization_date,
                 comparison_status,created_at,payload_sha256) VALUES (%s,%s,%s,%s,%s,%s,'COMPLETE',%s,%s)""",
                (realization_id,row["forecast_id"],start[1],end[1],actual,end[0],datetime.now(timezone.utc),fingerprint(realization_material)))
            _audit(conn,basket_id,"forecast_realization",realization_id,"FORECAST_REALIZED",realization_material)
        updated+=1
    return updated


def verify_trust_audit(rows: list[dict], basket_id: str) -> tuple[bool, str]:
    previous = ""
    for expected_sequence, row in enumerate(rows, 1):
        if row.get("basket_id") != basket_id or int(row.get("sequence_number", -1)) != expected_sequence:
            return False, "Audit sequence or basket scope is invalid"
        if (row.get("previous_hash") or "") != previous:
            return False, f"Broken audit link at sequence {expected_sequence}"
        envelope = row["payload_json"]
        expected = fingerprint({"previous_hash": previous, "event": envelope})
        if expected != row.get("event_hash"):
            return False, f"Audit hash mismatch at sequence {expected_sequence}"
        previous = row["event_hash"]
    return (bool(rows), f"Verified {len(rows)} basket-scoped records" if rows else "No trust records yet")
