"""Immutable, basket-scoped publication and forecast persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb

from public_portfolio_trust import canonical_json, fingerprint

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS portfolio_publications (
        publication_id TEXT PRIMARY KEY, basket_id TEXT NOT NULL REFERENCES public_baskets(basket_id),
        run_id TEXT NOT NULL, portfolio_version INTEGER NOT NULL, as_of TIMESTAMPTZ NOT NULL,
        calculation_version TEXT NOT NULL, strategy_version TEXT NOT NULL,
        cash_weight DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (cash_weight >= 0 AND cash_weight <= 1),
        published_at TIMESTAMPTZ NOT NULL, publication_status TEXT NOT NULL CHECK (publication_status = 'PUBLISHED'),
        portfolio_fingerprint TEXT NOT NULL, payload_json JSONB NOT NULL,
        UNIQUE (basket_id, portfolio_version), UNIQUE (basket_id, run_id),
        UNIQUE (basket_id, portfolio_fingerprint)
    )""",
    """CREATE TABLE IF NOT EXISTS portfolio_constituents (
        publication_id TEXT NOT NULL REFERENCES portfolio_publications(publication_id),
        ticker TEXT NOT NULL, target_weight DOUBLE PRECISION NOT NULL CHECK (target_weight >= 0 AND target_weight <= 1),
        PRIMARY KEY (publication_id, ticker)
    )""",
    """CREATE TABLE IF NOT EXISTS portfolio_forecasts (
        forecast_id TEXT PRIMARY KEY, basket_id TEXT NOT NULL REFERENCES public_baskets(basket_id),
        publication_id TEXT NOT NULL REFERENCES portfolio_publications(publication_id),
        forecast_date DATE NOT NULL, horizon_days INTEGER NOT NULL CHECK (horizon_days > 0),
        calculation_version TEXT NOT NULL, sample_start DATE NOT NULL, sample_end DATE NOT NULL,
        observation_count INTEGER NOT NULL, forecast_json JSONB NOT NULL,
        actual_end_date DATE, actual_return DOUBLE PRECISION, evaluated_at TIMESTAMPTZ,
        payload_sha256 TEXT NOT NULL, UNIQUE (basket_id, publication_id, forecast_date, horizon_days, calculation_version)
    )""",
    """CREATE TABLE IF NOT EXISTS portfolio_trust_audit (
        audit_id BIGSERIAL PRIMARY KEY, basket_id TEXT NOT NULL REFERENCES public_baskets(basket_id),
        sequence_number BIGINT NOT NULL, event_at TIMESTAMPTZ NOT NULL, entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL, event_type TEXT NOT NULL, payload_json JSONB NOT NULL,
        previous_hash TEXT, event_hash TEXT NOT NULL, UNIQUE (basket_id, sequence_number),
        UNIQUE (basket_id, event_hash)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_publications_current ON portfolio_publications (basket_id, portfolio_version DESC)",
    "CREATE INDEX IF NOT EXISTS idx_forecasts_basket_date ON portfolio_forecasts (basket_id, forecast_date DESC)",
    """CREATE OR REPLACE FUNCTION reject_trust_record_mutation() RETURNS TRIGGER AS $$
       BEGIN RAISE EXCEPTION 'Trust record % is immutable', TG_TABLE_NAME; END; $$ LANGUAGE plpgsql""",
    """DO $$ DECLARE t TEXT; BEGIN FOREACH t IN ARRAY ARRAY['portfolio_publications','portfolio_constituents','portfolio_trust_audit'] LOOP
       IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='immutable_trust_'||t) THEN
       EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reject_trust_record_mutation()', 'immutable_trust_'||t,t);
       END IF; END LOOP; END $$""",
    """CREATE OR REPLACE FUNCTION protect_forecast_record() RETURNS TRIGGER AS $$ BEGIN
       IF TG_OP='DELETE' THEN RAISE EXCEPTION 'Forecasts are immutable'; END IF;
       IF OLD.forecast_id<>NEW.forecast_id OR OLD.basket_id<>NEW.basket_id OR OLD.publication_id<>NEW.publication_id
          OR OLD.forecast_date<>NEW.forecast_date OR OLD.horizon_days<>NEW.horizon_days
          OR OLD.calculation_version<>NEW.calculation_version OR OLD.forecast_json<>NEW.forecast_json
          OR OLD.payload_sha256<>NEW.payload_sha256 OR OLD.actual_return IS NOT NULL
       THEN RAISE EXCEPTION 'Original forecast content is immutable'; END IF;
       RETURN NEW; END; $$ LANGUAGE plpgsql""",
    """DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='protect_portfolio_forecasts') THEN
       CREATE TRIGGER protect_portfolio_forecasts BEFORE UPDATE OR DELETE ON portfolio_forecasts
       FOR EACH ROW EXECUTE FUNCTION protect_forecast_record(); END IF; END $$""",
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
        existing = conn.execute("SELECT * FROM portfolio_publications WHERE basket_id=%s AND run_id=%s", (basket_id, run_id)).fetchone()
        if existing:
            if existing["portfolio_fingerprint"] != digest:
                raise ValueError("run_id already exists with different portfolio content")
            return dict(existing)
        row = conn.execute("SELECT COALESCE(MAX(portfolio_version),0)+1 AS next_version FROM portfolio_publications WHERE basket_id=%s", (basket_id,)).fetchone()
        version = int(row["next_version"])
        now = datetime.now(timezone.utc)
        conn.execute(
            """INSERT INTO portfolio_publications
            (publication_id,basket_id,run_id,portfolio_version,as_of,calculation_version,strategy_version,
             cash_weight,published_at,publication_status,portfolio_fingerprint,payload_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'PUBLISHED',%s,%s)""",
            (publication_id,basket_id,run_id,version,as_of,calculation_version,strategy_version,
             cash_weight,now,digest,Jsonb(material)),
        )
        for item in normalized:
            conn.execute("INSERT INTO portfolio_constituents (publication_id,ticker,target_weight) VALUES (%s,%s,%s)",
                         (publication_id,item["ticker"],item["target_weight"]))
        _audit(conn,basket_id,"portfolio_publication",publication_id,"PORTFOLIO_PUBLISHED",
               {"portfolio_version":version,"portfolio_fingerprint":digest,"run_id":run_id})
    return {"publication_id":publication_id,"basket_id":basket_id,"run_id":run_id,
            "portfolio_version":version,"portfolio_fingerprint":digest,"published_at":now}


def load_trust_records(conn: Any, basket_id: str) -> dict:
    init_trust_schema(conn)
    publications = conn.execute("SELECT * FROM portfolio_publications WHERE basket_id=%s ORDER BY portfolio_version DESC", (basket_id,)).fetchall()
    current = dict(publications[0]) if publications else None
    constituents = [] if not current else conn.execute(
        "SELECT ticker,target_weight FROM portfolio_constituents WHERE publication_id=%s ORDER BY target_weight DESC,ticker",
        (current["publication_id"],),
    ).fetchall()
    forecasts = conn.execute("SELECT * FROM portfolio_forecasts WHERE basket_id=%s ORDER BY forecast_date DESC", (basket_id,)).fetchall()
    audit = conn.execute("SELECT * FROM portfolio_trust_audit WHERE basket_id=%s ORDER BY sequence_number", (basket_id,)).fetchall()
    cash = conn.execute("SELECT event_at,event_type,amount_inr FROM cash_ledger WHERE basket_id=%s ORDER BY event_at", (basket_id,)).fetchall()
    return {"publications":[dict(r) for r in publications],"current":current,
            "constituents":[dict(r) for r in constituents],"forecasts":[dict(r) for r in forecasts],
            "audit":[dict(r) for r in audit],"cash_flows":[dict(r) for r in cash]}


def record_forecast(conn: Any, *, basket_id: str, publication_id: str, forecast_date,
                    calculation_version: str, forecast: dict) -> dict:
    material={"basket_id":basket_id,"publication_id":publication_id,"forecast_date":str(forecast_date),
              "calculation_version":calculation_version,"forecast":forecast}
    digest=fingerprint(material); forecast_id=f"FCST-{digest[:24].upper()}"
    with conn.transaction():
        init_trust_schema(conn)
        existing=conn.execute("""SELECT * FROM portfolio_forecasts WHERE basket_id=%s AND publication_id=%s
            AND forecast_date=%s AND horizon_days=%s AND calculation_version=%s""",
            (basket_id,publication_id,forecast_date,forecast["horizon_days"],calculation_version)).fetchone()
        if existing: return dict(existing)
        conn.execute("""INSERT INTO portfolio_forecasts
            (forecast_id,basket_id,publication_id,forecast_date,horizon_days,calculation_version,sample_start,
             sample_end,observation_count,forecast_json,payload_sha256)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (forecast_id,basket_id,publication_id,forecast_date,forecast["horizon_days"],calculation_version,
             forecast["sample_start"],forecast["sample_end"],forecast["observation_count"],Jsonb(forecast),digest))
        _audit(conn,basket_id,"forecast",forecast_id,"FORECAST_RECORDED",{"payload_sha256":digest,"forecast_date":str(forecast_date)})
    return {"forecast_id":forecast_id,"payload_sha256":digest}


def evaluate_due_forecasts(conn: Any, basket_id: str) -> int:
    """Attach outcomes after the requested number of subsequent NAV observations."""
    pending=conn.execute("SELECT * FROM portfolio_forecasts WHERE basket_id=%s AND actual_return IS NULL",(basket_id,)).fetchall()
    updated=0
    for row in pending:
        observations=conn.execute("""SELECT nav_date,nav FROM daily_nav WHERE basket_id=%s AND nav_date >= %s
            ORDER BY nav_date,calculation_version DESC""",(basket_id,row["forecast_date"])).fetchall()
        by_day={item["nav_date"]:float(item["nav"]) for item in observations}
        ordered=sorted(by_day.items())
        if len(ordered)<=int(row["horizon_days"]): continue
        start,end=ordered[0],ordered[int(row["horizon_days"])]
        actual=end[1]/start[1]-1
        with conn.transaction():
            conn.execute("UPDATE portfolio_forecasts SET actual_end_date=%s,actual_return=%s,evaluated_at=%s WHERE forecast_id=%s AND actual_return IS NULL",
                         (end[0],actual,datetime.now(timezone.utc),row["forecast_id"]))
            _audit(conn,basket_id,"forecast",row["forecast_id"],"FORECAST_EVALUATED",{"actual_end_date":str(end[0]),"actual_return":actual})
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
