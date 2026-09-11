"""Own append-only PostgreSQL chain; no writes to publication/trade tables."""
from psycopg.types.json import Jsonb
from .core import digest


def init(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS public_review_events (
        seq BIGSERIAL PRIMARY KEY, basket_id TEXT NOT NULL, event_key TEXT NOT NULL,
        kind TEXT NOT NULL, baseline_id TEXT, payload JSONB NOT NULL,
        previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE(basket_id,event_key))""")
    conn.execute("""CREATE INDEX IF NOT EXISTS public_review_latest
        ON public_review_events(basket_id,seq DESC)""")
    conn.execute("""CREATE OR REPLACE FUNCTION public_review_reject_mutation() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'Review events are append-only'; END $$""")
    conn.execute("""DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='public_review_immutable')
        THEN CREATE TRIGGER public_review_immutable BEFORE UPDATE OR DELETE ON public_review_events
        FOR EACH ROW EXECUTE FUNCTION public_review_reject_mutation(); END IF; END $$""")


def read(conn, basket_id):
    if not conn.execute("SELECT to_regclass('public.public_review_events') AS name").fetchone()["name"]:
        return []
    rows = [dict(r) for r in conn.execute("SELECT * FROM public_review_events WHERE basket_id=%s ORDER BY seq", (basket_id,)).fetchall()]
    previous = ""
    for row in rows:
        envelope = {k: row[k] for k in ("basket_id", "event_key", "kind", "baseline_id", "payload", "previous_hash")}
        if row["previous_hash"] != previous or digest(envelope) != row["event_hash"]:
            raise ValueError("Review audit verification failed")
        previous = row["event_hash"]
    return rows


def append(conn, basket_id, key, kind, baseline_id, payload):
    # Caller owns transaction; basket lock serializes chain head and idempotency.
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", ("public-review:" + basket_id,))
    rows = read(conn, basket_id)
    if any(r["event_key"] == key for r in rows):
        return False
    previous = rows[-1]["event_hash"] if rows else ""
    envelope = {"basket_id": basket_id, "event_key": key, "kind": kind, "baseline_id": baseline_id,
                "payload": payload, "previous_hash": previous}
    conn.execute("""INSERT INTO public_review_events
        (basket_id,event_key,kind,baseline_id,payload,previous_hash,event_hash)
        VALUES (%s,%s,%s,%s,%s,%s,%s)""", (basket_id, key, kind, baseline_id, Jsonb(payload), previous, digest(envelope)))
    return True


def latest(rows, kind, baseline_id=None):
    return next((r for r in reversed(rows) if r["kind"] == kind and
                 (baseline_id is None or r["baseline_id"] == baseline_id)), None)
