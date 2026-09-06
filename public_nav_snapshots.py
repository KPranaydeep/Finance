"""Persist and read complete model histories without joining unrelated runs."""
import hashlib
import json
import math
from datetime import date


def encode_snapshot(rows):
    if not rows:
        raise ValueError("Cannot publish an empty NAV history")
    material=json.loads(json.dumps(rows,default=str,allow_nan=False))
    dates=[date.fromisoformat(row["nav_date"]) for row in material]
    if dates != sorted(set(dates)):
        raise ValueError("NAV dates must be unique and increasing")
    for row in material:
        if not math.isfinite(float(row["nav"])) or float(row["nav"]) <= 0:
            raise ValueError("NAV values must be finite and positive")
    encoded=json.dumps(material,sort_keys=True,separators=(",",":"),allow_nan=False)
    return encoded,hashlib.sha256(encoded.encode()).hexdigest()


def save_nav_snapshot(conn,basket_id,rows,calculation_version):
    """Append a complete history in the caller's transaction.

    daily_nav remains for legacy inspection. It is not the read source for
    model metrics because its per-date keys cannot describe a whole rerun.
    """
    encoded,digest=encode_snapshot(rows)
    conn.execute("""CREATE TABLE IF NOT EXISTS public_model_nav_snapshots (
        snapshot_id BIGSERIAL PRIMARY KEY,
        basket_id TEXT NOT NULL REFERENCES public_baskets(basket_id),
        calculation_version INTEGER NOT NULL,
        nav_rows JSONB NOT NULL,
        payload_sha256 TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE INDEX IF NOT EXISTS model_nav_snapshots_basket
        ON public_model_nav_snapshots (basket_id,snapshot_id DESC)""")
    conn.execute("""INSERT INTO public_model_nav_snapshots
        (basket_id,calculation_version,nav_rows,payload_sha256)
        VALUES (%s,%s,%s::jsonb,%s)""",(basket_id,calculation_version,encoded,digest))
    return digest


def load_nav_snapshot(conn,basket_id):
    """Read one complete run. Never fill absent dates using an older run."""
    exists=conn.execute(
        "SELECT to_regclass('public_model_nav_snapshots') AS relation"
    ).fetchone()
    if not exists or exists["relation"] is None:
        return []
    snapshot=conn.execute("""SELECT nav_rows,payload_sha256
        FROM public_model_nav_snapshots WHERE basket_id=%s
        ORDER BY snapshot_id DESC LIMIT 1""",(basket_id,)).fetchone()
    if not snapshot:
        return []
    rows=snapshot["nav_rows"]
    if isinstance(rows,str):
        rows=json.loads(rows)
    _,digest=encode_snapshot(rows)
    if digest != snapshot["payload_sha256"]:
        raise ValueError("Model NAV snapshot integrity check failed")
    return [{**row,"nav_date":date.fromisoformat(row["nav_date"])} for row in rows]
