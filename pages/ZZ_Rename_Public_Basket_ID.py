from __future__ import annotations

import hmac
from typing import Any

import streamlit as st
from psycopg import sql

from public_basket_postgres import (
    DEFAULT_BASKET_ID,
    PUBLIC_BASKET_SCHEMA_VERSION,
    connect_public_basket_db,
    get_public_basket_database_url,
)


TARGET_BASKET_ID = "PUBLIC-01"
CONFIRMATION_PHRASE = "RENAME BASKET TO PUBLIC-01"


st.set_page_config(
    page_title="Rename Public Basket ID",
    page_icon="🔐",
    layout="centered",
)


def secret_value(name: str) -> str:
    try:
        section = st.secrets["public_basket"]
        return str(section.get(name, "")).strip()
    except (KeyError, TypeError, AttributeError):
        return ""


def fetch_baskets(conn: Any) -> list[dict[str, Any]]:
    return list(
        conn.execute(
            """
            SELECT basket_id, basket_name, strategy_version, schema_version,
                   base_currency, status, created_at, updated_at
            FROM public_baskets
            ORDER BY created_at, basket_id
            """
        ).fetchall()
    )


def basket_tables(conn: Any) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT c.table_name
        FROM information_schema.columns AS c
        JOIN information_schema.tables AS t
          ON t.table_schema = c.table_schema
         AND t.table_name = c.table_name
        WHERE c.table_schema = 'public'
          AND c.column_name = 'basket_id'
          AND c.table_name <> 'public_baskets'
          AND t.table_type = 'BASE TABLE'
        ORDER BY c.table_name
        """
    ).fetchall()
    return [str(row["table_name"]) for row in rows]


def dependent_counts(conn: Any, source_id: str, tables: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for table in tables:
        row = conn.execute(
            sql.SQL("SELECT COUNT(*) AS row_count FROM {} WHERE basket_id = %s").format(
                sql.Identifier(table)
            ),
            (source_id,),
        ).fetchone()
        result[table] = int(row["row_count"])
    return result


def clone_basket(conn: Any, source_id: str, target_id: str) -> None:
    columns = [
        str(row["column_name"])
        for row in conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'public_baskets'
            ORDER BY ordinal_position
            """
        ).fetchall()
    ]
    if "basket_id" not in columns:
        raise RuntimeError("public_baskets.basket_id is missing.")

    select_items = [
        sql.SQL("%s") if column == "basket_id" else sql.Identifier(column)
        for column in columns
    ]
    statement = sql.SQL(
        "INSERT INTO public_baskets ({columns}) "
        "SELECT {values} FROM public_baskets WHERE basket_id = %s"
    ).format(
        columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
        values=sql.SQL(", ").join(select_items),
    )
    result = conn.execute(statement, (target_id, source_id))
    if result.rowcount != 1:
        raise RuntimeError("The source basket was not found during the rename.")


def rename_basket(conn: Any, source_id: str, target_id: str, tables: list[str]) -> None:
    conn.execute("LOCK TABLE public_baskets IN SHARE ROW EXCLUSIVE MODE")

    source = conn.execute(
        "SELECT basket_id FROM public_baskets WHERE basket_id = %s FOR UPDATE",
        (source_id,),
    ).fetchone()
    target = conn.execute(
        "SELECT basket_id FROM public_baskets WHERE basket_id = %s FOR UPDATE",
        (target_id,),
    ).fetchone()
    if source is None:
        raise RuntimeError(f"Source basket {source_id!r} no longer exists.")
    if target is not None:
        target_counts = dependent_counts(conn, target_id, tables)
        if any(target_counts.values()):
            raise RuntimeError(
                f"Target basket {target_id!r} already contains linked records."
            )
        conn.execute("DELETE FROM public_baskets WHERE basket_id = %s", (target_id,))

    # Insert the new parent first, then repoint every direct basket_id reference.
    # This avoids relying on ON UPDATE CASCADE and keeps the operation portable.
    clone_basket(conn, source_id, target_id)
    for table in tables:
        conn.execute(
            sql.SQL("UPDATE {} SET basket_id = %s WHERE basket_id = %s").format(
                sql.Identifier(table)
            ),
            (target_id, source_id),
        )

    result = conn.execute(
        "DELETE FROM public_baskets WHERE basket_id = %s",
        (source_id,),
    )
    if result.rowcount != 1:
        raise RuntimeError("The old basket record could not be removed.")

    remaining_source = conn.execute(
        "SELECT COUNT(*) AS row_count FROM public_baskets WHERE basket_id = %s",
        (source_id,),
    ).fetchone()
    remaining_target = conn.execute(
        "SELECT COUNT(*) AS row_count FROM public_baskets WHERE basket_id = %s",
        (target_id,),
    ).fetchone()
    if int(remaining_source["row_count"]) != 0 or int(remaining_target["row_count"]) != 1:
        raise RuntimeError("Post-rename verification failed; the transaction was rolled back.")


st.title("🔐 Rename Public Basket ID")
st.warning(
    "Temporary operator-only page. Delete this file from the repository after the "
    "rename succeeds. This page never runs the optimizer or creates trades."
)

configured_token = secret_value("migration_token")
if len(configured_token) < 32:
    st.error(
        "A private migration token of at least 32 characters is required at "
        "[public_basket].migration_token in Streamlit secrets."
    )
    st.stop()

entered_token = st.text_input("Migration token", type="password")
if not entered_token or not hmac.compare_digest(entered_token, configured_token):
    st.info("Enter the private migration token to continue.")
    st.stop()

try:
    database_url = get_public_basket_database_url()
    with connect_public_basket_db(database_url) as conn:
        baskets = fetch_baskets(conn)
        tables = basket_tables(conn)
except Exception:
    st.error("Could not inspect the public basket database. No changes were made.")
    st.stop()

if not baskets:
    st.error("No basket records were found. No changes were made.")
    st.stop()

basket_ids = [str(row["basket_id"]) for row in baskets if row["basket_id"] != TARGET_BASKET_ID]
if not basket_ids:
    st.success("PUBLIC-01 is the only basket identity left. There is nothing to migrate.")
    st.info(
        "Make sure DEFAULT_BASKET_ID in public_basket_postgres.py is also "
        "\"PUBLIC-01\", then delete this temporary page and remove migration_token "
        "from Streamlit secrets."
    )
    st.stop()

default_index = basket_ids.index(DEFAULT_BASKET_ID) if DEFAULT_BASKET_ID in basket_ids else 0
source_id = st.selectbox("Existing basket to rename", basket_ids, index=default_index)
source_record = next(row for row in baskets if row["basket_id"] == source_id)

if int(source_record.get("schema_version") or 0) != PUBLIC_BASKET_SCHEMA_VERSION:
    st.error(
        f"The selected basket is schema version {source_record.get('schema_version')}; "
        f"version {PUBLIC_BASKET_SCHEMA_VERSION} is required."
    )
    st.stop()

with connect_public_basket_db(database_url) as conn:
    counts = dependent_counts(conn, source_id, tables)
    target_counts = dependent_counts(conn, TARGET_BASKET_ID, tables)

if any(target_counts.values()):
    st.error(
        "PUBLIC-01 already contains ledger records, so it cannot be safely replaced. "
        "No changes were made."
    )
    st.stop()

st.subheader("Rename plan")
st.json(
    {
        "from": source_id,
        "to": TARGET_BASKET_ID,
        "basket_name": source_record.get("basket_name"),
        "schema_version": source_record.get("schema_version"),
        "linked_rows_by_table": counts,
        "existing_public_01_rows": target_counts,
    },
    expanded=True,
)
st.caption(
    "The basket metadata and every table containing basket_id will be updated in one "
    "database transaction. Any error rolls back the entire operation."
)

confirmed_backup = st.checkbox(
    "I still have the restorable Neon backup branch created before the schema migration."
)
confirmation = st.text_input(
    f'Type exactly: {CONFIRMATION_PHRASE}',
    placeholder=CONFIRMATION_PHRASE,
)
ready = confirmed_backup and confirmation == CONFIRMATION_PHRASE

if st.button("Rename basket to PUBLIC-01", type="primary", disabled=not ready):
    try:
        with connect_public_basket_db(database_url) as conn:
            with conn.transaction():
                current_tables = basket_tables(conn)
                rename_basket(conn, source_id, TARGET_BASKET_ID, current_tables)

        st.success(f"Basket ID changed from {source_id} to {TARGET_BASKET_ID}.")
        st.subheader("Required cleanup")
        st.code('DEFAULT_BASKET_ID = "PUBLIC-01"', language="python")
        st.markdown(
            "1. Change that constant in `public_basket_postgres.py` and commit it.\n"
            "2. Confirm Public Basket Status shows `PUBLIC-01`.\n"
            "3. Delete this temporary page from `pages/`.\n"
            "4. Remove `migration_token` from Streamlit secrets."
        )
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        st.error("The rename failed and was rolled back. No partial rename was committed.")
