from __future__ import annotations

import hmac
import logging
from dataclasses import asdict

import streamlit as st

from migrate_public_basket_v1_to_v2 import (
    MIGRATION_SCRIPT_VERSION,
    apply_migration,
    inspect_migration,
)
from public_basket_postgres import connect_public_basket_db, get_public_basket_database_url


LOGGER = logging.getLogger(__name__)
CONFIRMATION_PHRASE = "MIGRATE PUBLIC BASKET TO V2"


st.set_page_config(
    page_title="Public Basket Schema Migration",
    page_icon="🔐",
    layout="centered",
)

st.title("🔐 Public Basket Schema Migration")
st.caption(f"Migration script: {MIGRATION_SCRIPT_VERSION}")
st.warning(
    "Temporary operator-only page. Remove this file from the repository immediately "
    "after the migration succeeds."
)


def read_migration_token() -> str | None:
    try:
        section = st.secrets.get("public_basket", {})
        value = section.get("migration_token")
        return str(value) if value else None
    except Exception:
        return None


configured_token = read_migration_token()
database_url = get_public_basket_database_url()

if not configured_token:
    st.error(
        "Migration access is disabled. Add a strong migration_token to the existing "
        "[public_basket] section in Streamlit secrets."
    )
    st.code(
        '[public_basket]\n'
        'database_url = "postgresql://..."\n'
        'migration_token = "USE-A-UNIQUE-RANDOM-SECRET-OF-AT-LEAST-32-CHARACTERS"',
        language="toml",
    )
    st.stop()

if len(configured_token) < 32:
    st.error("The configured migration_token must contain at least 32 characters.")
    st.stop()

if not database_url:
    st.error("The durable public-basket PostgreSQL connection is not configured.")
    st.stop()

entered_token = st.text_input("Migration token", type="password")
authorized = bool(entered_token) and hmac.compare_digest(entered_token, configured_token)

if not authorized:
    if entered_token:
        st.error("The migration token is incorrect.")
    st.stop()

st.success("Operator access confirmed.")

if "migration_plan" not in st.session_state:
    st.session_state.migration_plan = None

if st.button("Inspect database only", type="primary", use_container_width=True):
    connection = None
    try:
        connection = connect_public_basket_db(database_url)
        st.session_state.migration_plan = asdict(inspect_migration(connection))
    except Exception:
        LOGGER.exception("Public-basket migration inspection failed")
        st.error("Inspection failed. Check the private Streamlit Cloud logs.")
    finally:
        if connection is not None:
            connection.close()

plan = st.session_state.migration_plan
if plan is None:
    st.info("Inspect the database before enabling the migration controls.")
    st.stop()

st.subheader("Migration plan")
st.json(plan)

if not plan.get("safe_to_apply"):
    st.error(
        "Migration is blocked. The database did not pass the loss-prevention checks. "
        "Do not bypass this result."
    )
    st.stop()

st.success(plan["reason"])
st.write(
    "Before continuing, create a restorable backup or snapshot using your PostgreSQL "
    "provider. This page cannot verify that an external backup exists."
)

backup_confirmed = st.checkbox(
    "I created and verified a restorable PostgreSQL backup."
)
confirmation = st.text_input(
    f'Type exactly: {CONFIRMATION_PHRASE}',
    key="migration_confirmation",
)

ready = backup_confirmed and hmac.compare_digest(confirmation, CONFIRMATION_PHRASE)

if st.button(
    "Apply version-2 migration",
    type="primary",
    disabled=not ready,
    use_container_width=True,
):
    connection = None
    try:
        connection = connect_public_basket_db(database_url)
        with st.spinner("Applying and verifying the atomic migration…"):
            result = apply_migration(connection)
        st.session_state.migration_plan = asdict(result)
        st.cache_data.clear()
        st.success("Schema version 2 was committed successfully.")
        st.error(
            "Required cleanup: remove pages/ZZ_Migrate_Public_Basket_V1_to_V2.py "
            "and delete migration_token from Streamlit secrets now."
        )
        st.balloons()
    except Exception:
        LOGGER.exception("Public-basket schema migration failed")
        st.error(
            "Migration failed and the transaction was rolled back. Check the private "
            "Streamlit Cloud logs before trying again."
        )
    finally:
        if connection is not None:
            connection.close()
