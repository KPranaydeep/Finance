"""Temporary operator-only page for an immutable publication correction."""

from __future__ import annotations

import hmac

import streamlit as st

from public_basket_postgres import connect_public_basket_db, get_public_basket_database_url
from public_portfolio_publications import record_publication_correction


st.set_page_config(page_title="Correct public portfolio version", page_icon="🧾")
st.title("🧾 Correct Public Portfolio Version")
st.warning("Temporary operator page. Remove it immediately after the correction is verified.")

try:
    expected_token = str(st.secrets["public_basket"]["publisher_token"]).strip()
except Exception:
    expected_token = ""

if len(expected_token) < 32:
    st.error("Configure public_basket.publisher_token with at least 32 characters.")
    st.stop()

with st.form("publication_correction"):
    token = st.text_input("Publisher token", type="password")
    target_version = st.number_input("Version to void", min_value=1, step=1, value=2)
    authoritative_version = st.number_input("Authoritative version", min_value=1, step=1, value=1)
    reason = st.text_input("Reason", value="P002 was an accidental duplicate of adjacent version P001.")
    phrase = st.text_input("Type VOID P002")
    confirmed = st.checkbox("I understand this creates an immutable correction and does not delete history")
    submit = st.form_submit_button("Record correction", type="primary")

if submit:
    expected_phrase = f"VOID P{int(target_version):03d}"
    if not hmac.compare_digest(token, expected_token):
        st.error("Publisher token is invalid.")
    elif phrase.strip().upper() != expected_phrase or not confirmed:
        st.error(f"Complete the confirmation and type {expected_phrase} exactly.")
    else:
        try:
            with connect_public_basket_db(get_public_basket_database_url()) as conn:
                result = record_publication_correction(
                    conn,
                    basket_id="PUBLIC-01",
                    portfolio_version=int(target_version),
                    correction_type="VOIDED_DUPLICATE",
                    authoritative_version=int(authoritative_version),
                    reason=reason,
                )
            st.success(f"P{int(target_version):03d} is now marked as a voided duplicate.")
            st.json(result)
            st.info("Run the daily trust workflow, verify the dashboard, then delete this page from GitHub.")
        except Exception as exc:
            st.error(f"No correction was recorded: {exc}")
