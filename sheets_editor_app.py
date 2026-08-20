# -*- coding: utf-8 -*-
"""Streamlit app to view and edit any Google Sheet you have access to.

Setup (one-time):
1. Create a Google Cloud service account with the Sheets & Drive API enabled,
   download its JSON key, and share the target Sheet(s) with the service
   account's `client_email` (Editor access).
2. Add the service account JSON to Streamlit secrets under the key
   `gcp_service_account` (see `.streamlit/secrets.toml.example` for the shape).
3. Run: streamlit run sheets_editor_app.py
"""

from __future__ import annotations

import re
from typing import Optional

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError, WorksheetNotFound

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

st.set_page_config(page_title="Sheet Editor", page_icon="📝", layout="wide")


@st.cache_resource(show_spinner=False)
def get_client() -> gspread.Client:
    """Build an authorized gspread client from Streamlit secrets."""
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


def extract_sheet_id(url_or_id: str) -> str:
    """Accept a full Sheet URL or a bare Sheet ID and return the ID."""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url_or_id)
    return match.group(1) if match else url_or_id.strip()


def load_worksheet_df(worksheet: gspread.Worksheet) -> pd.DataFrame:
    values = worksheet.get_all_values()
    if not values:
        return pd.DataFrame()
    header, *rows = values
    width = len(header)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(rows, columns=header)


def save_worksheet_df(worksheet: gspread.Worksheet, df: pd.DataFrame) -> None:
    df = df.fillna("")
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    worksheet.clear()
    worksheet.update(values, value_input_option="USER_ENTERED")


st.title("📝 Google Sheet Editor")

if "gcp_service_account" not in st.secrets:
    st.error(
        "Missing `gcp_service_account` in Streamlit secrets. "
        "Add your service-account JSON to `.streamlit/secrets.toml` "
        "(locally) or the app's Secrets settings (Streamlit Community Cloud)."
    )
    st.stop()

with st.sidebar:
    st.header("Open a sheet")
    sheet_input = st.text_input(
        "Google Sheet URL or ID",
        placeholder="https://docs.google.com/spreadsheets/d/xxxx/edit",
    )
    load_clicked = st.button("Load sheet", type="primary", use_container_width=True)

if "spreadsheet_id" not in st.session_state:
    st.session_state.spreadsheet_id = None

if load_clicked and sheet_input:
    st.session_state.spreadsheet_id = extract_sheet_id(sheet_input)
    st.session_state.pop("worksheet_title", None)

if not st.session_state.spreadsheet_id:
    st.info("Paste a Google Sheet URL (or ID) in the sidebar and click **Load sheet**.")
    st.stop()

client = get_client()

try:
    spreadsheet = client.open_by_key(st.session_state.spreadsheet_id)
except APIError as exc:
    st.error(
        "Could not open the sheet. Make sure it is shared "
        f"(Editor access) with the service account email: "
        f"`{st.secrets['gcp_service_account']['client_email']}`.\n\n{exc}"
    )
    st.stop()

worksheet_titles = [ws.title for ws in spreadsheet.worksheets()]
default_index = 0
if st.session_state.get("worksheet_title") in worksheet_titles:
    default_index = worksheet_titles.index(st.session_state["worksheet_title"])

worksheet_title = st.selectbox("Worksheet (tab)", worksheet_titles, index=default_index)
st.session_state.worksheet_title = worksheet_title

try:
    worksheet = spreadsheet.worksheet(worksheet_title)
except WorksheetNotFound:
    st.error("Worksheet not found.")
    st.stop()

df = load_worksheet_df(worksheet)

st.caption(f"Spreadsheet: **{spreadsheet.title}**  ·  Tab: **{worksheet_title}**  ·  Rows: {len(df)}")

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    key=f"editor_{spreadsheet.id}_{worksheet_title}",
)

col_save, col_reload = st.columns([1, 1])
with col_save:
    if st.button("💾 Save changes to Google Sheet", type="primary", use_container_width=True):
        try:
            save_worksheet_df(worksheet, edited_df)
            st.success("Saved to Google Sheet.")
        except APIError as exc:
            st.error(f"Failed to save: {exc}")
with col_reload:
    if st.button("🔄 Reload from Sheet", use_container_width=True):
        st.rerun()
