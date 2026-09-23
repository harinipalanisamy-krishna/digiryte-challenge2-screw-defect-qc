"""
db.py - Supabase connection, authentication, and persistent logging.

This is the ONLY file that talks to Supabase directly. Credentials come
from Streamlit secrets (st.secrets), never hardcoded here or in Git.

Tables expected (already created by you in Supabase):
  users:
    id, username (unique), password_hash, role ('operator'/'supervisor'/'admin'), created_at
  inspections:
    id, username, filename, label, confidence, threshold_used, inspected_at
"""

import streamlit as st
import bcrypt
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client:
    """
    Creates one Supabase client for the whole app session (cached, not
    recreated on every rerun). Reads credentials from Streamlit secrets -
    these must be set in the app's Settings > Secrets, never in code.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def verify_login(username: str, password: str):
    """
    Checks a username/password against the users table.
    Returns (True, role) on success, (False, None) on failure.
    Never raises on a wrong password/username - just returns False.
    """
    if not username or not password:
        return False, None

    supabase = get_supabase_client()
    try:
        result = (
            supabase.table("users")
            .select("username, password_hash, role")
            .eq("username", username)
            .execute()
        )
    except Exception as e:
        st.error(f"Could not reach the login database: {e}")
        return False, None

    if not result.data:
        return False, None

    user_row = result.data[0]
    stored_hash = user_row["password_hash"].encode("utf-8")

    try:
        password_matches = bcrypt.checkpw(password.encode("utf-8"), stored_hash)
    except ValueError:
        # stored_hash isn't a valid bcrypt hash - misconfigured row, not a match
        return False, None

    if password_matches:
        return True, user_row["role"]
    return False, None


def log_inspection_to_db(username: str, filename: str, label: str,
                          confidence: float, threshold_used: float):
    """
    Writes one permanent inspection record to the database. Failures here
    are shown as a warning but never crash the app - a logging failure
    shouldn't block the operator from seeing their prediction result.
    """
    supabase = get_supabase_client()
    try:
        supabase.table("inspections").insert({
            "username": username,
            "filename": filename,
            "label": label,
            "confidence": float(confidence),
            "threshold_used": float(threshold_used),
        }).execute()
    except Exception as e:
        st.warning(f"Result shown, but could not save to the permanent log: {e}")


def fetch_all_inspections(limit: int = 500):
    """
    Returns the most recent inspection records across ALL users/sessions,
    newest first - this is the persistent, all-time history, separate from
    the current browser session's in-memory results.
    """
    supabase = get_supabase_client()
    try:
        result = (
            supabase.table("inspections")
            .select("*")
            .order("inspected_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data
    except Exception as e:
        st.warning(f"Could not load inspection history: {e}")
        return []


def fetch_all_users():
    """Returns every user account (username, role, created_at) - admin only feature."""
    supabase = get_supabase_client()
    try:
        result = (
            supabase.table("users")
            .select("username, role, created_at")
            .order("created_at", desc=False)
            .execute()
        )
        return result.data
    except Exception as e:
        st.warning(f"Could not load user list: {e}")
        return []


def update_user_role(username: str, new_role: str):
    """Changes a user's role - admin only feature. Returns True on success."""
    supabase = get_supabase_client()
    try:
        supabase.table("users").update({"role": new_role}).eq("username", username).execute()
        return True
    except Exception as e:
        st.error(f"Could not update role: {e}")
        return False


def hash_password(plain_password: str) -> str:
    """
    Utility for creating new user accounts (used by the standalone
    create_user.py helper script, not by the app itself at runtime).
    """
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
