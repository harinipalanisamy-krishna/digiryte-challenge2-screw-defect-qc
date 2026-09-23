"""
app.py - Entry point for the BoltGuard AI Screw Defect Detector.

This file sets up page config, global styling, a login gate, and
navigation between the two pages. All actual logic lives in:
  - predict.py           (model inference - untouched core logic)
  - common.py            (shared styling / session-state helpers)
  - db.py                (Supabase auth + persistent inspection logging)
  - views/inspect.py     (single-image, long-form inspection page)
  - views/dashboard.py   (session-wide analytics, filters, exports)

Run with: streamlit run app.py
"""

import streamlit as st

from common import inject_global_styles
from db import verify_login

st.set_page_config(
    page_title="BoltGuard AI — Screw Defect Detector",
    layout="wide",
    page_icon="🔩",
    initial_sidebar_state="expanded",
)

inject_global_styles()

# ---------------------------------------------------------------------------
# Login gate - nothing past this point runs until authenticated.
# On success, st.session_state.username and .role are set and used
# throughout the app (e.g. to log who inspected what, and to restrict
# certain actions to supervisor/admin roles).
# ---------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        """
        <div style="max-width:420px; margin: 8vh auto 0 auto; text-align:center;">
            <div style="font-size:2.4rem;">🔩</div>
            <h1 style="margin-bottom:0.2rem;">BoltGuard AI</h1>
            <p style="color:#9aa0ab; margin-top:0;">Sign in to access the QC dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            success, role = verify_login(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.stop()  # nothing below this line runs until login succeeds

# ---------------------------------------------------------------------------
# Authenticated area - navigation is built based on the logged-in role.
# Operators see only Inspect. Supervisors also see Dashboard. Admins see
# Inspect, Dashboard, AND Admin. A page not in this list is completely
# invisible to that role - not just hidden behind a locked button, but
# absent from the sidebar entirely.
# ---------------------------------------------------------------------------
role = st.session_state.get("role", "operator")

inspect_page = st.Page("views/inspect.py", title="Inspect", icon="🔍", default=True)
pages = [inspect_page]

if role in ("supervisor", "admin"):
    dashboard_page = st.Page("views/dashboard.py", title="Dashboard", icon="📊")
    pages.append(dashboard_page)

if role == "admin":
    admin_page = st.Page("views/admin.py", title="Admin", icon="🛡️")
    pages.append(admin_page)

pg = st.navigation(pages)
pg.run()
