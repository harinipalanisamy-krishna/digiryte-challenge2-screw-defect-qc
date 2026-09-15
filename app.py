"""
app.py - Entry point for the BoltGuard AI Screw Defect Detector.

This file only sets up page config, global styling, and navigation between
the two pages. All actual logic lives in:
  - predict.py           (model inference - untouched core logic)
  - common.py            (shared styling / session-state helpers)
  - views/inspect.py     (single-image, long-form inspection page)
  - views/dashboard.py   (session-wide analytics, filters, exports)

Run with: streamlit run app.py
"""

import streamlit as st

from common import inject_global_styles

st.set_page_config(
    page_title="BoltGuard AI — Screw Defect Detector",
    layout="wide",
    page_icon="🔩",
    initial_sidebar_state="expanded",
)

inject_global_styles()

inspect_page = st.Page("views/inspect.py", title="Inspect", icon="🔍", default=True)
dashboard_page = st.Page("views/dashboard.py", title="Dashboard", icon="📊")

pg = st.navigation([inspect_page, dashboard_page])
pg.run()
