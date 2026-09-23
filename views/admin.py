"""
views/admin.py - the "Admin" page: user management and all-time,
cross-user inspection history pulled from the permanent database.

Visible only to the admin role (enforced in app.py's navigation) - this
page does not even appear in the sidebar for operator or supervisor
accounts.
"""

import streamlit as st

from common import inject_global_styles, render_page_header, render_sidebar
from db import fetch_all_users, update_user_role, fetch_all_inspections

inject_global_styles()
render_sidebar()

render_page_header(
    tag="🛡️ Admin Only",
    title="Admin",
    subtitle="User management and all-time inspection history across every account.",
)

# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------
st.markdown("## 👥 User Accounts")

users = fetch_all_users()

if not users:
    st.info("No user accounts found (or the database couldn't be reached).")
else:
    for user in users:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{user['username']}**")
        with col2:
            new_role = st.selectbox(
                "Role",
                options=["operator", "supervisor", "admin"],
                index=["operator", "supervisor", "admin"].index(user["role"]),
                key=f"role_select_{user['username']}",
                label_visibility="collapsed",
            )
        with col3:
            if new_role != user["role"]:
                if st.button("Save", key=f"save_{user['username']}", use_container_width=True):
                    if update_user_role(user["username"], new_role):
                        st.success(f"Updated {user['username']} to {new_role}.")
                        st.rerun()
            else:
                st.caption("—")

st.caption("Changing a role here takes effect the next time that user logs in "
           "(it does not force out an already-logged-in session).")

st.markdown("---")

# ---------------------------------------------------------------------------
# All-time history across every user
# ---------------------------------------------------------------------------
st.markdown("## 🗄️ All-Time History (All Users)")
st.caption("Pulled live from the permanent database — persists across sessions, "
           "browser refreshes, and different users.")

all_records = fetch_all_inspections()

if not all_records:
    st.info("No permanent inspection records yet.")
else:
    total = len(all_records)
    defective = sum(1 for r in all_records if r["label"] == "defective")
    good = total - defective
    defect_rate = (defective / total * 100) if total > 0 else 0.0

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("All-Time Inspected", total)
    d2.metric("All-Time Passed", good)
    d3.metric("All-Time Defective", defective)
    d4.metric("All-Time Defect Rate", f"{defect_rate:.1f}%")

    st.dataframe(
        all_records,
        use_container_width=True,
        column_order=["inspected_at", "username", "filename", "label",
                      "confidence", "threshold_used"],
        hide_index=True,
    )

st.markdown(
    '<div class="footer-tag">BOLTGUARD AI · RESNET18 + GRAD-CAM · BUILT FOR PRECISION QC</div>',
    unsafe_allow_html=True,
)
