"""
views/dashboard.py - the "Dashboard" page: cumulative session analytics,
filterable results grid, and export buttons. All numbers here are derived
live from st.session_state.session_log - nothing is simulated.
"""

import streamlit as st

from common import (
    inject_global_styles,
    render_page_header,
    render_sidebar,
    init_session_state,
    session_summary,
    build_report_text,
    build_report_csv,
)

inject_global_styles()
init_session_state()
render_sidebar()

render_page_header(
    tag="📊 Live Session Analytics",
    title="Dashboard",
    subtitle="Cumulative results across everything inspected this session — real counts, "
             "no placeholders.",
)

log = st.session_state.session_log

if not log:
    st.info("No inspections yet this session. Go to the **Inspect** page (left sidebar) "
            "to upload or scan a screw — results will appear here automatically.")
else:
    total, good_count, defective_count, defect_rate = session_summary()

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    c1.metric("Inspected", total)
    c2.metric("Passed", good_count)
    c3.metric("Defective", defective_count)
    c4.metric("Defect Rate", f"{defect_rate:.1f}%")
    with c5:
        st.write("")
        if st.button("🔄 Reset Metrics", use_container_width=True):
            st.session_state.session_log = []
            st.rerun()

    st.markdown("---")
    st.markdown("## Results")

    filter_choice = st.radio(
        "Filter",
        options=["All Screws", "Passed", "Defective"],
        horizontal=True,
        label_visibility="collapsed",
    )

    log_newest_first = list(reversed(log))
    if filter_choice == "Passed":
        filtered = [r for r in log_newest_first if r["label"] == "good"]
    elif filter_choice == "Defective":
        filtered = [r for r in log_newest_first if r["label"] == "defective"]
    else:
        filtered = log_newest_first

    if not filtered:
        st.info("No results in this view yet.")
    else:
        columns_per_row = 3
        rows = [filtered[i:i + columns_per_row] for i in range(0, len(filtered), columns_per_row)]
        for row in rows:
            cols = st.columns(columns_per_row)
            for col, r in zip(cols, row):
                with col:
                    css_class = "defective" if r["label"] == "defective" else "good"
                    badge_label = "⚠ Defective" if r["label"] == "defective" else "✓ Good"
                    st.markdown(f'<div class="feed-card {css_class}">', unsafe_allow_html=True)
                    st.image(r["thumb"], use_container_width=True)
                    st.markdown(
                        f"""
                        <span class="badge {css_class}">{badge_label}</span>
                        <div style="margin-top:0.4rem; font-family:'JetBrains Mono',monospace;">
                            {r['confidence']:.1f}% confidence
                        </div>
                        <div class="fname">{r['filename']} · {r['source']}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if r["label"] == "defective" and r["overlay"] is not None:
                        with st.expander("🎯 View Grad-CAM"):
                            st.image(r["overlay"], use_container_width=True)
                            st.markdown(
                                '<div class="disclaimer">📍 Approximate attention region — '
                                'not a precise defect boundary.</div>',
                                unsafe_allow_html=True,
                            )
                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📄 Download Session Report (.txt)",
            data=build_report_text(),
            file_name="qc_session_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "📑 Download Defect Log (.csv)",
            data=build_report_csv(),
            file_name="qc_defect_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown(
    '<div class="footer-tag">BOLTGUARD AI · RESNET18 + GRAD-CAM · BUILT FOR PRECISION QC</div>',
    unsafe_allow_html=True,
)
