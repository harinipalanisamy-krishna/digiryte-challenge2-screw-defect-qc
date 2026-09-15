"""
views/inspect.py - the "Inspect" page: a long-form, one-image-at-a-time
view. Each screw gets its own full-width card: the photo, then the
verdict, then (for defects) the Grad-CAM heatmap and disclaimer below it -
stacked, not side-by-side, so each result is easy to read top to bottom.
"""

import datetime

import streamlit as st

from common import (
    inject_global_styles,
    render_page_header,
    render_sidebar,
    init_session_state,
    process_and_log,
)

inject_global_styles()
init_session_state()
render_sidebar()

render_page_header(
    tag="⚡ AI-Powered Visual Inspection",
    title="🔩 BoltGuard AI — Inspect",
    subtitle="Screws lie. Pixels don't. Upload one or more images, or catch a live shot, "
             "and BoltGuard walks through each one in detail below.",
)

tab_upload, tab_webcam = st.tabs(["📤 Upload", "📷 Live Scan"])

new_results = []

with tab_upload:
    uploaded_files = st.file_uploader(
        "Drop screw images here — one or a hundred, BoltGuard doesn't blink.",
        type=["png", "jpg", "jpeg"], accept_multiple_files=True,
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            new_results.append(
                process_and_log(
                    uploaded_file.getvalue(), uploaded_file.name,
                    st.session_state.threshold, source="Upload",
                )
            )
    else:
        st.info("💡 Drop some screw images above and BoltGuard gets to work instantly.")

with tab_webcam:
    st.write("Line up the shot. One click, one verdict.")
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        new_results.append(
            process_and_log(
                camera_image.getvalue(),
                f"live_capture_{datetime.datetime.now().strftime('%H%M%S')}.jpg",
                st.session_state.threshold, source="Live Camera",
            )
        )
    else:
        st.info("📸 Fire up your camera above and catch a screw in the act.")

if new_results:
    st.markdown("## Inspection Detail")
    for r in new_results:
        css_class = "defective" if r["label"] == "defective" else "good"
        badge_label = "⚠ Defective" if r["label"] == "defective" else "✓ Good"

        st.markdown(f'<div class="inspect-card {css_class}">', unsafe_allow_html=True)
        st.image(r["thumb"], use_container_width=True, caption=r["filename"])
        st.markdown(
            f"""
            <span class="badge {css_class}">{badge_label}</span>
            <span class="conf-text-inline {css_class}">{r['confidence']:.1f}% confidence · {r['source']}</span>
            """,
            unsafe_allow_html=True,
        )

        if r["label"] == "defective" and r["overlay"] is not None:
            st.markdown(
                '<div class="disclaimer">⚠ This heatmap shows where the model was looking, '
                'not a confirmed defect boundary. The highlighted region can land near — but '
                'not exactly on — the visible flaw. See the README for why.</div>',
                unsafe_allow_html=True,
            )
            st.image(r["overlay"], use_container_width=True, caption="Grad-CAM attention map")

        st.markdown("</div>", unsafe_allow_html=True)

    st.info("📊 Head to the **Dashboard** page (left sidebar) for cumulative session stats, "
            "filters, and exports.")

st.markdown(
    '<div class="footer-tag">BOLTGUARD AI · RESNET18 + GRAD-CAM · BUILT FOR PRECISION QC</div>',
    unsafe_allow_html=True,
)
