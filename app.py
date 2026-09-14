"""
app.py - Streamlit UI for the Screw Defect Detector.
Uses predict.py for all model logic (kept separate as required by the brief).

v2: adds a live webcam capture tab alongside batch upload, and a visual
refresh (custom CSS, styled result cards). No changes to predict.py.
"""

import streamlit as st
from PIL import Image, ImageDraw
import datetime

from predict import load_model, predict_image, get_gradcam_overlay, get_defect_bounding_box

st.set_page_config(
    page_title="BoltGuard AI — Screw Defect Detector",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — dark glassmorphism theme with animated accents
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .main {
            background: radial-gradient(circle at 20% 0%, #14213d 0%, #0a0e17 45%, #05070d 100%);
        }

        /* ---- animated hero header ---- */
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .app-header {
            position: relative;
            padding: 2.2rem 2.5rem;
            border-radius: 20px;
            background: linear-gradient(120deg, #ff6b6b, #f7b733, #4ecdc4, #556fb5, #ff6b6b);
            background-size: 300% 300%;
            animation: gradientShift 12s ease infinite;
            margin-bottom: 1.8rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.45);
            overflow: hidden;
        }
        .app-header::after {
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(5, 7, 13, 0.45);
        }
        .app-header * { position: relative; z-index: 1; }
        .app-header .eyebrow {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #ffe66d;
            background: rgba(0,0,0,0.35);
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.7rem;
        }
        .app-header h1 {
            color: #ffffff;
            margin: 0 0 0.4rem 0;
            font-size: 2.4rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .app-header p {
            color: #f1f5ff;
            margin-bottom: 0;
            font-size: 1.05rem;
            max-width: 640px;
        }

        /* ---- glass result cards ---- */
        .result-card {
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 0.9rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.10);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .result-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 28px rgba(0,0,0,0.35);
        }
        .result-card.good {
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.14), rgba(46, 204, 113, 0.04));
            border-left: 4px solid #2ecc71;
        }
        .result-card.defective {
            background: linear-gradient(135deg, rgba(255, 82, 82, 0.16), rgba(255, 82, 82, 0.05));
            border-left: 4px solid #ff5252;
        }
        .result-card h4 {
            margin: 0.5rem 0 0 0;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        /* ---- pulsing badge for defective ---- */
        @keyframes pulseGlow {
            0%   { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.55); }
            70%  { box-shadow: 0 0 0 10px rgba(255, 82, 82, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); }
        }
        .badge {
            display: inline-block;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }
        .badge.good {
            background: linear-gradient(135deg, #2ecc71, #1abc9c);
            color: #052e1c;
        }
        .badge.defective {
            background: linear-gradient(135deg, #ff5252, #ff1744);
            color: #2b0006;
            animation: pulseGlow 2s infinite;
        }

        /* ---- metrics ---- */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            transition: transform 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: scale(1.03);
        }
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace;
        }

        /* ---- tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.02rem;
            font-weight: 600;
            padding: 0.65rem 1.3rem;
            border-radius: 12px 12px 0 0;
            background: rgba(255,255,255,0.03);
        }
        .stTabs [aria-selected="true"] {
            background: rgba(255,255,255,0.09) !important;
            color: #ffe66d !important;
        }

        /* ---- images ---- */
        div[data-testid="stImage"] img {
            border-radius: 14px;
            transition: transform 0.25s ease;
        }
        div[data-testid="stImage"] img:hover {
            transform: scale(1.015);
        }

        /* ---- footer tagline ---- */
        .footer-tag {
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: rgba(255,255,255,0.35);
            margin-top: 2.5rem;
            letter-spacing: 0.04em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <span class="eyebrow">⚡ AI-Powered Visual Inspection</span>
        <h1>🛡️ BoltGuard AI</h1>
        <p>Screws lie. Pixels don't. Upload a batch or catch a live shot —
        BoltGuard flags the flaw and shows you exactly where it's looking.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🛡️ BoltGuard Console")
    st.write("**Architecture:** ResNet18 (transfer learning)")
    st.write("**Test Accuracy:** ~87%")
    st.write("**Recall on defects:** ~88%")
    st.write("**Dataset:** MVTec AD (screw category)")
    st.markdown("---")
    st.markdown("### 🎚️ Sensitivity Dial")
    threshold = st.slider(
        "Lower = catches more defects, but more false alarms",
        min_value=0.2, max_value=0.5, value=0.5, step=0.05
    )
    st.caption(f"Current threshold: {threshold}")


@st.cache_resource
def get_model():
    return load_model()


model = get_model()


def render_result(image, filename, threshold, key_prefix, precomputed=None):
    """
    Runs Grad-CAM (and prediction, unless already computed) on a single PIL
    image and renders a styled result card. Returns a dict summarizing the
    result (used for batch summaries and the downloadable report).

    precomputed: optional (label, confidence) tuple to avoid re-running
    inference when the caller already has it (e.g. for the batch summary).
    """
    if precomputed is not None:
        label, confidence = precomputed
    else:
        label, confidence = predict_image(model, image, threshold=threshold)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption=filename, use_container_width=True)

    with col2:
        if label == "defective":
            st.markdown(
                f"""
                <div class="result-card defective">
                    <span class="badge defective">⚠ Defective</span>
                    <h4>{confidence:.1f}% confidence</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.spinner("🔍 Tracing the flaw with Grad-CAM..."):
                overlay, grayscale_cam = get_gradcam_overlay(model, image)
                bbox = get_defect_bounding_box(grayscale_cam)

            overlay_img = Image.fromarray(overlay)
            if bbox:
                draw = ImageDraw.Draw(overlay_img)
                draw.rectangle(bbox, outline="yellow", width=3)

            st.image(
                overlay_img,
                caption="🎯 Grad-CAM heatmap — the model's focus, boxed",
                use_container_width=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card good">
                    <span class="badge good">✓ Good</span>
                    <h4>{confidence:.1f}% confidence</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return {"filename": filename, "label": label, "confidence": confidence}


def render_batch_summary(results):
    total = len(results)
    defective_count = sum(1 for r in results if r["label"] == "defective")
    good_count = total - defective_count
    defect_rate = (defective_count / total * 100) if total > 0 else 0

    st.markdown("## Batch Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Checked", total)
    col2.metric("Good", good_count)
    col3.metric("Defective", defective_count)
    col4.metric("Defect Rate", f"{defect_rate:.1f}%")
    st.markdown("---")

    return total, good_count, defective_count, defect_rate


def render_download_report(results, threshold, total, good_count, defective_count, defect_rate):
    report_lines = [
        "SCREW DEFECT DETECTION - QC REPORT",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Detection threshold used: {threshold}",
        "",
        f"Total screws checked: {total}",
        f"Good: {good_count}",
        f"Defective: {defective_count}",
        f"Defect rate: {defect_rate:.1f}%",
        "",
        "DETAILED RESULTS:",
    ]
    for r in results:
        report_lines.append(f"  {r['filename']}: {r['label'].upper()} ({r['confidence']:.1f}% confidence)")

    report_text = "\n".join(report_lines)

    st.download_button(
        label="📄 Download QC Report",
        data=report_text,
        file_name=f"qc_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )


# ---------------------------------------------------------------------------
# Tabs: Upload vs Live Webcam
# ---------------------------------------------------------------------------
tab_upload, tab_webcam = st.tabs(["📤 Batch Upload", "📷 Live Scan"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Drop screw images here — one or a hundred, BoltGuard doesn't blink.",
        type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        images = [Image.open(f) for f in uploaded_files]

        # Run predictions once up front so the summary metrics can be shown
        # before rendering each individual result card below.
        prelim_results = []
        for uploaded_file, image in zip(uploaded_files, images):
            label, confidence = predict_image(model, image, threshold=threshold)
            prelim_results.append({"filename": uploaded_file.name, "label": label, "confidence": confidence})

        total, good_count, defective_count, defect_rate = render_batch_summary(prelim_results)

        st.markdown("## Results")
        results = []
        for uploaded_file, image, prelim in zip(uploaded_files, images, prelim_results):
            result = render_result(
                image, uploaded_file.name, threshold,
                key_prefix=uploaded_file.name,
                precomputed=(prelim["label"], prelim["confidence"]),
            )
            results.append(result)
            st.markdown("---")

        render_download_report(results, threshold, total, good_count, defective_count, defect_rate)
    else:
        st.info("💡 Drop some screw images above and BoltGuard gets to work instantly.")

with tab_webcam:
    st.write("Line up the shot. One click, one verdict.")
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image)
        st.markdown("## Result")
        result = render_result(image, "Live Capture", threshold, key_prefix="webcam")

        results = [result]
        total, good_count, defective_count, defect_rate = render_batch_summary(results)
        render_download_report(results, threshold, total, good_count, defective_count, defect_rate)
    else:
        st.info("📸 Fire up your camera above and catch a screw in the act.")

st.markdown(
    '<div class="footer-tag">BOLTGUARD AI · RESNET18 + GRAD-CAM · BUILT FOR PRECISION QC</div>',
    unsafe_allow_html=True,
)
