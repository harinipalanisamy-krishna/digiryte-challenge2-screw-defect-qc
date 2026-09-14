"""
app.py - Streamlit UI for the Screw Defect Detector.
Uses predict.py for all model logic (kept separate as required by the brief).

v3 changes (all data shown is genuinely computed from real model inference —
nothing here is simulated or hardcoded):
  - Session-wide analytics HUD: Inspected / Passed / Defective / Defect Rate
    accumulate across every image processed this session (upload + camera),
    not just the current batch. Includes a Reset Metrics button.
  - Filter tabs (All / Passed / Defective) over the session's real results.
  - Results shown as a "camera feed" style grid, inspired by common
    industrial visual-inspection dashboard conventions (status badges,
    KPI row, exportable defect log).
  - CSV export of the full session log, alongside the original text report.
  - Grad-CAM overlay now carries an explicit "approximate region" disclaimer.

v3.1 (styling-only update):
  - Swapped the rainbow glassmorphism theme for a dark charcoal industrial
    palette (red/green/amber status colors), and added a simple line-art
    screw icon to the header. No logic, features, or page structure changed.
"""

import streamlit as st
from PIL import Image, ImageDraw
import datetime
import hashlib
import csv
import io

from predict import load_model, predict_image, get_gradcam_overlay, get_defect_bounding_box

st.set_page_config(
    page_title="BoltGuard AI — Screw Defect Detector",
    layout="wide",
    page_icon="🔩",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Simple line-art screw icon (self-drawn SVG, no external image dependency)
# ---------------------------------------------------------------------------
SCREW_ICON_SVG = """
<svg width="56" height="56" viewBox="0 0 56 56" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#f5a623" stroke-width="2.2" stroke-linecap="round">
    <rect x="14" y="4" width="28" height="10" rx="2" fill="#2a2a2e" />
    <line x1="19" y1="4" x2="19" y2="14" stroke="#3a3a3f" stroke-width="1.4" />
    <line x1="25" y1="4" x2="25" y2="14" stroke="#3a3a3f" stroke-width="1.4" />
    <line x1="31" y1="4" x2="31" y2="14" stroke="#3a3a3f" stroke-width="1.4" />
    <line x1="37" y1="4" x2="37" y2="14" stroke="#3a3a3f" stroke-width="1.4" />
    <line x1="20" y1="9" x2="36" y2="9" stroke="#f5a623" stroke-width="2" />
    <line x1="28" y1="14" x2="28" y2="48" stroke="#f5a623" stroke-width="3" />
    <path d="M 28 16 L 34 20 L 28 24 L 34 28 L 28 32 L 34 36 L 28 40 L 34 44 L 28 48"
          stroke="#c77c0e" stroke-width="2" />
  </g>
</svg>
"""

# ---------------------------------------------------------------------------
# Styling — dark charcoal industrial theme, red/green/amber status colors
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .main {
            background: radial-gradient(circle at 20% 0%, #232326 0%, #17171a 45%, #0e0e10 100%);
        }

        .app-header {
            position: relative;
            display: flex;
            align-items: center;
            gap: 1.4rem;
            padding: 1.8rem 2.4rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #232326, #18181b);
            border-left: 5px solid #f5a623;
            margin-bottom: 1.8rem;
            box-shadow: 0 8px 26px rgba(0,0,0,0.5);
        }
        .app-header .eyebrow {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #f5a623;
            background: rgba(245, 166, 35, 0.12);
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.6rem;
        }
        .app-header h1 {
            color: #f4f4f5;
            margin: 0 0 0.35rem 0;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .app-header p {
            color: #b8b8bd;
            margin-bottom: 0;
            font-size: 1.0rem;
            max-width: 640px;
        }
        .app-header .icon-box {
            flex-shrink: 0;
            width: 64px;
            height: 64px;
            border-radius: 12px;
            background: #1c1c1f;
            border: 1px solid rgba(245, 166, 35, 0.25);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        div[data-testid="stMetric"] {
            background: #1c1c1f;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace;
            color: #f4f4f5;
        }

        .feed-card {
            border-radius: 14px;
            padding: 0.9rem;
            margin-bottom: 1rem;
            background: #1c1c1f;
            border: 1px solid rgba(255,255,255,0.06);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .feed-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        }
        .feed-card.good {
            border-left: 4px solid #2ecc71;
        }
        .feed-card.defective {
            border-left: 4px solid #e74c3c;
        }
        .feed-card .fname {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            opacity: 0.55;
            color: #d0d0d5;
            margin-top: 0.4rem;
            word-break: break-all;
        }

        @keyframes pulseGlow {
            0%   { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.5); }
            70%  { box-shadow: 0 0 0 8px rgba(231, 76, 60, 0); }
            100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
        }
        .badge {
            display: inline-block;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.72rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }
        .badge.good {
            background: rgba(46, 204, 113, 0.15);
            color: #2ecc71;
            border: 1px solid rgba(46, 204, 113, 0.4);
        }
        .badge.defective {
            background: rgba(231, 76, 60, 0.15);
            color: #ff6b5f;
            border: 1px solid rgba(231, 76, 60, 0.4);
            animation: pulseGlow 2s infinite;
        }

        .disclaimer {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            opacity: 0.55;
            color: #d0d0d5;
            margin-top: 0.3rem;
        }

        .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.0rem;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            border-radius: 10px 10px 0 0;
            background: #1c1c1f;
            color: #b8b8bd;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(245, 166, 35, 0.12) !important;
            color: #f5a623 !important;
        }

        div[data-testid="stImage"] img {
            border-radius: 10px;
        }

        .footer-tag {
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: rgba(255,255,255,0.28);
            margin-top: 2.5rem;
            letter-spacing: 0.04em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="app-header">
        <div class="icon-box">{SCREW_ICON_SVG}</div>
        <div>
            <span class="eyebrow">⚡ AI-Powered Visual Inspection</span>
            <h1>BoltGuard AI</h1>
            <p>Screws lie. Pixels don't. Upload a batch or catch a live shot —
            BoltGuard flags the flaw and shows you exactly where it's looking.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🔩 BoltGuard Console")
    st.write("**Architecture:** ResNet18 (transfer learning)")
    st.write("**Test Accuracy:** ~87%")
    st.write("**Recall on defects:** ~88%")
    st.write("**Dataset:** MVTec AD (screw category)")
    st.markdown("---")
    st.markdown("### 🎚️ Strictness Level")
    threshold = st.slider(
        "Lower = catches more defects, but more false alarms",
        min_value=0.2, max_value=0.5, value=0.5, step=0.05
    )
    st.caption(f"Current threshold: {threshold}")


@st.cache_resource
def get_model():
    return load_model()


model = get_model()

# ---------------------------------------------------------------------------
# Session-wide state — every number shown anywhere in this app is derived
# from this list, which only grows when a NEW image is actually processed.
# ---------------------------------------------------------------------------
if "session_log" not in st.session_state:
    st.session_state.session_log = []          # list of result dicts, in order
if "seen_hashes" not in st.session_state:
    st.session_state.seen_hashes = set()        # dedupe: avoid recounting the
                                                 # same image on an unrelated rerun


def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def process_and_log(image_bytes, filename, threshold, source):
    """
    Runs real inference (+ Grad-CAM for defective results) on one image and
    appends the result to the session-wide log, unless this exact image was
    already processed earlier in the session (prevents double-counting when
    Streamlit reruns the script for an unrelated widget interaction).
    Returns the result dict (existing or newly created).
    """
    img_hash = _hash_bytes(image_bytes)

    for existing in st.session_state.session_log:
        if existing["hash"] == img_hash:
            return existing

    image = Image.open(io.BytesIO(image_bytes))
    label, confidence = predict_image(model, image, threshold=threshold)

    overlay_img = None
    if label == "defective":
        overlay, grayscale_cam = get_gradcam_overlay(model, image)
        bbox = get_defect_bounding_box(grayscale_cam)
        overlay_img = Image.fromarray(overlay)
        if bbox:
            draw = ImageDraw.Draw(overlay_img)
            draw.rectangle(bbox, outline="yellow", width=3)

    thumb = image.convert("RGB").copy()
    thumb.thumbnail((260, 260))

    result = {
        "hash": img_hash,
        "filename": filename,
        "label": label,
        "confidence": confidence,
        "source": source,
        "timestamp": datetime.datetime.now(),
        "thumb": thumb,
        "overlay": overlay_img,
    }
    st.session_state.session_log.append(result)
    st.session_state.seen_hashes.add(img_hash)
    return result


def render_hud():
    log = st.session_state.session_log
    total = len(log)
    defective_count = sum(1 for r in log if r["label"] == "defective")
    good_count = total - defective_count
    defect_rate = (defective_count / total * 100) if total > 0 else 0.0

    st.markdown("## 📊 Session Inspection HUD")
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    c1.metric("Inspected", total)
    c2.metric("Passed", good_count)
    c3.metric("Defective", defective_count)
    c4.metric("Defect Rate", f"{defect_rate:.1f}%")
    with c5:
        st.write("")
        if st.button("🔄 Reset Metrics", use_container_width=True):
            st.session_state.session_log = []
            st.session_state.seen_hashes = set()
            st.rerun()
    st.markdown("---")
    return total, good_count, defective_count, defect_rate


def render_feed_grid(results, columns_per_row=3):
    if not results:
        st.info("No results in this view yet.")
        return

    rows = [results[i:i + columns_per_row] for i in range(0, len(results), columns_per_row)]
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
                    <div style="margin-top:0.4rem; font-family:'JetBrains Mono',monospace; color:#d0d0d5;">
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
                            '<div class="disclaimer">📍 Approximate region of interest — '
                            'not a precise defect boundary.</div>',
                            unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)


def render_exports():
    log = st.session_state.session_log
    if not log:
        return

    total = len(log)
    defective_count = sum(1 for r in log if r["label"] == "defective")
    good_count = total - defective_count
    defect_rate = (defective_count / total * 100) if total > 0 else 0.0

    report_lines = [
        "SCREW DEFECT DETECTION - QC SESSION REPORT",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Detection threshold used: {threshold}",
        "",
        f"Total screws inspected this session: {total}",
        f"Good: {good_count}",
        f"Defective: {defective_count}",
        f"Defect rate: {defect_rate:.1f}%",
        "",
        "DETAILED RESULTS:",
    ]
    for r in log:
        report_lines.append(
            f"  [{r['timestamp'].strftime('%H:%M:%S')}] {r['filename']} ({r['source']}): "
            f"{r['label'].upper()} ({r['confidence']:.1f}% confidence)"
        )
    report_text = "\n".join(report_lines)

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["timestamp", "filename", "source", "label", "confidence_percent"])
    for r in log:
        writer.writerow([
            r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            r["filename"],
            r["source"],
            r["label"],
            f"{r['confidence']:.1f}",
        ])

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download Session Report (.txt)",
            data=report_text,
            file_name=f"qc_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="📑 Download Defect Log (.csv)",
            data=csv_buffer.getvalue(),
            file_name=f"defect_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
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
        for uploaded_file in uploaded_files:
            process_and_log(uploaded_file.getvalue(), uploaded_file.name, threshold, source="Upload")
    else:
        st.info("💡 Drop some screw images above and BoltGuard gets to work instantly.")

with tab_webcam:
    st.write("Line up the shot. One click, one verdict.")
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        process_and_log(
            camera_image.getvalue(),
            f"live_capture_{datetime.datetime.now().strftime('%H%M%S')}.jpg",
            threshold,
            source="Live Camera",
        )
    else:
        st.info("📸 Fire up your camera above and catch a screw in the act.")

# ---------------------------------------------------------------------------
# Shared HUD + filterable results grid, driven entirely by session_log
# ---------------------------------------------------------------------------
if st.session_state.session_log:
    render_hud()

    st.markdown("## 🔍 Results")
    filter_choice = st.radio(
        "Filter",
        options=["All Screws", "Passed", "Defective"],
        horizontal=True,
        label_visibility="collapsed",
    )

    log_newest_first = list(reversed(st.session_state.session_log))
    if filter_choice == "Passed":
        filtered = [r for r in log_newest_first if r["label"] == "good"]
    elif filter_choice == "Defective":
        filtered = [r for r in log_newest_first if r["label"] == "defective"]
    else:
        filtered = log_newest_first

    render_feed_grid(filtered)

    st.markdown("---")
    render_exports()

st.markdown(
    '<div class="footer-tag">BOLTGUARD AI · RESNET18 + GRAD-CAM · BUILT FOR PRECISION QC</div>',
    unsafe_allow_html=True,
)
