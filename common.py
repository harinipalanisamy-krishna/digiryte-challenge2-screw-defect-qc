"""
common.py - Shared styling, model access, and session-state logic used by
both pages of the BoltGuard AI app (views/inspect.py, views/dashboard.py).

Keeping this here means predict.py's model logic stays completely untouched
and un-duplicated, and both pages render with an identical dark theme.
"""

import datetime
import hashlib
import io
import csv

import streamlit as st
from PIL import Image, ImageDraw

from predict import load_model, predict_image, get_gradcam_overlay, get_defect_bounding_box

# ---------------------------------------------------------------------------
# Theme: industry-standard dark dashboard.
# Main canvas: near-black. Sidebar: a distinctly different dark slate, so the
# two regions read as separate panels the way most SaaS dashboards do.
# ---------------------------------------------------------------------------
BG_MAIN = "#0a0a0c"
BG_SIDEBAR = "#12151c"
BG_CARD = "#15161a"
ACCENT = "#22d3ee"       # cyan accent for headings / active nav / focus states
GOOD = "#2ecc71"
BAD = "#ef4444"
WARN = "#f5a623"
TEXT_PRIMARY = "#f4f4f6"
TEXT_SECONDARY = "#9aa0ab"


def inject_global_styles():
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Inter', sans-serif;
                color: {TEXT_PRIMARY};
            }}

            .stApp {{
                background: {BG_MAIN};
            }}

            section[data-testid="stSidebar"] {{
                background: {BG_SIDEBAR};
                border-right: 1px solid rgba(255,255,255,0.06);
            }}
            section[data-testid="stSidebar"] * {{
                color: {TEXT_PRIMARY};
            }}

            /* Page header banner used at the top of every page */
            .page-header {{
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1.4rem 1.8rem;
                border-radius: 14px;
                background: {BG_CARD};
                border: 1px solid rgba(255,255,255,0.07);
                border-left: 4px solid {ACCENT};
                margin-bottom: 1.6rem;
            }}
            .page-header h1 {{
                margin: 0;
                font-size: 1.8rem;
                font-weight: 700;
                color: {TEXT_PRIMARY};
                letter-spacing: -0.01em;
            }}
            .page-header p {{
                margin: 0.15rem 0 0 0;
                color: {TEXT_SECONDARY};
                font-size: 0.95rem;
            }}
            .page-header .tag {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.68rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: {ACCENT};
                background: rgba(34, 211, 238, 0.10);
                border: 1px solid rgba(34, 211, 238, 0.25);
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                display: inline-block;
                margin-bottom: 0.35rem;
            }}

            /* Long-form single-image inspection card (Inspect page) */
            .inspect-card {{
                background: {BG_CARD};
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 16px;
                padding: 1.4rem;
                margin-bottom: 1.6rem;
            }}
            .inspect-card.good {{ border-left: 4px solid {GOOD}; }}
            .inspect-card.defective {{ border-left: 4px solid {BAD}; }}

            /* Compact grid card (Dashboard page) */
            .feed-card {{
                background: {BG_CARD};
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 14px;
                padding: 0.9rem;
                margin-bottom: 1rem;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }}
            .feed-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.45);
            }}
            .feed-card.good {{ border-left: 4px solid {GOOD}; }}
            .feed-card.defective {{ border-left: 4px solid {BAD}; }}
            .feed-card .fname {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                color: {TEXT_SECONDARY};
                margin-top: 0.4rem;
                word-break: break-all;
            }}

            @keyframes pulseGlow {{
                0%   {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }}
                70%  {{ box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
            }}
            .badge {{
                display: inline-block;
                padding: 0.3rem 0.8rem;
                border-radius: 999px;
                font-weight: 700;
                font-size: 0.75rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                font-family: 'JetBrains Mono', monospace;
            }}
            .badge.good {{
                background: rgba(46, 204, 113, 0.14);
                color: {GOOD};
                border: 1px solid rgba(46, 204, 113, 0.4);
            }}
            .badge.defective {{
                background: rgba(239, 68, 68, 0.14);
                color: {BAD};
                border: 1px solid rgba(239, 68, 68, 0.4);
                animation: pulseGlow 2s infinite;
            }}

            .disclaimer {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.72rem;
                color: {WARN};
                background: rgba(245, 166, 35, 0.08);
                border: 1px solid rgba(245, 166, 35, 0.3);
                border-radius: 8px;
                padding: 0.55rem 0.8rem;
                margin: 0.6rem 0;
                line-height: 1.45;
            }}

            div[data-testid="stMetric"] {{
                background: {BG_CARD};
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 12px;
                padding: 0.9rem 1rem;
            }}
            div[data-testid="stMetricLabel"] {{ color: {TEXT_SECONDARY}; }}
            div[data-testid="stMetricValue"] {{
                font-family: 'JetBrains Mono', monospace;
                color: {TEXT_PRIMARY};
            }}

            .stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; }}
            .stTabs [data-baseweb="tab"] {{
                font-size: 0.95rem;
                font-weight: 600;
                padding: 0.55rem 1.1rem;
                border-radius: 10px 10px 0 0;
                background: {BG_CARD};
                color: {TEXT_SECONDARY};
            }}
            .stTabs [aria-selected="true"] {{
                background: rgba(34, 211, 238, 0.12) !important;
                color: {ACCENT} !important;
            }}

            div[data-testid="stImage"] img {{
                border-radius: 10px;
            }}

            .footer-tag {{
                text-align: center;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                color: rgba(255,255,255,0.25);
                margin-top: 2.5rem;
                letter-spacing: 0.04em;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(tag, title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <span class="tag">{tag}</span>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔩 BoltGuard Console")
        st.write("**Architecture:** ResNet18 (transfer learning)")
        st.write("**Test Accuracy:** ~87%")
        st.write("**Recall on defects:** ~88%")
        st.write("**Dataset:** MVTec AD (screw category)")
        st.markdown("---")
        st.markdown("### 🎚️ Strictness Level")
        st.session_state.threshold = st.slider(
            "Lower = catches more defects, but more false alarms",
            min_value=0.2, max_value=0.5,
            value=st.session_state.get("threshold", 0.5),
            step=0.05,
        )
        st.caption(f"Current threshold: {st.session_state.threshold}")


@st.cache_resource
def get_model():
    return load_model()


def init_session_state():
    if "session_log" not in st.session_state:
        st.session_state.session_log = []      # list of result dicts, in order
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.5


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
    model = get_model()
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
    thumb.thumbnail((320, 320))

    result = {
        "hash": img_hash,
        "filename": filename,
        "label": label,
        "confidence": confidence,
        "source": source,
        "threshold_used": threshold,
        "timestamp": datetime.datetime.now(),
        "thumb": thumb,
        "overlay": overlay_img,
    }
    st.session_state.session_log.append(result)
    return result


def session_summary():
    log = st.session_state.session_log
    total = len(log)
    defective_count = sum(1 for r in log if r["label"] == "defective")
    good_count = total - defective_count
    defect_rate = (defective_count / total * 100) if total > 0 else 0.0
    return total, good_count, defective_count, defect_rate


def build_report_text():
    log = st.session_state.session_log
    total, good_count, defective_count, defect_rate = session_summary()

    lines = [
        "SCREW DEFECT DETECTION - QC SESSION REPORT",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total screws inspected this session: {total}",
        f"Good: {good_count}",
        f"Defective: {defective_count}",
        f"Defect rate: {defect_rate:.1f}%",
        "",
        "DETAILED RESULTS:",
    ]
    for r in log:
        lines.append(
            f"  [{r['timestamp'].strftime('%H:%M:%S')}] {r['filename']} "
            f"({r['source']}, threshold {r['threshold_used']}): "
            f"{r['label'].upper()} ({r['confidence']:.1f}% confidence)"
        )
    return "\n".join(lines)


def build_report_csv():
    log = st.session_state.session_log
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["timestamp", "filename", "source", "threshold_used", "label", "confidence_percent"])
    for r in log:
        writer.writerow([
            r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            r["filename"],
            r["source"],
            r["threshold_used"],
            r["label"],
            f"{r['confidence']:.1f}",
        ])
    return buffer.getvalue()
