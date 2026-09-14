"""
app.py - Streamlit UI for the Screw Defect Detector.
Uses predict.py for all model logic (kept separate as required by the brief).
"""

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io
import datetime

from predict import load_model, predict_image, get_gradcam_overlay, get_defect_bounding_box

st.set_page_config(page_title="Screw Defect Detector", layout="wide")

st.title("Screw Defect Detection — QC Assistant")
st.write(
    "Upload one or more screw images to check for defects. "
    "Built with a ResNet18 model fine-tuned on the MVTec AD dataset."
)

with st.sidebar:
    st.header("Model Info")
    st.write("**Architecture:** ResNet18 (transfer learning)")
    st.write("**Test Accuracy:** ~87%")
    st.write("**Recall on defects:** ~88%")
    st.write("**Dataset:** MVTec AD (screw category)")
    st.markdown("---")
    st.header("Detection Sensitivity")
    threshold = st.slider(
        "Lower = catches more defects, but more false alarms",
        min_value=0.2, max_value=0.5, value=0.5, step=0.05
    )
    st.caption(f"Current threshold: {threshold}")

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

uploaded_files = st.file_uploader(
    "Upload screw image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        label, confidence = predict_image(model, image, threshold=threshold)
        results.append({
            "filename": uploaded_file.name,
            "label": label,
            "confidence": confidence
        })

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
    st.markdown("## Results")

    for uploaded_file, result in zip(uploaded_files, results):
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with col2:
            if result["label"] == "defective":
                st.error(f"⚠️ DEFECTIVE — {result['confidence']:.1f}% confidence")

                overlay, grayscale_cam = get_gradcam_overlay(model, image)
                bbox = get_defect_bounding_box(grayscale_cam)

                overlay_img = Image.fromarray(overlay)
                if bbox:
                    draw = ImageDraw.Draw(overlay_img)
                    draw.rectangle(bbox, outline="yellow", width=3)

                st.image(overlay_img, caption="Grad-CAM: where the model looked (approx. defect region boxed)", use_container_width=True)
            else:
                st.success(f"✅ GOOD — {result['confidence']:.1f}% confidence")

        st.markdown("---")

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
        label="Download QC Report",
        data=report_text,
        file_name=f"qc_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
else:
    st.info("Upload one or more screw images above to get started.")
