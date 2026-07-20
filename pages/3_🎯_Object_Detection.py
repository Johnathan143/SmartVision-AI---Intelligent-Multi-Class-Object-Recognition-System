import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

import config
import utils

from utils import footer
st.set_page_config(
    page_title="Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YOLOv8 Object Detection")

st.markdown(
"""
Detect multiple objects in an uploaded image using the trained YOLOv8 model.
"""
)

st.markdown("---")

model = utils.load_yolo()
st.sidebar.header("🎯 Detection Settings")

mode = st.sidebar.radio(
    "Detection Mode",
    ["Auto", "Manual"],
    index=0
)

if mode == "Auto":
    confidence = 0.25
    iou = 0.45

    st.sidebar.success("Using recommended YOLOv8 settings")

    st.sidebar.write(f"Confidence: **{confidence:.2f}**")
    st.sidebar.write(f"IoU: **{iou:.2f}**")

else:
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05
    )

    iou = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05
    )

uploaded_file = st.file_uploader(
    "Upload Image",
    type=config.SUPPORTED_FORMATS
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    start = time.time()

    results = model.predict(
        image_np,
        conf=confidence,
        iou=iou,
        verbose=False
    )
    print(results[0].boxes)

    inference = time.time() - start

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    with col2:
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.subheader("Detection Result")
        st.image(annotated, width="stretch")

    detections = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "Object": model.names[cls],
            "Confidence": round(conf * 100, 2)
        })

    df = pd.DataFrame(detections)

    st.markdown("---")
    st.subheader("Detected Objects")

    if len(df):
        st.dataframe(df, width="stretch")
    else:
        st.warning("No objects detected.")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Objects Detected", len(df))

    with c2:
        st.metric("Inference Time", f"{inference:.3f} sec")

    if len(df):
        st.subheader("Object Distribution")
        st.bar_chart(df["Object"].value_counts())

    success, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    )

    st.download_button(
        "📥 Download Detection",
        buffer.tobytes(),
        file_name="detection.jpg",
        mime="image/jpeg"
    )

footer()