import streamlit as st
import config
from utils import footer

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------------
# HEADER
# --------------------------------------------------------

st.title("🤖 SmartVision AI")

st.markdown(
"""
### Intelligent Multi-Class Image Classification & Object Detection

A deep learning application capable of classifying images and detecting multiple
objects using state-of-the-art Computer Vision models.

---
"""
)

col1, col2 = st.columns([2,1])

with col1:

    st.markdown(
    """
    ## Welcome 👋

    SmartVision AI combines **Image Classification** and **Object Detection**
    into one intelligent application.

    The project demonstrates how multiple deep learning architectures can
    solve different computer vision tasks while allowing users to compare
    their performance.

    **Included Models**

    - VGG16
    - ResNet50
    - MobileNetV2
    - EfficientNetB0
    - YOLOv8
    """
    )

with col2:

    st.info(
    """
    📌 **Project Information**

    Version : 1.0

    Framework : Streamlit

    Backend : TensorFlow + YOLOv8

    Deployment Ready
    """
    )

st.markdown("---")

st.subheader("📊 Project Statistics")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "Classification Models",
        "4"
    )

with c2:
    st.metric(
        "Detection Models",
        "1"
    )

with c3:
    st.metric(
        "Deep Learning Frameworks",
        "2"
    )

with c4:
    st.metric(
        "Application",
        "Streamlit"
    )

st.markdown("---")

st.subheader("🛠 Technology Stack")

col1,col2,col3 = st.columns(3)

with col1:

    st.success("""
### Deep Learning

- TensorFlow
- Keras
- YOLOv8
""")

with col2:

    st.info("""
### Computer Vision

- OpenCV
- Pillow
- NumPy
""")

with col3:

    st.warning("""
### Web Application

- Streamlit
- Pandas
- Matplotlib
""")
    
st.markdown("---")

st.subheader("⚙️ Project Workflow")

st.code("""

Image Upload
      │
      ▼
Choose AI Model
      │
      ▼
Image Preprocessing
      │
      ▼
Deep Learning Inference
      │
      ▼
Prediction
      │
      ▼
Visualization

""")

st.markdown("---")

st.subheader("🧩 Available Modules")

col1,col2 = st.columns(2)

with col1:

    st.markdown("""
### 🖼 Image Classification

✔ Upload an Image

✔ Choose Classification Model

✔ Predict Image Class

✔ Confidence Score

✔ Inference Time
""")

with col2:

    st.markdown("""
### 🎯 Object Detection

✔ Upload Image

✔ Detect Multiple Objects

✔ Bounding Boxes

✔ Confidence Values

✔ Download Result
""")
    
st.markdown("---")

st.subheader("🧠 Models Used")

models = [
    "VGG16",
    "ResNet50",
    "MobileNetV2",
    "EfficientNetB0",
    "YOLOv8"
]

for model in models:

    st.checkbox(model, value=True, disabled=True)

st.markdown("---")

st.subheader("✨ Key Features")

features = [
    "Image Classification",
    "Object Detection",
    "Confidence Score",
    "Inference Time",
    "Model Comparison",
    "Performance Metrics",
    "Interactive Dashboard",
    "Download Predictions"
]

for feature in features:

    st.write("✅", feature)

st.markdown("---")

st.subheader("🎯 Project Objectives")

st.markdown(
"""
- Build a multi-model computer vision application.

- Compare transfer learning models.

- Demonstrate object detection using YOLOv8.

- Measure performance using standard evaluation metrics.

- Deploy the application using Streamlit.
"""
)

st.markdown("---")

st.success(
"""
### Next Steps

➡ Image Classification

➡ Object Detection

➡ Model Comparison

➡ Performance Metrics

➡ About Project

Use the left sidebar to navigate through the application.
"""
)

footer()