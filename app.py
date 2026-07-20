import streamlit as st

import config

st.set_page_config(

    page_title=config.PROJECT_NAME,

    page_icon="🤖",

    layout="wide",

    initial_sidebar_state="expanded"
)
st.title("🤖 SmartVision AI")
st.markdown(
"""
### Intelligent Multi-Class Image Classification and Object Detection

A Deep Learning application built using

- TensorFlow
- Keras
- YOLOv8
- OpenCV
- Streamlit
"""
)
col1,col2,col3,col4 = st.columns(4)

with col1:

    st.metric(
        "Classification Models",
        "4"
    )

with col2:

    st.metric(
        "Detection Models",
        "1"
    )

with col3:

    st.metric(
        "Framework",
        "YOLOv8"
    )

with col4:

    st.metric(
        "Interface",
        "Streamlit"
    )

    st.markdown("---")

st.subheader("Workflow")

st.markdown(
"""
Image Upload

⬇

Classification or Detection

⬇

Deep Learning Model

⬇

Prediction

⬇

Visualization
"""
)

st.sidebar.title("Navigation")

st.sidebar.success(
    "Select a page from the sidebar."
)

st.sidebar.info(
"""
Available Modules

🏠 Home

🖼 Image Classification

🎯 Object Detection

📊 Model Comparison

📈 Performance Metrics

ℹ About
"""
)

st.markdown("---")

st.caption("SmartVision AI Version 1.0")