import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

import config
import utils

from utils import footer

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

#Page Config
st.set_page_config(

    page_title="Image Classification",

    page_icon="🖼️",

    layout="wide"
)

#Title
st.title("🖼️ Image Classification")

st.markdown(
"""
Upload an image and classify it using one of the trained deep learning models.
"""
)

st.markdown("---")

#Sidebar
st.sidebar.header("Classification Settings")

selected_model = st.sidebar.selectbox(

    "Choose Model",

    [

        "VGG16",

        "ResNet50",

        "MobileNetV2",

        "EfficientNetB0"

    ]
)

#Upload Image
uploaded_image = st.file_uploader(

    "Upload Image",

    type=config.SUPPORTED_FORMATS
)

#Load Class Names
CLASS_NAMES = [

    "Apple",

    "Banana",

    "Bottle",

    "Chair",

    "Dog",

    "Laptop",

    "Orange",

    "Person",

    "Phone",

    "TV"

]
CLASS_NAMES = utils.get_class_names()

#Model Loader
if selected_model == "VGG16":

    model = utils.load_vgg()

    preprocess = vgg_preprocess

elif selected_model == "ResNet50":

    model = utils.load_resnet()

    preprocess = resnet_preprocess

elif selected_model == "MobileNetV2":

    model = utils.load_mobilenet()

    preprocess = mobilenet_preprocess

else:

    model = utils.load_efficientnet()

    preprocess = efficientnet_preprocess

#Prediction
if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Prediction")

        predicted_class, confidence, inference = utils.classify_image(
            model,
            image,
            preprocess,
            CLASS_NAMES
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Prediction", predicted_class)

    with c2:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    with c3:
        st.metric("Inference Time", f"{inference:.3f}s")

    st.progress(confidence)

    img = utils.preprocess_image(image, preprocess)

    prediction = model.predict(img, verbose=0)[0]

    results = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": prediction
    })

    results = results.sort_values(
        by="Probability",
        ascending=False
    )

    st.subheader("Top Predictions")
    st.dataframe(results.head(), width="stretch")

    st.subheader("Prediction Confidence")
    st.bar_chart(results.set_index("Class"))

    csv = results.to_csv(index=False)

    st.download_button(
        "Download Predictions",
        csv,
        file_name="classification_results.csv",
        mime="text/csv"
    )

#Model Information
st.markdown("---")

st.subheader("Selected Model")

if selected_model == "VGG16":

    st.info("""

Architecture : VGG16

Transfer Learning

Input Size : 224×224

Framework : TensorFlow

""")

elif selected_model == "ResNet50":

    st.info("""

Architecture : ResNet50

Residual Learning

Input Size : 224×224

""")

elif selected_model == "MobileNetV2":

    st.info("""

Architecture : MobileNetV2

Lightweight CNN

Designed for Mobile Devices

""")

else:

    st.info("""

Architecture : EfficientNetB0

Compound Scaling

High Accuracy

""")
    
footer()