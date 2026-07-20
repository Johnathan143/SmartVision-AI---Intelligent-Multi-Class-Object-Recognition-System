import time
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

from ultralytics import YOLO

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

import config

import os

def get_class_names():

    classes = sorted([
        folder
        for folder in os.listdir(config.CLASSIFICATION_DATASET)
        if os.path.isdir(
            os.path.join(config.CLASSIFICATION_DATASET, folder)
        )
    ])

    return classes

@st.cache_resource
def load_vgg():

    return load_model(config.VGG_MODEL)


@st.cache_resource
def load_resnet():

    return load_model(config.RESNET_MODEL)


@st.cache_resource
def load_mobilenet():

    return load_model(config.MOBILENET_MODEL)


@st.cache_resource
def load_efficientnet():

    return load_model(config.EFFICIENTNET_MODEL)


@st.cache_resource
def load_yolo():

    return YOLO(config.YOLO_MODEL)

def preprocess_image(image, preprocess_function):

    image = image.resize(config.IMAGE_SIZE)

    image = np.array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_function(image)

    return image

def classify_image(model, image, preprocess_function, class_names):

    img = preprocess_image(image, preprocess_function)

    start = time.time()

    prediction = model.predict(img, verbose=0)

    end = time.time()

    confidence = float(np.max(prediction))

    predicted_index = int(np.argmax(prediction))

    predicted_class = class_names[predicted_index]

    inference = end - start

    return predicted_class, confidence, inference

def detect_objects(model, image, conf=0.25):

    image_np = np.array(image)

    start = time.time()

    results = model.predict(
        image_np,
        conf=conf,
        verbose=False
    )

    end = time.time()

    inference = end - start

    annotated = results[0].plot()

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    return results, annotated, inference

def metric_card(title, value):

    st.metric(
        label=title,
        value=value
    )

def footer():

    st.markdown("---")

    st.caption(
        "SmartVision AI • Built using TensorFlow, YOLOv8 and Streamlit"
    )