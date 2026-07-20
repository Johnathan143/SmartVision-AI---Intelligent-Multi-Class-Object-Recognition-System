import os

PROJECT_NAME = "SmartVision AI"
VERSION = "1.0"
AUTHOR = "Mizaru John"
DESCRIPTION = "Intelligent Multi-Class Image Classification and Object Detection"

# DIRECTORY PATHS
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_DATASET = os.path.join(BASE_DIR,"classification","train")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# MODEL PATHS
VGG_MODEL = os.path.join(MODELS_DIR, "vgg16_best.keras")
RESNET_MODEL = os.path.join(MODELS_DIR, "resnet50_best.keras")
MOBILENET_MODEL = os.path.join(MODELS_DIR, "mobilenetv2_best.keras")
EFFICIENTNET_MODEL = os.path.join(MODELS_DIR,"efficientnetb0_best.keras")
YOLO_MODEL = os.path.join(MODELS_DIR, "best.pt")

# IMAGE SETTINGS
IMAGE_SIZE = (224,224)
YOLO_IMAGE_SIZE = 640

# STREAMLIT THEME
PRIMARY_COLOR = "#0E6EFD"
BACKGROUND_COLOR = "#0E1117"
SECONDARY_BACKGROUND = "#262730"
TEXT_COLOR = "#FAFAFA"

# SUPPORTED IMAGE TYPES
SUPPORTED_FORMATS = [
    "jpg",
    "jpeg",
    "png"
]