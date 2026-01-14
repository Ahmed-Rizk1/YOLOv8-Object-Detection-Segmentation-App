import streamlit as st
from ultralytics import YOLO
from .settings import DETECTION_MODEL, SEGMENTATION_MODEL

@st.cache_resource
def load_models():
    """
    Loads and caches the YOLOv8 detection and segmentation models.
    """
    det_model = YOLO(DETECTION_MODEL)
    seg_model = YOLO(SEGMENTATION_MODEL)
    return det_model, seg_model
