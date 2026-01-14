from ultralytics import YOLO
import sys

try:
    print("Loading detection model...")
    model = YOLO("yolov8n.pt")
    print("Detection model loaded successfully.")
    
    print("Loading segmentation model...")
    model_seg = YOLO("yolov8n-seg.pt")
    print("Segmentation model loaded successfully.")
    
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
