import cv2
import time
from streamlit_image_comparison import image_comparison

def overlay_glow(result):
    """
    Overlays a glowing bounding box on the result image.
    Color changes based on time to create a shimmering effect.
    """
    img = result.copy()
    if hasattr(result, "boxes"):
        for box in result.boxes.xyxy:
            # Create a dynamic color based on time
            color = (0, int((time.time() * 100) % 255), 255)
            cv2.rectangle(
                img, 
                (int(box[0]), int(box[1])), 
                (int(box[2]), int(box[3])), 
                color, 
                2
            )
    return img

def before_after_slider(original, result, width=700):
    """
    Displays the before and after image comparison slider.
    """
    image_comparison(original, result, width=width)
