import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

def roi_selector(image_key, frame_or_path, width=700, height=500):
    """
    Renders a canvas for drawing ROI.
    Returns the canvas result object.
    """
    st.markdown("##### ✏️ Draw Region of Interest")
    st.caption("Draw a rectangle or polygon to define the area for detection.")
    
    # Prepare background image
    if isinstance(frame_or_path, str) or isinstance(frame_or_path, Path):
        bg_image = Image.open(frame_or_path) 
    elif isinstance(frame_or_path, np.ndarray):
        bg_image = Image.fromarray(cv2.cvtColor(frame_or_path, cv2.COLOR_BGR2RGB))
    else:
        bg_image = frame_or_path

    # st_canvas wrapper
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color="#E04F5F",
        background_image=bg_image,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode="polygon", # Default to polygon for flexibility
        key=f"canvas_{image_key}",
    )
    return canvas_result

def process_roi_coords(canvas_result, img_shape):
    """
    Converts canvas JSON output to a binary mask.
    img_shape: (height, width)
    """
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if not objects:
            return None
        
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        for obj in objects:
            if obj["type"] == "rect":
                x = int(obj["left"])
                y = int(obj["top"])
                w = int(obj["width"])
                h = int(obj["height"])
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                
            elif obj["type"] == "path":
                # Polygon paths need parsing
                # This is a simplified handler; complex SVG paths might need more work
                # But st_canvas polygon tool usually outputs 'path' with straightforward points
                points = []
                for p in obj["path"]:
                    if p[0] == 'M' or p[0] == 'L':
                        points.append([int(p[1]), int(p[2])])
                if points:
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
                    
        return mask
    return None

def filter_results_by_roi(results, mask):
    """
    Filters YOLOv8 results to keep only detections within the ROI mask.
    """
    if mask is None:
        return results
        
    filtered_boxes = []
    
    # Assuming batch size 1 for simplicity
    result = results[0]
    
    if not hasattr(result, "boxes") or result.boxes is None:
        return results

    # We need to manually construct a new result or modify existing one
    # Modifying internal state of Ultralytics objects is risky, 
    # so we might just zero-out conf for invalid boxes or list valid indices
    
    valid_indices = []
    
    for i, box in enumerate(result.boxes):
        # Check center of the box
        x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
        y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
        
        # Check if center is within mask (mask value 255)
        # Ensure coordinates are within image bounds
        h, w = mask.shape
        if 0 <= y_center < h and 0 <= x_center < w:
            if mask[y_center, x_center] > 0:
                valid_indices.append(i)
                
    if valid_indices:
        # Create a new Boxes object with only valid indices
        # This is the cleanest way using Ultralytics API
        return [result[valid_indices]]
    else:
        # Return empty result-like object or cleared result
        # Simplest is to return a result with 0 boxes
        # result.update(boxes=torch.zeros(0,6)) is one way but API varies
        # Let's try to just return the filtered subset (which might handle empty gracefully)
        return [result[[]]] # Empty selection

def draw_roi_overlay(image, mask):
    """
    Draws the ROI boundary on the image.
    """
    if mask is None:
        return image
        
    overlay = image.copy()
    
    # Find contours of the mask to draw boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    # Semi-transparent fill outside? or inside?
    # Let's make outside darker
    # Invert mask
    inv_mask = cv2.bitwise_not(mask)
    
    # Create dark overlay
    dark_layer = np.zeros_like(image)
    
    # Apply alpha blending
    alpha = 0.3
    
    # Darken outside ROI
    # image = image * 0.7 + black * 0.3 for pixels outside ROI
    # We can do this with masking
    
    # Convert mask to 3-channel
    inv_mask_3ch = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    
    # Darken the original image
    darkened = (image * (1 - alpha)).astype(np.uint8)
    
    # Combine: where mask is 0 (outside ROI), use darkened. Where mask is 255 (inside), use original
    # logic: if mask check
    
    # Easier opencv way:
    # 1. Copy original
    # 2. Draw black rectangle over whole thing
    # 3. Add weighted
    # 4. But we need ROI to be clear.
    
    # Final Plan:
    # return image with green contour
    return overlay
