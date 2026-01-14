import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from pathlib import Path

import src.settings as settings
from src.utils import overlay_glow, before_after_slider
from src.model import load_models

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Object Detection App", layout="wide")

# ------------------ ANIMATED TITLE ------------------
st.markdown(
    """
<div style="text-align:center; font-size:40px; font-weight:700; color:#3E8DED;
            animation: glow 1.5s ease-in-out infinite alternate;">
Detecting Objects…
</div>
<style>
@keyframes glow {0% { text-shadow: 0 0 5px #3E8DED;} 100% { text-shadow:0 0 20px #3E8DED;}}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center; font-size:18px; color:#666;'>Detect or Segment objects in images, videos, or live webcam!</div>",
    unsafe_allow_html=True,
)


# ------------------ LOAD MODELS ------------------
det_model, seg_model = load_models()

# ------------------ SIDEBAR ------------------
with st.sidebar.expander("⚙️ Settings", expanded=True):
    conf_thres = st.slider("Confidence Threshold", 0.1, 1.0, 0.35)
    seg_mode = st.checkbox("🎨 Enable Segmentation Mode (YOLOv8-Seg)")
    
    source_type = st.radio(
        "Choose Input Source",
        [settings.IMAGE, settings.VIDEO, settings.WEBCAM],
        index=0,
    )

# ------------------ IMAGE UPLOAD -------------------------
if source_type == settings.IMAGE:
    st.subheader("🖼 Object Detection on Images")
    
    # Example Selection
    example_images = list(settings.IMAGES_DIR.glob("*.jpg")) + list(settings.IMAGES_DIR.glob("*.jpeg")) + list(settings.IMAGES_DIR.glob("*.png"))
    example_names = [p.name for p in example_images]
    
    col1, col2 = st.columns([1, 1])
    
    img = None
    
    with col1:
        st.markdown("**Option 1: Choose an Example**")
        selected_example = st.selectbox("Select an example image", ["None"] + example_names)
        
        if selected_example != "None":
            img = Image.open(settings.IMAGES_DIR / selected_example)

    with col2:
        st.markdown("**Option 2: Upload Your Own**")
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded)
    
    if img:
        st.divider()
        img_array = np.array(img)
        model = seg_model if seg_mode else det_model

        # Perform prediction
        results = model.predict(
            cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), conf=conf_thres, verbose=False
        )

        result_img = results[0].plot()
        result_img = overlay_glow(result_img)

        st.markdown("### Before / After Comparison")
        before_after_slider(img, result_img)

        # Download Button
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmp_file.name, result_img)

        with open(tmp_file.name, "rb") as f:
            st.download_button("⬇️ Download Result Image", f, "detection_result.jpg")
    else:
        st.info("Please select an example or upload an image to start.")

# ------------------ VIDEO UPLOAD -------------------------
elif source_type == settings.VIDEO:
    st.subheader("🎬 Object Detection on Videos")
    
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        st.video(uploaded_video)

        if st.button("🚀 Process Video"):
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

            with open(temp_input.name, "wb") as f:
                f.write(uploaded_video.read())

            cap = cv2.VideoCapture(temp_input.name)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(temp_output.name, fourcc, fps, (w, h))

            progress = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            stframe = st.empty()
            thumbnails = []

            model = seg_model if seg_mode else det_model

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame += 1

                results = model.predict(frame, conf=conf_thres, verbose=False)
                result_frame = results[0].plot()
                result_frame = overlay_glow(result_frame)

                out.write(result_frame)

                if current_frame % int(fps * 2) == 0:
                    thumb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    thumbnails.append(Image.fromarray(thumb))

                if current_frame % 10 == 0:
                    stframe.image(
                        result_frame[:, :, ::-1],
                        caption=f"Processing Frame {current_frame}/{frame_count}",
                        use_container_width=True,
                    )

                if frame_count > 0:
                    progress.progress(current_frame / frame_count)

            cap.release()
            out.release()

            st.success("✔️ Video processed!")
            st.video(temp_output.name)

            with open(temp_output.name, "rb") as f:
                st.download_button("⬇️ Download Video", f, "video_processed.mp4")

            if thumbnails:
                st.markdown("### 🎞 Video Thumbnails")
                cols = st.columns(len(thumbnails))
                for i, thumb in enumerate(thumbnails):
                    cols[i].image(
                        thumb, caption=f"Frame {i * 2}s", use_container_width=True
                    )

# ------------------ WEBCAM LIVE DETECTION ----------------
elif source_type == settings.WEBCAM:
    st.subheader("📡 Live Object Detection / Segmentation (Webcam)")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Cannot access webcam.")
                break

            model = seg_model if seg_mode else det_model

            results = model(frame)
            result_frame = results[0].plot()
            stframe.image(result_frame, channels="BGR")

        cap.release()
