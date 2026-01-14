# 🖼 YOLOv8 Object Detection & Segmentation App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)

**A modular, professional Streamlit web application for real-time object detection and segmentation using YOLOv8.**
Seamlessly process images, videos, or live webcam feeds with a modern UI.

---

## 📌 Features

- **Advanced Detection & Segmentation**:
  - Leverages **YOLOv8** for state-of-the-art accuracy.
  - Supports both bounding box detection and pixel-wise segmentation.

- **📊 Analytics Dashboard**:
  - **Real-time Stats**: View total object counts, average confidence, and FPS.
  - **Charts**: Visual distribution of detected classes and confidence scores.
  - **Timeline**: Track detection activity over video frames.

- **📐 Region of Interest (ROI)**:
  - **Interactive Drawing**: Use the built-in canvas to draw Polygons or Rectangles on images.
  - **Precision Filtering**: Analyze only the specific parts of the image you care about.
  - **Visual Feedback**: Clearly see the ROI boundary overlaid on results.

- **Versatile Input Support**:
  - **Images**: `.jpg`, `.jpeg`, `.png`
  - **Videos**: `.mp4`, `.mov`, `.avi`
  - **Live Webcam**: Real-time inference.

- **Interactive User Interface**:
  - **Before/After Comparison**: Slide to see the difference clearly.
  - **Dynamic Visuals**: Glowing bounding boxes for a futuristic feel.
  - **Adjustable Parameters**: Fine-tune confidence thresholds on the fly.

- **Export Ready**:
  - Download processed images and videos immediately.

---

## 📂 Project Structure

This project follows a modular architecture for scalability and maintainability:

```
├── src/
│   ├── settings.py   # Configuration constants
│   ├── model.py      # YOLOv8 model loading & handling
│   ├── utils.py      # Helper functions (plotting, effects)
│   ├── analytics.py  # Dashboard & Statistics logic
│   └── roi.py        # Canvas & ROI filtering logic
├── app.py            # Main Streamlit application entry point
├── requirements.txt  # Project dependencies
└── README.md         # Documentation
```

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yolov8-object-detection
   ```

2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## 📸 Demo

### Image Detection
![Image Detection Screenshot](./Screenshot%20(65).png)

### Video Processing
![Video Processing Screenshot](./Screenshot%20(66).png)
