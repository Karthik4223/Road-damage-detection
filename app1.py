import os
import cv2
import numpy as np
import torch
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pathlib
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Patch PosixPath to WindowsPath for Windows compatibility
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

# ==== CONFIGURATION ====
best_weights_path = r'D:\RD\b1\best.pt'  # Path to your YOLOv5 model weights (ensure it's yolov5s for lightweight)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add YOLOv5 repo path if needed
yolov5_repo_path = os.path.join(os.getcwd(), 'yolov5')
if yolov5_repo_path not in sys.path:
    sys.path.append(yolov5_repo_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Load lightweight models once at startup
@lru_cache(maxsize=1)
def load_models():
    yolo_model = DetectMultiBackend(best_weights_path, device=device)  # Ideally yolov5s
    midas_model = torch.hub.load("isl-org/MiDaS", "MiDaS_small").to(device).eval()  # Lightweight MiDaS
    midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms").small_transform
    return yolo_model, midas_model, midas_transforms

yolo_model, midas_model, midas_transforms = load_models()

# Resize image to reduce computational load
def resize_image(img, max_size=640):
    h, w = img.shape[:2]
    scale = min(max_size / h, max_size / w)
    if scale < 1:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

# Lightweight image processing function
def process_image(img):
    img = resize_image(img)  # Downscale for speed
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # YOLOv5 Detection
    with torch.no_grad():
        pred = yolo_model(img_tensor)
        detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]  # Faster thresholds

    if detections is None or detections.shape[0] == 0:
        cv2.putText(img, "No Damages Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img, []

    boxes = detections[:, :4].cpu().numpy().astype(int)
    confidences = detections[:, 4].cpu().numpy()
    class_ids = detections[:, -1].cpu().numpy().astype(int)

    # MiDaS Depth Estimation (lightweight)
    input_batch = midas_transforms(img_rgb).to(device)
    with torch.no_grad():
        depth_map = midas_model(input_batch)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # Calculate road surface depth
    road_surface_mask = np.ones_like(depth_map, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = box
        road_surface_mask[y1:y2, x1:x2] = False
    road_surface_depth = np.mean(depth_map[road_surface_mask])

    # Annotate and calculate severity
    results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        confidence = confidences[i]
        class_name = 'Pothole' if class_ids[i] == 1 else 'Road Damage'

        depth_patch = depth_map[y1:y2, x1:x2]
        pothole_depth = road_surface_depth - np.min(depth_patch) if depth_patch.size > 0 else 0
        pothole_area = (x2 - x1) * (y2 - y1)
        severity_score = pothole_depth * pothole_area

        severity = "Low" if severity_score < 1000 else "Medium" if severity_score < 5000 else "High"

        # Lightweight annotation
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        label = f"{class_name}, Conf: {confidence:.2f}, Sev: {severity}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        results.append({
            "class_name": class_name,
            "confidence": confidence,
            "depth": pothole_depth,
            "area": pothole_area,
            "severity": severity,
            "severity_score": severity_score,
            "image": img.copy()
        })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU memory
    return img, results

# Async batch processing for multiple images
async def process_multiple_images_async(uploaded_files, batch_size=2):
    loop = asyncio.get_event_loop()
    all_results = []
    batches = [uploaded_files[i:i + batch_size] for i in range(0, len(uploaded_files), batch_size)]

    with ThreadPoolExecutor() as executor:
        for i, batch in enumerate(batches):
            batch_images = [cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1) for f in batch]
            batch_results = await loop.run_in_executor(executor, lambda imgs: [process_image(img) for img in imgs], batch_images)
            for annotated_img, results in batch_results:
                for res in results:
                    res["annotated_image"] = annotated_img
                    all_results.append(res)
            progress_bar.progress((i + 1) / len(batches))  # Progress feedback
    return all_results

# Webcam transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = resize_image(img)  # Downscale for real-time
        annotated_img, _ = process_image(img)
        return annotated_img

# Streamlit App
st.title("Pothole Detection System")
st.sidebar.title("Options")

option = st.sidebar.selectbox("Choose an option", ["Upload Single Image", "Upload Multiple Images", "Real-Time Camera"])

# Global progress bar for multiple images
progress_bar = st.empty()

# Single Image Upload
if option == "Upload Single Image":
    uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            annotated_img, results = process_image(img)
        st.image(annotated_img, channels="BGR", caption="Annotated Image")
        for res in results:
            st.write(f"{res['class_name']}: Conf = {res['confidence']:.2f}, Depth = {res['depth']:.2f}m, Area = {res['area']}px², Severity = {res['severity']}")

# Multiple Image Upload with Severity Sorting
elif option == "Upload Multiple Images":
    uploaded_files = st.file_uploader("Upload multiple road images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing images..."):
            progress_bar.progress(0)  # Initialize progress bar
            all_results = asyncio.run(process_multiple_images_async(uploaded_files, batch_size=2))
            sorted_results = sorted(all_results, key=lambda x: x["severity_score"], reverse=True)

        st.subheader("Results Sorted by Severity")
        for res in sorted_results:
            st.image(res["annotated_image"], channels="BGR", caption=f"{res['class_name']} - Severity: {res['severity']}")
            st.write(f"Conf = {res['confidence']:.2f}, Depth = {res['depth']:.2f}m, Area = {res['area']}px²")

# Real-Time Camera
elif option == "Real-Time Camera":
    st.write("Use your camera to detect potholes in real-time:")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Instructions
st.sidebar.markdown("""
### Instructions
1. **Single Image**: Upload one image to see pothole detection results.
2. **Multiple Images**: Upload multiple images to view results sorted by severity.
3. **Real-Time Camera**: Use your webcam for live detection.
""")