import os
import cv2
import numpy as np
import torch
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pathlib
import sys

# Patch PosixPath to WindowsPath for Windows compatibility
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

# ==== CONFIGURATION ====
best_weights_path = r'D:\RD\b1\best.pt'  # Path to your YOLOv5 model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add YOLOv5 repo path if needed
yolov5_repo_path = os.path.join(os.getcwd(), 'yolov5')
if yolov5_repo_path not in sys.path:
    sys.path.append(yolov5_repo_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Load YOLOv5 model
yolo_model = DetectMultiBackend(best_weights_path, device=device)

# Load MiDaS model and transforms
midas_model = torch.hub.load("isl-org/MiDaS", "DPT_Large").to(device).eval()
midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms").dpt_transform

# Function to process an image and return annotated image and details
def process_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # YOLOv5 Detection
    with torch.no_grad():
        pred = yolo_model(img_tensor)
        detections = non_max_suppression(pred)[0]

    if detections is None or detections.shape[0] == 0:
        cv2.putText(img, "No Damages Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img, []

    boxes = detections[:, :4].cpu().numpy().astype(int)
    confidences = detections[:, 4].cpu().numpy()
    class_ids = detections[:, -1].cpu().numpy().astype(int)

    # MiDaS Depth Estimation
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

        # Draw and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}, Conf: {confidence:.2f}, Depth: {pothole_depth:.2f}m, Area: {pothole_area}px², Sev: {severity}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        results.append({
            "class_name": class_name,
            "confidence": confidence,
            "depth": pothole_depth,
            "area": pothole_area,
            "severity": severity,
            "severity_score": severity_score,
            "image": img.copy()
        })

    return img, results

# Webcam transformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated_img, _ = process_image(img)
        self.frame = annotated_img
        return annotated_img

# Streamlit App
st.title("Pothole Detection System")
st.sidebar.title("Options")

option = st.sidebar.selectbox("Choose an option", ["Upload Single Image", "Upload Multiple Images", "Real-Time Camera"])

# Single Image Upload
if option == "Upload Single Image":
    uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        annotated_img, results = process_image(img)
        st.image(annotated_img, channels="BGR", caption="Annotated Image")
        for res in results:
            st.write(f"{res['class_name']}: Conf = {res['confidence']:.2f}, Depth = {res['depth']:.2f}m, Area = {res['area']}px², Severity = {res['severity']}")

# Multiple Image Upload with Severity Sorting
elif option == "Upload Multiple Images":
    uploaded_files = st.file_uploader("Upload multiple road images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        all_results = []
        for uploaded_file in uploaded_files:
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            annotated_img, results = process_image(img)
            for res in results:
                res["annotated_image"] = annotated_img
                all_results.append(res)

        # Sort by severity score (descending)
        sorted_results = sorted(all_results, key=lambda x: x["severity_score"], reverse=True)

        st.subheader("Results Sorted by Severity")
        for res in sorted_results:
            st.image(res["annotated_image"], channels="BGR", caption=f"{res['class_name']} - Severity: {res['severity']}")
            st.write(f"Conf = {res['confidence']:.2f}, Depth = {res['depth']:.2f}m, Area = {res['area']}px²")

# Real-Time Camera with Location Tagging
elif option == "Real-Time Camera":
    st.write("Use your camera to detect potholes in real-time:")
    ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
    if ctx.video_transformer and ctx.video_transformer.frame is not None:
        st.image(ctx.video_transformer.frame, channels="BGR", caption="Real-Time Detection")

    # Location tagging (manual input for simplicity)
    location = st.text_input("Enter location (e.g., latitude, longitude)", "")
    if location and ctx.video_transformer and ctx.video_transformer.frame is not None:
        st.write(f"Image captured at: {location}")

# Instructions
st.sidebar.markdown("""
### Instructions
1. **Single Image**: Upload one image to see pothole detection results.
2. **Multiple Images**: Upload multiple images to view results sorted by severity.
3. **Real-Time Camera**: Use your webcam for live detection (requires `streamlit-webrtc`).
""")