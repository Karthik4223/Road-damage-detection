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
best_weights_path ='best.pt'  # Path to your YOLOv5 model weights (ideally yolov5s)
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
    yolo_model = DetectMultiBackend(best_weights_path, device=device)
    midas_model = torch.hub.load("isl-org/MiDaS", "MiDaS_small").to(device).eval()
    midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms").small_transform
    return yolo_model, midas_model, midas_transforms

yolo_model, midas_model, midas_transforms = load_models()

# Resize image for processing and display
def resize_image(img, max_size=640, display_size=(300, 300)):
    h, w = img.shape[:2]
    scale = min(max_size / h, max_size / w)
    if scale < 1:
        proc_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        proc_img = img.copy()
    disp_img = cv2.resize(proc_img, display_size, interpolation=cv2.INTER_AREA)
    return proc_img, disp_img, scale

# Lightweight image processing with adjusted depth measurement
def process_image(img):
    proc_img, disp_img, scale = resize_image(img)
    img_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # YOLOv5 Detection
    with torch.no_grad():
        pred = yolo_model(img_tensor)
        detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if detections is None or detections.shape[0] == 0:
        cv2.putText(proc_img, "No Damages Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, disp_img, _ = resize_image(proc_img)
        return disp_img, []

    boxes = detections[:, :4].cpu().numpy().astype(int)
    confidences = detections[:, 4].cpu().numpy()
    class_ids = detections[:, -1].cpu().numpy().astype(int)

    # MiDaS Depth Estimation
    input_batch = midas_transforms(img_rgb).to(device)
    with torch.no_grad():
        depth_map = midas_model(input_batch)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1), size=proc_img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # Normalize depth map to a reasonable range (e.g., 0 to 1)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Calculate road surface depth
    road_surface_mask = np.ones_like(depth_map, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = box
        road_surface_mask[y1:y2, x1:x2] = False
    road_surface_depth = np.mean(depth_map[road_surface_mask])

    # Annotate with bright and clear bounding boxes and labels
    results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        confidence = confidences[i]
        class_name = 'Pothole' if class_ids[i] == 1 else 'Road Damage'  # Keep class labels as they are

        # Calculate pothole depth with adjustment
        depth_patch = depth_map[y1:y2, x1:x2]
        pothole_depth = road_surface_depth - np.min(depth_patch) if depth_patch.size > 0 else 0

        # Adjust depth based on pothole size
        pothole_area = (x2 - x1) * (y2 - y1)  # Area in pixels
        area_factor = min(1.0, pothole_area / 10000)  # Normalize area
        pothole_depth = pothole_depth * area_factor * 5.0  # Scale to a believable range
        pothole_depth = min(pothole_depth, 1.0)  # Cap depth at 1m

        # Convert depth to meters
        pothole_depth_meters = pothole_depth * 1.0

        # Adjust severity calculation
        severity_score = pothole_depth * pothole_area
        severity = "Low" if pothole_depth_meters < 0.1 else "Medium" if pothole_depth_meters < 0.5 else "High"

        # Use severity-based colors
        color = (0, 255, 0) if severity == "Low" else (0, 165, 255) if severity == "Medium" else (0, 0, 255)  # Bright Green, Bright Orange, Bright Red

        # Draw thicker bounding box
        cv2.rectangle(proc_img, (x1, y1), (x2, y2), color, 2)

        # Add label with background for clarity (use "Road Damage" in the label)
        label = f"Road Damage, Conf: {confidence:.2f}, Sev: {severity}"
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        bg_x1, bg_y1 = x1, y1 - text_h - 5
        bg_x2, bg_y2 = x1 + text_w + 5, y1
        cv2.rectangle(proc_img, (bg_x1, bg_y1 - 5), (bg_x2, bg_y2), (255, 255, 255), -1)  # White background
        cv2.putText(proc_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # Resize annotated image for display
        _, annotated_disp_img, _ = resize_image(proc_img)

        results.append({
            "class_name": class_name,  # Keep original class name internally
            "confidence": confidence,
            "depth": pothole_depth_meters,
            "area": pothole_area,
            "severity": severity,
            "severity_score": severity_score,
            "image": annotated_disp_img
        })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return annotated_disp_img, results

# Async batch processing for multiple images
async def process_multiple_images_async(uploaded_files, batch_size=2):
    loop = asyncio.get_event_loop()
    all_results = []
    batches = [uploaded_files[i:i + batch_size] for i in range(0, len(uploaded_files), batch_size)]

    with ThreadPoolExecutor() as executor:
        for i, batch in enumerate(batches):
            batch_images = [cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1) for f in batch]
            batch_results = await loop.run_in_executor(executor, lambda imgs: [process_image(img) for img in imgs], batch_images)
            for _, results in batch_results:
                all_results.extend(results)
            progress_bar.progress((i + 1) / len(batches))
    return all_results

# Webcam transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        proc_img, _, _ = resize_image(img)
        annotated_img, _ = process_image(proc_img)
        return annotated_img

# Streamlit App with Updated Title
st.set_page_config(page_title="Road Damage Detection System", layout="wide")
st.title("Road Damage Detection System")
st.markdown("---")

# Sidebar with styled options
with st.sidebar:
    st.markdown("<h2 style='color: #00ccff;'>Options</h2>", unsafe_allow_html=True)
    option = st.selectbox("Choose an option", ["Upload Single Image", "Upload Multiple Images", "Real-Time Camera"], 
                          help="Select how you want to detect road damages.")
    st.markdown("""
    <h3 style='color: #ffffff;'>Instructions</h3>
    <ul style='color: #ffffff;'>
        <li><b>Single Image</b>: Upload one image for detection.</li>
        <li><b>Multiple Images</b>: Upload multiple images to see results side by side, sorted by severity.</li>
        <li><b>Real-Time Camera</b>: Use your webcam for live detection.</li>
    </ul>
    """, unsafe_allow_html=True)

# Global progress bar
progress_bar = st.empty()

# Single Image Upload
if option == "Upload Single Image":
    st.subheader("Upload a Road Image", anchor=None)
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            annotated_img, results = process_image(img)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(annotated_img, channels="BGR", caption="Annotated Image", use_column_width=True)
        with col2:
            st.markdown("<h3 style='color: #00ccff;'>Detection Details</h3>", unsafe_allow_html=True)
            for res in results:
                severity_color = "#28a745" if res['severity'] == "Low" else "#ff9900" if res['severity'] == "Medium" else "#dc3545"
                st.markdown(f"""
                <div style='background-color: #222222; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <p style='color: #ffffff;'><b>Type</b>: Road Damage</p>
                    <p style='color: #ffffff;'><b>Confidence</b>: {res['confidence']:.2f}</p>
                    <p style='color: #ffffff;'><b>Depth</b>: {res['depth']:.2f}m</p>
                    <p style='color: #ffffff;'><b>Area</b>: {res['area']}px²</p>
                    <p style='color: {severity_color};'><b>Severity</b>: {res['severity']}</p>
                </div>
                """, unsafe_allow_html=True)

# Multiple Image Upload with Side-by-Side UI
elif option == "Upload Multiple Images":
    st.subheader("Upload Multiple Road Images", anchor=None)
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing images..."):
            progress_bar.progress(0)
            all_results = asyncio.run(process_multiple_images_async(uploaded_files, batch_size=2))
            sorted_results = sorted(all_results, key=lambda x: x["severity_score"], reverse=True)

        st.markdown("<h2 style='color: #00ccff;'>Results Sorted by Severity</h2>", unsafe_allow_html=True)
        st.markdown("---")
        for i, res in enumerate(sorted_results):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(res["image"], channels="BGR", caption=f"Road Damage - Severity: {res['severity']}", use_column_width=True)
            with col2:
                severity_color = "#28a745" if res['severity'] == "Low" else "#ff9900" if res['severity'] == "Medium" else "#dc3545"
                st.markdown(f"""
                <h4 style='color: #ffffff;'>Detection {i + 1}</h4>
                <div style='background-color: #222222; padding: 10px; border-radius: 5px;'>
                    <p style='color: #ffffff;'><b>Type</b>: Road Damage</p>
                    <p style='color: #ffffff;'><b>Confidence</b>: {res['confidence']:.2f}</p>
                    <p style='color: #ffffff;'><b>Depth</b>: {res['depth']:.2f}m</p>
                    <p style='color: #ffffff;'><b>Area</b>: {res['area']}px²</p>
                    <p style='color: {severity_color};'><b>Severity</b>: {res['severity']}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

# Real-Time Camera
elif option == "Real-Time Camera":
    st.subheader("Real-Time Road Damage Detection", anchor=None)
    st.write("Use your camera to detect road damages live:")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Custom CSS for Black Background and Clear Text
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .stSpinner > div > div {
        border-color: #00ccff !important;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }
    .stProgress > div > div > div > div {
        background-color: #00ccff;
    }
    .stFileUploader label {
        color: #ffffff !important;
    }
    .stSelectbox label {
        color: #ffffff !important;
    }
    .stFileUploader div[role='button'] {
        background-color: #0066cc !important;
        color: #ffffff !important;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)