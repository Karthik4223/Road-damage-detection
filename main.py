import os
import cv2
import numpy as np
import torch
import sys

# ==== CONFIGURATION ====
image_dir = r'D:\pothole\test\images'  # Your local test image directory
output_dir = r'D:\RD\Output'      # Folder to save annotated results

import pathlib
import sys

# Patch PosixPath to WindowsPath
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath


best_weights_path = r'D:\RD\b1\best.pt'  # Path to your best.pt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== SETUP ====
os.makedirs(output_dir, exist_ok=True)

# Add YOLOv5 repo path if needed
yolov5_repo_path = os.path.join(os.getcwd(), 'yolov5')
if yolov5_repo_path not in sys.path:
    sys.path.append(yolov5_repo_path)

# Import YOLOv5 modules after appending the repo path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Load YOLOv5 model
yolo_model = DetectMultiBackend(best_weights_path, device=device)

# Load MiDaS model and transforms
midas_model = torch.hub.load("isl-org/MiDaS", "DPT_Large").to(device).eval()
midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms").dpt_transform

# ==== PROCESS IMAGES ====
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_dir, filename)
    print(f"Processing: {filename}")

    # Step 1: YOLOv5 Detection
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = yolo_model(img_tensor)
        detections = non_max_suppression(pred)[0]

        if detections is None or detections.shape[0] == 0:
            print(f"No Damages detected in {filename}")
            # Annotate the original image
            cv2.putText(img, "No Damages Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            # Save the image to output directory
            output_path = os.path.join(output_dir, f"annotated_{filename}")
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")
            continue

    boxes = detections[:, :4].cpu().numpy().astype(int)
    confidences = detections[:, 4].cpu().numpy()
    class_ids = detections[:, -1].cpu().numpy().astype(int)

    # Step 2: MiDaS Depth Estimation
    input_batch = midas_transforms(img_rgb).to(device)
    with torch.no_grad():
        depth_map = midas_model(input_batch)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    # Calculate road surface depth (baseline)
    road_surface_mask = np.ones_like(depth_map, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = box
        road_surface_mask[y1:y2, x1:x2] = False
    road_surface_depth = np.mean(depth_map[road_surface_mask])

    # Step 3: Annotate Potholes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        confidence = confidences[i]
        class_name = 'Pothole' if class_ids[i] == 1 else 'Road Damage'

        depth_patch = depth_map[y1:y2, x1:x2]
        if depth_patch.size > 0:
            min_pothole_depth = np.min(depth_patch)
            pothole_depth = road_surface_depth - min_pothole_depth
        else:
            pothole_depth = 0

        pothole_area = (x2 - x1) * (y2 - y1)
        severity_score = pothole_depth * pothole_area

        if severity_score < 1000:
            severity = "Low"
        elif severity_score < 5000:
            severity = "Medium"
        else:
            severity = "High"

        # Draw and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}, Conf: {confidence:.2f}, Depth: {pothole_depth:.2f}m, Area: {pothole_area}px², Sev: {severity}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        print(f"{class_name} {i+1}: Conf = {confidence:.2f}, Depth = {pothole_depth:.2f}m, Area = {pothole_area}px², Severity = {severity}")

    # Save output image
    output_path = os.path.join(output_dir, f"annotated_{filename}")
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
