"""
utils_a2.py
Shared utilities for Assignment 2 (Tasks 1–3)
GroundingDINO + Prompt Learning + Evaluation
"""

import os
import requests
import torch
import pandas as pd
import numpy as np
from torchvision.ops import box_iou

# ---------------------------------------------------------
# DEFAULT ARGUMENTS (shared across all scripts)
# ---------------------------------------------------------
DEFAULT_PROMPTS = {
    "A": ["a malignant tumor"],
    "B": ["a malignant tumor"],
    "C": ["a malignant tumor"],
}

DEFAULT_THRESHOLDS = {
    "box_threshold": 0.35,
    "text_threshold": 0.25,
}

GROUNDING_DINO_CONFIG_URL = (
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

GROUNDING_DINO_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
)

GROUNDING_DINO_CONFIG_URL_SWINB = (
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/",
    "groundingdino/config/GroundingDINO_SwinB_cfg.py"
)

GROUNDING_DINO_WEIGHTS_URL_SWINB = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/",
    "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
)

# ---------------------------------------------------------
# Path Utilities
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def path_exists(path):
    return os.path.exists(path)

# ---------------------------------------------------------
# Download Utility
# ---------------------------------------------------------
def download_file(url, dst_path):
    if os.path.exists(dst_path):
        return dst_path
    print(f"[INFO] Downloading: {url}")
    resp = requests.get(url)
    with open(dst_path, "wb") as f:
        f.write(resp.content)
    print(f"[INFO] Saved to {dst_path}")
    return dst_path

# ---------------------------------------------------------
# Annotation Utilities
# ---------------------------------------------------------
def load_annotations(csv_path):
    return pd.read_csv(csv_path)

def get_gt_boxes(df, image_name):
    rows = df[df["image_name"] == image_name]
    return rows[["xmin", "ymin", "xmax", "ymax"]].values

# ---------------------------------------------------------
# Evaluating Predicted vs GT (AP Computation)
# ---------------------------------------------------------
def compute_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Simple AP estimator (one-point precision × recall).
    """
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    ious = box_iou(
        torch.tensor(pred_boxes, dtype=torch.float32),
        torch.tensor(gt_boxes, dtype=torch.float32)
    ).numpy()

    matched = set()
    tp, fp = 0, 0

    for i in range(len(pred_boxes)):
        max_iou = ious[i].max()
        if max_iou >= iou_threshold:
            g = np.argmax(ious[i])
            if g not in matched:
                matched.add(g)
                tp += 1
            else:
                fp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision * recall  # simplified AP

# ---------------------------------------------------------
# Formatting / Display Utilities
# ---------------------------------------------------------
def format_report(results):
    lines = ["\n========== Zero-Shot Report ==========\n"]
    for name, info in results.items():
        lines.append(f"\nDataset: {name}")
        lines.append(f"  Prompt Used     : {info['prompt']}")
        lines.append(f"  Box Threshold   : {info['box_threshold']}")
        lines.append(f"  Text Threshold  : {info['text_threshold']}")
        lines.append(f"  Mean AP         : {info['ap']:.4f}")
    lines.append("\n======================================\n")
    return "\n".join(lines)
