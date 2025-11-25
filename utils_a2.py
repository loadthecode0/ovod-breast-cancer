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
from groundingdino.util.inference import annotate
import matplotlib.pyplot as plt
import supervision as sv
import numpy as np
from torchvision.ops import box_convert

# ---------------------------------------------------------
# DEFAULT ARGUMENTS (shared across all scripts)
# ---------------------------------------------------------
DEFAULT_PROMPTS = {
    "A": ["benign tissue . malignant tumor . dark background . dense tumor lump . no object ."],
    "B": ["benign tissue . malignant tumor . dark background . dense tumor lump . no object ."],
    # "B": ["background . dense malignant tumor lump ."],
    "C": ["benign tissue . malignant tumor . dark background . dense tumor lump . no object ."],
    # "A": ["tumor ."],
    # "B": ["tumor ."],
    # "C": ["tumor ."],
}

DEFAULT_THRESHOLDS = {
    "box_threshold": 0.15,
    "text_threshold": 0.1,
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

# def get_gt_boxes(df, image_name):
#     rows = df[df["image_name"] == image_name]
#     return rows[["xmin", "ymin", "xmax", "ymax"]].values

def get_gt_boxes(df, image_name):
    """
    Extract ground truth boxes for an image.
    Returns empty list for benign (no annotations).
    Returns list of [xmin, ymin, xmax, ymax] for malignant cases.
    """
    img_df = df[df["image_name"] == image_name]
    
    if len(img_df) == 0:
        return []
    
    boxes = []
    for _, row in img_df.iterrows():
        # Check if this row has valid box coordinates
        # (benign cases will have NaN/empty values)
        if pd.notna(row['xmin']) and pd.notna(row['ymin']) and \
           pd.notna(row['xmax']) and pd.notna(row['ymax']):
            try:
                xmin = float(row['xmin'])
                ymin = float(row['ymin'])
                xmax = float(row['xmax'])
                ymax = float(row['ymax'])
                
                # Validate box has positive area
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                else:
                    print(f"[WARNING] Invalid box for {image_name}: [{xmin}, {ymin}, {xmax}, {ymax}]")
            except (ValueError, TypeError) as e:
                print(f"[WARNING] Error parsing box for {image_name}: {e}")
    
    return boxes

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
        # print(f"max_iou : {max_iou:.8f}")
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

from torchvision.ops import box_iou
import torch
import numpy as np

def compute_phrase_aware_ap(
    pred_boxes,
    pred_phrases,
    gt_boxes,
    positive_phrases,
    iou_threshold=0.5,
    ignore_non_positive=True,
):
    """
    Phrase-aware AP for single-class detection (e.g., malignant tumors).

    Parameters:
    -----------
    pred_boxes : array-like (N, 4)
        Predicted boxes in xyxy pixel format.
    pred_phrases : list[str]
        List of N phrases returned by GroundingDINO.
    gt_boxes : array-like (M, 4)
        Ground truth boxes (xyxy).
    positive_phrases : list[str]
        Phrases that count as *positive* detections (malignant).
    iou_threshold : float
        IoU threshold for true positive.
    ignore_non_positive : bool
        If True: predictions whose phrase is NOT positive become ignored.
        If False: they count as false positives.

    Returns:
    --------
    AP = precision × recall
    """

    # Convert to numpy
    pred_boxes = np.array(pred_boxes)
    gt_boxes = np.array(gt_boxes)

    # Handle empty cases
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0:
        return 0.0

    # STEP 1: Filter predictions by phrases
    positive_indices = [
        i for i, phr in enumerate(pred_phrases)
        if phr.lower().strip() in positive_phrases
    ]
    # print(f"positive_indices : {positive_indices}")

    pred_pos_boxes = pred_boxes[positive_indices]

    if len(pred_pos_boxes) == 0:
        # No predicted malignant boxes → all GTs are unmet
        return 0.0

    # STEP 2: IoU matrix
    pred_t = torch.tensor(pred_pos_boxes, dtype=torch.float32)
    gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
    ious = box_iou(pred_t, gt_t).numpy()

    print(f"ious : {ious}")

    matched = set()
    tp, fp = 0, 0

    # STEP 3: For each predicted malignant box, match the best GT box
    for i in range(len(pred_pos_boxes)):
        iou_row = ious[i]
        max_iou = iou_row.max()
        best_gt = np.argmax(iou_row)

        if max_iou >= iou_threshold:
            if best_gt not in matched:
                matched.add(best_gt)
                tp += 1
            else:
                fp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision * recall


import numpy as np
import torch
from torchvision.ops import box_iou
import numpy as np
import torch
from torchvision.ops import box_iou

def compute_coco_style_ap(
    pred_boxes,
    pred_phrases,
    gt_boxes,
    logits,
    positive_phrases,
    pred_scores=None,
    ignore_non_positive=False,
    img_path=None,
):
    """
    Full COCO-style AP evaluator for a SINGLE CLASS (e.g., "malignant tumor").

    Includes:
    - AP@[0.50:0.95] averaged over IoU thresholds 0.50, 0.55, ..., 0.95
    - AP50 = IoU 0.50
    - AP75 = IoU 0.75
    - AP_small, AP_medium, AP_large based on GT box area

    Input format:
        pred_boxes: (N,4) xyxy predicted boxes
        pred_phrases: list[str] length N
        gt_boxes: (M,4) xyxy GT boxes
        positive_phrases: list[str] that count as positive predictions
        pred_scores: list/np.array length N (optional)
        ignore_non_positive: remove or treat as FP

    Output:
        dict containing:
            AP          (mAP@[.50:.95])
            AP50
            AP75
            AP_small
            AP_medium
            AP_large
    """

    

    print(f"len(gt_boxes) : {len(gt_boxes)}, {gt_boxes.shape}, {gt_boxes.size}, {gt_boxes}")
    print(f"len(pred_boxes) : {len(pred_boxes)}")

    print(f"gt_boxes: {gt_boxes}, {type(gt_boxes)}")
    # Handle empty GT
    if ((len(gt_boxes.tolist()) ==0) or (gt_boxes.size == 0)):
        return {
            "AP": 0.0,
            "AP-positive": 1.0 if (len(pred_boxes) == 0 or pred_boxes.size == 0) else 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
        }

    pred_boxes = np.array(pred_boxes)
    gt_boxes   = np.array(gt_boxes)

    # Default scores = 1.0
    if pred_scores is None:
        pred_scores = np.ones(len(pred_boxes), dtype=float)
    else:
        pred_scores = np.array(pred_scores)

    # PHRASE FILTERING

    

    # pos_idx = [
    #     i for i, phr in enumerate(pred_phrases) 
    #     if (phr.lower().strip() in positive_phrases or "tumor" in phr.lower().strip())
    # ]
    # print(f"pred_phrases : {pred_phrases}")

    pos_idx = []
    for i, phr in enumerate(pred_phrases):
        phrase_stripped = phr.lower().strip()
        # print(f"phrase_stripped : {phrase_stripped}")
        if "tumor" in phrase_stripped or "lump" in phrase_stripped:
            pos_idx.append(i)
        elif phrase_stripped == '' and logits[i] > 0.5:
            pos_idx.append(i)
            # print(f"positive phr : {phr}")

    # print(f"pos_idx : {pos_idx}")
    if ignore_non_positive:
        pred_boxes = pred_boxes[pos_idx]
        pred_scores = pred_scores[pos_idx]
    else:
        mask = np.zeros(len(pred_phrases), dtype=bool)
        mask[pos_idx] = True
        pred_scores = np.where(mask, pred_scores, 0.0)

    # If nothing left
    if len(pred_boxes) == 0 or pred_boxes.size == 0:
        return {
            "AP": 0.0,
            "AP-positive": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
        }

    # Sort predictions by confidence descending
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    # Compute IoU matrix once
    print(f"pred_boxes: {pred_boxes}")
    print(f"gt_boxes: {gt_boxes}")
    ious = box_iou(
        torch.tensor(pred_boxes, dtype=torch.float32),
        torch.tensor(gt_boxes, dtype=torch.float32)
    ).numpy()

    # COCO IoU thresholds 0.50–0.95
    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    ###########################################################################
    # Helper: Compute AP for ONE IoU threshold
    ###########################################################################
    def compute_ap_at_threshold(iou_thr):
        M = 1
        gt_matched = np.zeros(M, dtype=bool)
        tps, fps = [], []

        for i in range(len(pred_boxes)):
            row = ious[i]
            best_j = np.argmax(row)
            best_iou = row[best_j]

            if best_iou >= iou_thr and not gt_matched[best_j]:
                tps.append(1)
                fps.append(0)
                gt_matched[best_j] = True
            else:
                tps.append(0)
                fps.append(1)

        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)

        recalls = tp_cum / (len(gt_boxes) + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        # VOC/COCO interpolation
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))

        for i in range(len(mpre)-1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])

        ap = 0.0
        for i in range(1, len(mrec)):
            ap += (mrec[i] - mrec[i-1]) * mpre[i]

        return ap

    ###########################################################################
    # Compute AP across thresholds
    ###########################################################################
    ap_list = [compute_ap_at_threshold(t) for t in iou_thresholds]
    AP = float(np.mean(ap_list))
    AP50 = float(ap_list[0])
    AP75 = float(ap_list[int((0.75 - 0.5)/0.05)])  # which is index 5

    ###########################################################################
    # COCO defines small/medium/large by GT area
    ###########################################################################
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Use COCO thresholds (in pixel^2)
    small_thr = 32**2
    medium_thr = 96**2

    small_idx = np.where(gt_areas < small_thr)[0]
    medium_idx = np.where((gt_areas >= small_thr) & (gt_areas < medium_thr))[0]
    large_idx = np.where(gt_areas >= medium_thr)[0]

    def compute_ap_subset(indices):
        if len(indices) == 0:
            return 0.0

        subset_gt = gt_boxes[indices]
        subset_ious = box_iou(
            torch.tensor(pred_boxes, dtype=torch.float32),
            torch.tensor(subset_gt, dtype=torch.float32)
        ).numpy()

        ap_list_subset = []
        for thr in iou_thresholds:
            M = len(subset_gt)
            gt_matched = np.zeros(M, dtype=bool)
            tps, fps = [], []

            for i in range(len(pred_boxes)):
                row = subset_ious[i]
                best_j = np.argmax(row)
                best_iou = row[best_j]

                if best_iou >= thr and not gt_matched[best_j]:
                    tps.append(1)
                    fps.append(0)
                    gt_matched[best_j] = True
                else:
                    tps.append(0)
                    fps.append(1)

            tp_cum = np.cumsum(tps)
            fp_cum = np.cumsum(fps)

            recalls = tp_cum / (M + 1e-6)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

            # Interpolation
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([1.0], precisions, [0.0]))

            for i in range(len(mpre)-1, 0, -1):
                mpre[i-1] = max(mpre[i-1], mpre[i])

            ap = 0.0
            for i in range(1, len(mrec)):
                ap += (mrec[i] - mrec[i-1]) * mpre[i]
            ap_list_subset.append(ap)

        return float(np.mean(ap_list_subset))

    AP_small  = compute_ap_subset(small_idx)
    AP_medium = compute_ap_subset(medium_idx)
    AP_large  = compute_ap_subset(large_idx)

    return {
        "AP": AP,
        "AP-positive": AP,
        "AP50": AP50,
        "AP75": AP75,
        "AP_small": AP_small,
        "AP_medium": AP_medium,
        "AP_large": AP_large,
    }


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
        lines.append(f"  Mean AP         : {info['ap']:.8f}")
        lines.append(f"  Mean AP-positive : {info['ap_positive']:.8f}")
        lines.append(f"  Mean AP50 : {info['ap50']:.8f}")
        lines.append(f"  Mean AP75 : {info['ap75']:.8f}")
    lines.append("\n======================================\n")
    return "\n".join(lines)

def normalize_dataset_path(dataset_path):
    id = ""
    if "dataset_A" in dataset_path:
        id = "A"
    elif "dataset_B" in dataset_path:
        id = "B"
    elif "dataset_C" in dataset_path:
        id = "C"

    if "test" in dataset_path:
        id = id + "_test"
    elif "train" in dataset_path:
        id = id + "_train"
    return id

def visualize_predictions(img_name, dataset_path, image_source, gt_boxes, boxes, logits, phrases):
    try:
        print(f"img_name : {img_name}")
        save_dir = f"outputs/zero_shot/{normalize_dataset_path(dataset_path)}"
        ensure_dir(save_dir)

        # Prediction annotate
        annotated_pred = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )  # BGR

        # --- SAFE GT EXTRACTION ---
        # gt_boxes = get_gt_boxes(df, img_name)

        # Convert to numpy and validate shape
        gt_boxes = np.array(gt_boxes)

        # Case 1: No GT for this image → skip GT overlay
        if gt_boxes.size == 0:
            # Just save prediction-only visualization
            save_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_pred_only.jpg")
            plt.figure(figsize=(10, 10))
            plt.imshow(annotated_pred[:, :, ::-1])
            # plt.axis("off")
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            return

        # Case 2: Single box stored as shape (4,) → expand to (1,4)
        if gt_boxes.ndim == 1 and gt_boxes.shape[0] == 4:
            gt_boxes = gt_boxes[np.newaxis, :]

        # Case 3: Any bad shape → skip GT entirely
        if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
            print(f"[WARN] Bad GT box shape for {img_name}: {gt_boxes.shape}")
            return

        # --- Annotate GT ---
        gt_detections = sv.Detections(xyxy=gt_boxes)

        gt_annotator = sv.BoxAnnotator(
            color=sv.Color.from_hex("#38FAF4"), 
            thickness=16,
            text_thickness=6,
            text_scale=2.5,
        )
        annotated_full = gt_annotator.annotate(
            scene=annotated_pred.copy(),
            detections=gt_detections,
            labels=["GT (malignancy)"] * len(gt_boxes)
        )

        # Save
        save_path = os.path.join(
            save_dir, f"{os.path.splitext(img_name)[0]}_pred_gt.jpg"
        )

        # plt.figure(figsize=(10, 10))
        plt.figure(figsize=(12, 12))

        # Show image
        plt.imshow(annotated_full[:, :, ::-1])  # BGR -> RGB

        # Add coordinate axes
        # plt.gca().invert_yaxis()  # optional, to match image origin
        plt.gca().set_aspect('equal', adjustable='box')

        # Draw X/Y ticks every 500 pixels
        h, w = annotated_full.shape[:2]
        plt.xticks(np.arange(0, w, 500))
        plt.yticks(np.arange(0, h, 500))




        plt.grid(color='yellow', linestyle='--', linewidth=1, alpha=0.4)
        plt.scatter([0], [0], s=20, c='red')  # show origin point
        plt.text(0, 0, "(0,0)", color='red', fontsize=15)

        plt.xlabel("X coordinate (pixels)")
        plt.ylabel("Y coordinate (pixels)")
        plt.title(f"{img_name} — Zero Shot Evaluation)")

        save_path = os.path.join(
            save_dir,
            f"{os.path.splitext(img_name)[0]}_pred_gt.jpg"
        )

        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        # plt.imshow(annotated_full[:, :, ::-1])  # BGR→RGB
        # # plt.axis("off")
        # plt.savefig(save_path, bbox_inches="tight", dpi=150)
        # plt.close()

    except Exception as e:
        print(f"[WARN] Visualization (pred+GT) failed for {img_name}: {e}")

def filter_pos_pred_boxes(pred_boxes, logits, pred_phrases, positive_phrases=None):

    # pos_idx, neg_idx = [], []
    pos_pred_boxes = []
    pos_logits = []
    pos_phrases = []
    for i, phr in enumerate(pred_phrases):
        phrase_stripped = phr.lower().strip()
        if "tumor" in phrase_stripped or "lump" in phrase_stripped:
            pos_pred_boxes.append(pred_boxes[i])
            pos_logits.append(logits[i])
            pos_phrases.append(phr)
        # else:
        #     neg_idx.append(i)

    return pos_pred_boxes, pos_logits, pos_phrases