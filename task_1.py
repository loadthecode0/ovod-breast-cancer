# %% [code]
#!/usr/bin/env python3

"""
task_1.py
Zero-Shot Evaluation for Assignment 2 using GroundingDINO
Supports:
- Different CSV filenames per dataset
- Single CSV reused for all datasets
- Default Kaggle dataset paths
"""

import os
import argparse
from tqdm import tqdm
import torch

_original_torch_load = torch.load

def torch_load_patched(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = torch_load_patched

from groundingdino.util.inference import load_model, load_image, predict

import sys
sys.path.append("/kaggle/usr/lib/utils_a2_py")

from utils_a2_py import (
    DEFAULT_PROMPTS,
    DEFAULT_THRESHOLDS,
    download_file,
    ensure_dir,
    load_annotations,
    get_gt_boxes,
    compute_ap,
    format_report,
    GROUNDING_DINO_CONFIG_URL,
    GROUNDING_DINO_WEIGHTS_URL,
)


# =====================================================
# CLASS: GroundingDINOEvaluator
# =====================================================

class GroundingDINOEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.config_path = None
        self.weights_path = None

    # -------------------------------------------------
    # (A) Download model files
    # -------------------------------------------------
    def download_requirements(self, dst_dir="weights"):
        ensure_dir(dst_dir)

        self.config_path = os.path.join(dst_dir, "GroundingDINO_SwinT_OGC.py")
        self.weights_path = os.path.join(dst_dir, "groundingdino_swint_ogc.pth")

        download_file(GROUNDING_DINO_CONFIG_URL, self.config_path)
        download_file(GROUNDING_DINO_WEIGHTS_URL, self.weights_path)

        print(self.config_path)
        print(self.weights_path)

    # -------------------------------------------------
    # (B) Load model
    # -------------------------------------------------
    def load_model(self):
        print("[INFO] Loading GroundingDINO...")
        self.model = load_model(self.config_path, self.weights_path, self.device)
        print("[INFO] Model loaded.")

    # -------------------------------------------------
    # (C) Evaluate one dataset
    # -------------------------------------------------
    def evaluate(self, dataset_path, csv_name, prompt,
                 box_threshold, text_threshold, iou_threshold=0.5):

        csv_path = os.path.join(dataset_path, csv_name)
        df = load_annotations(csv_path)

        image_names = df["image_name"].unique()
        aps = []

        for img_name in tqdm(image_names, desc=f"Evaluating ({dataset_path})"):

            img_path = os.path.join(dataset_path, img_name)
            if not os.path.exists(img_path):
                continue

            image_source, image = load_image(img_path)

            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )

            pred_boxes = boxes * torch.tensor([
                image_source.shape[1], image_source.shape[0],
                image_source.shape[1], image_source.shape[0]
            ])

            gt_boxes = get_gt_boxes(df, img_name)

            ap = compute_ap(pred_boxes.numpy(), gt_boxes, iou_threshold)
            aps.append(ap)

        return float(sum(aps) / max(len(aps), 1))

    # -------------------------------------------------
    # (D) Report
    # -------------------------------------------------
    def generate_report(self, results):
        text = format_report(results)
        print(text)
        return text


# =====================================================
# CLI
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Task 1 Zero-Shot Evaluation")

    parser.add_argument(
        "--dataset_paths",
        nargs="+",
        required=False,
        default=[
            "/kaggle/input/ovod-datasets/dataset_A/dataset_A/test",
            "/kaggle/input/ovod-datasets/dataset_B/dataset_B/test",
            "/kaggle/input/ovod-datasets/dataset_C/dataset_C/test"
        ],
        help="Paths to datasets (each dataset/test folder)."
    )

    parser.add_argument(
        "--wts_dir",
        nargs="+",
        required=False,
        default="/kaggle/working/weights/",
        help="Paths to datasets (each dataset/test folder)."
    )

    parser.add_argument(
        "--annotations",
        nargs="+",
        required=False,
        default=[
            "test.csv",
            "test_updated.csv",
            "test.csv"
        ],
        help="One CSV or one per dataset."
    )

    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="One prompt or one per dataset."
    )

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    evaluator = GroundingDINOEvaluator(device=args.device)

    # A: download
    # evaluator.download_requirements(args.wts_dir)

    # B: load model
    evaluator.load_model()

    # Annotation logic
    if len(args.annotations) == 1:
        annotations_list = [args.annotations[0]] * len(args.dataset_paths)
    elif len(args.annotations) == len(args.dataset_paths):
        annotations_list = args.annotations
    else:
        raise ValueError("Provide ONE CSV or ONE CSV PER dataset.")

    # Prompt logic
    if args.prompts and len(args.prompts) == len(args.dataset_paths):
        prompts = args.prompts
    elif args.prompts and len(args.prompts) == 1:
        prompts = [args.prompts[0]] * len(args.dataset_paths)
    else:
        prompts = [
            DEFAULT_PROMPTS.get(chr(ord("A") + i), ["a malignant tumor"])[0]
            for i in range(len(args.dataset_paths))
        ]

    # Evaluate each dataset
    results = {}

    for i, dpath in enumerate(args.dataset_paths):
        ds_name = f"Dataset_{chr(ord('A') + i)}"
        ap = evaluator.evaluate(
            dataset_path=dpath,
            csv_name=annotations_list[i],
            prompt=prompts[i],
            box_threshold=DEFAULT_THRESHOLDS["box_threshold"],
            text_threshold=DEFAULT_THRESHOLDS["text_threshold"],
        )

        results[ds_name] = {
            "prompt": prompts[i],
            "box_threshold": DEFAULT_THRESHOLDS["box_threshold"],
            "text_threshold": DEFAULT_THRESHOLDS["text_threshold"],
            "ap": ap,
        }

    evaluator.generate_report(results)


if __name__ == "__main__":
    main()
