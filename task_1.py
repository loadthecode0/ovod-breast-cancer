#!/usr/bin/env python3
"""
task_1.py
Zero-Shot Evaluation using GroundingDINO
Now with:
- Automatic device selection
- Automatic path selection
- Default = local Mac
"""

import os
import torch
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict
import sys

# === Load central configuration ===
from setup import get_paths_and_device
PATHS, DEVICE, MODE = get_paths_and_device()   # default = local

if MODE == "kaggle":

    sys.path.append("/kaggle/usr/lib/utils_a2")


# === Utils ===
from utils_a2 import (
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
    def __init__(self, device):
        self.device = device
        self.model = None
        self.config_path = None
        self.weights_path = None

    def download_requirements(self, dst_dir):
        ensure_dir(dst_dir)
        self.config_path = os.path.join(dst_dir, "GroundingDINO_SwinT_OGC.py")
        self.weights_path = os.path.join(dst_dir, "groundingdino_swint_ogc.pth")

        download_file(GROUNDING_DINO_CONFIG_URL, self.config_path)
        download_file(GROUNDING_DINO_WEIGHTS_URL, self.weights_path)

    def load_model(self):
        print(f"[INFO] Loading model on device: {self.device}")
        self.model = load_model(self.config_path, self.weights_path, self.device)
        print("[INFO] Model loaded.")

    def evaluate(self, dataset_path, csv_name, prompt,
                 box_threshold, text_threshold, iou_threshold=0.5, max_examples=100):

        df = load_annotations(os.path.join(dataset_path, csv_name))

        if max_examples is not None:
            df = df.sample(max_examples)

        image_names = df["image_name"].unique()
        aps = []

        for img_name in tqdm(image_names, desc=f"Evaluating {dataset_path}"):
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

    def generate_report(self, results):
        text = format_report(results)
        print(text)
        return text


# =====================================================
# MAIN
# =====================================================
def main(dataset_path=None, csv_path=None, prompt=None, max_examples=None):

    evaluator = GroundingDINOEvaluator(device=DEVICE)

    evaluator.download_requirements(PATHS["weights"])
    evaluator.load_model()

    if dataset_path is not None:
        dataset_paths = [dataset_path]
    else:   
        dataset_paths = [
            PATHS["test_A"],
            PATHS["test_B"],
            PATHS["test_C"]
        ]

    if csv_path is not None:
        annotation_files = [csv_path]
    else:
        annotation_files = [
            "test.csv",
            "test_updated.csv",
            "test.csv"
        ]

    if prompt is not None:
        prompts = [prompt]
    else:
        prompts = [
            DEFAULT_PROMPTS["A"][0],
            DEFAULT_PROMPTS["B"][0],
            DEFAULT_PROMPTS["C"][0],
        ]

    results = {}

    for i, dpath in enumerate(dataset_paths):
        ds_name = f"Dataset_{chr(ord('A') + i)}"
        ap = evaluator.evaluate(
            dataset_path=dpath,
            csv_name=annotation_files[i],
            prompt=prompts[i],
            box_threshold=DEFAULT_THRESHOLDS["box_threshold"],
            text_threshold=DEFAULT_THRESHOLDS["text_threshold"],
            max_examples=max_examples,
        )
        results[ds_name] = {
            "prompt": prompts[i],
            "box_threshold": DEFAULT_THRESHOLDS["box_threshold"],
            "text_threshold": DEFAULT_THRESHOLDS["text_threshold"],
            "ap": ap,
        }

    evaluator.generate_report(results)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = None

    if len(sys.argv) > 2:
        csv_path = sys.argv[2]
    else:
        csv_path = None

    if len(sys.argv) > 3:
        prompt = sys.argv[3]
    else:
        prompt = None

    if len(sys.argv) > 4:
        max_examples = int(sys.argv[4])
    else:
        max_examples = None

    main(dataset_path=dataset_path, csv_path=csv_path, prompt=prompt, max_examples=max_examples)


