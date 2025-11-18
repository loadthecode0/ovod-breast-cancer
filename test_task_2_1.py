#!/usr/bin/env python3
"""
test_task_2_1.py
Evaluation script for Task 2.1 (CoOp)

Loads:
- Patched GroundingDINO model
- Frozen class token embeddings
- Trained CoOp context vectors

Evaluates:
- AP on test dataset(s)
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Device / paths
from setup import get_paths_and_device
PATHS, DEVICE, MODE = get_paths_and_device()

# Use patched local GroundingDINO
if MODE == "kaggle":
    sys.path.append("/kaggle/usr/lib/utils_a2")
elif MODE == "local":
    sys.path.insert(0, "local_gdino")

from groundingdino.util.inference import load_image, load_model
from torchvision.ops import box_convert

from utils_a2 import (
    DEFAULT_PROMPTS,
    DEFAULT_THRESHOLDS,
    load_annotations,
    get_gt_boxes,
    compute_ap,
    ensure_dir,
    download_file,
    GROUNDING_DINO_CONFIG_URL,
    GROUNDING_DINO_WEIGHTS_URL,
)


# ======================================================
#  CO-OP PROMPT EVALUATOR
# ======================================================
class CoOpTester:
    def __init__(self, device, k_context=8):
        self.device = device
        self.k = k_context
        self.model = None
        self.config_path = None
        self.weights_path = None
        self.class_token_embs = None      # [T_c, D]
        self.learned_context = None       # [K, D]

    # ---------------------------------------------
    def download_requirements(self, dst_dir):
        ensure_dir(dst_dir)
        self.config_path  = os.path.join(dst_dir, "GroundingDINO_SwinT_OGC.py")
        self.weights_path = os.path.join(dst_dir, "groundingdino_swint_ogc.pth")
        download_file(GROUNDING_DINO_CONFIG_URL,  self.config_path)
        download_file(GROUNDING_DINO_WEIGHTS_URL, self.weights_path)

    # ---------------------------------------------
    def load_model(self):
        print(f"[INFO] Loading patched GroundingDINO on {self.device}")
        self.model = load_model(self.config_path, self.weights_path, self.device)
        self.model.to(self.device)

        # Freeze model
        for p in self.model.parameters():
            p.requires_grad = False

        print("[INFO] Model loaded and frozen.")

    # ---------------------------------------------
    def load_context_and_class_tokens(self, init_prompt, coop_path):
        """
        coop_path: path to trained K x D context vectors
        """
        print(f"[INFO] Loading trained CoOp context from {coop_path}")
        self.learned_context = torch.load(coop_path, map_location=self.device)   # [K,D]

        print("[INFO] Extracting frozen class token embeddings...")
        with torch.no_grad():
            self.class_token_embs = self.model.get_text_embeddings(init_prompt).to(self.device)

        print(f"[INFO] Loaded context shape: {self.learned_context.shape}")
        print(f"[INFO] Loaded class token embeddings: {self.class_token_embs.shape}")

    # ---------------------------------------------
    def build_prompt(self):
        """
        Build final prompt embedding: [1, K+T, D]
        """
        full = torch.cat([self.learned_context, self.class_token_embs], dim=0)
        return full.unsqueeze(0).to(self.device)

    # ---------------------------------------------
    def evaluate_dataset(self, dataset_path, csv_name="test.csv", num_examples=None):
        df = load_annotations(os.path.join(dataset_path, csv_name))
        if num_examples is not None:
            df = df.sample(num_examples)
        image_names = df["image_name"].unique()

        full_prompt = self.build_prompt()

        aps = []
        print(f"[INFO] Evaluating dataset: {dataset_path}")

        for img_name in tqdm(image_names):

            img_path = os.path.join(dataset_path, img_name)
            if not os.path.exists(img_path):
                continue

            image_source, image = load_image(img_path)

            with torch.no_grad():
                outputs = self.model(
                    image.unsqueeze(0).to(self.device),
                    text_embeddings=full_prompt,
                    embeddings_mapped=False,
                )

            pred_boxes = outputs["pred_boxes"][0]
            pred_boxes = pred_boxes * torch.tensor([
                image_source.shape[1], image_source.shape[0],
                image_source.shape[1], image_source.shape[0]
            ], device=self.device)

            gt_boxes = get_gt_boxes(df, img_name)

            ap = compute_ap(pred_boxes.cpu().numpy(), gt_boxes)
            aps.append(ap)

        mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0
        return mAP


# ======================================================
# ENTRY POINT
# ======================================================
def main(test_paths=None, coop_path=None, csv_names=None, num_examples=None):
    tester = CoOpTester(device=DEVICE, k_context=8)
    tester.download_requirements(PATHS["weights"])
    tester.load_model()

    if test_paths is None:
        test_paths = [PATHS["test_A"]]

    if coop_path is None:
        coop_path = PATHS["task_2_1_load"] + test_paths[0].replace("/", "_").replace(".", "").replace("test", "train") + ".pt"

    if csv_names is None:
        csv_names = ["test.csv"]    

    # Always use same class prompt as training
    tester.load_context_and_class_tokens(DEFAULT_PROMPTS["A"][0], coop_path)

    print("\n================ CoOp Evaluation ================\n")
    for i, dpath in enumerate(test_paths):
        name = f"Dataset_{chr(ord('A') + i)}"
        ap = tester.evaluate_dataset(dpath, csv_names[i], num_examples)
        print(f"{name}:   mAP = {ap:.4f}")
    print("\n=================================================\n")


if __name__ == "__main__":

    coop_path = sys.argv[1] if len(sys.argv) > 1 else None
    num_datasets = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    test_paths = sys.argv[3:3+num_datasets] if len(sys.argv) > 3+num_datasets else None
    csv_names = sys.argv[3+num_datasets:3+num_datasets*2] if len(sys.argv) > 3+num_datasets*2 else None
    num_examples = int(sys.argv[3+num_datasets]) if len(sys.argv) > 3+num_datasets else None

    main(test_paths=test_paths, coop_path=coop_path, csv_names=csv_names, num_examples=num_examples)
