#!/usr/bin/env python3
"""
train_task_2_1.py
Task 2: CoOp Prompt Learning for GroundingDINO

Implements:
- Learnable K context vectors
- Concatenation with frozen class token embeddings (from model.get_text_embeddings)
- Forward pass using model.forward(text_embeddings=...)
- AP-based loss (1 - AP)
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou

# --- your local import of patched GroundingDINO inference wrapper ---
# project setup
from setup import get_paths_and_device
PATHS, DEVICE, MODE = get_paths_and_device()
if MODE == "kaggle":
    sys.path.append("/kaggle/usr/lib/utils_a2")
elif MODE == "local":
    sys.path.insert(0, "local_gdino")   # <--- points to folder that contains a patched groundingdino/

from groundingdino.util.inference import load_image  




print(f"[INFO] GroundingDINO path: {sys.path}")

from utils_a2 import (
    DEFAULT_PROMPTS,
    DEFAULT_THRESHOLDS,
    ensure_dir,
    download_file,
    load_annotations,
    get_gt_boxes,
    compute_ap,
    GROUNDING_DINO_CONFIG_URL,
    GROUNDING_DINO_WEIGHTS_URL,
)


# ================================================================
#  CoOp MODULE (learnable context tokens)
# ================================================================
class CoOpPromptLearner(torch.nn.Module):
    def __init__(self, K: int, D: int, init_vec: torch.Tensor):
        """
        K: number of context tokens
        D: hidden size (BERT hidden)
        init_vec: initial embedding vector (D,)
        """
        super().__init__()
        init = init_vec.unsqueeze(0).repeat(K, 1)  # [K, D]
        self.context = torch.nn.Parameter(init)

    def forward(self):
        """Return learnable context tokens [K, D]."""
        return self.context


# ================================================================
#  TRAINER
# ================================================================
class CoOpTrainer:
    def __init__(self, device, k_context=8):
        self.device = device
        self.k = k_context
        self.model = None
        self.config_path = None
        self.weights_path = None

    # ---------------------------------------------
    def download_requirements(self, dst_dir):
        ensure_dir(dst_dir)
        self.config_path = os.path.join(dst_dir, "GroundingDINO_SwinT_OGC.py")
        self.weights_path = os.path.join(dst_dir, "groundingdino_swint_ogc.pth")
        download_file(GROUNDING_DINO_CONFIG_URL, self.config_path)
        download_file(GROUNDING_DINO_WEIGHTS_URL, self.weights_path)

    # ---------------------------------------------
    def load_model(self):
        from groundingdino.util.inference import load_model  # lazy import

        print(f"[INFO] Loading patched GroundingDINO on {self.device}")
        self.model = load_model(self.config_path, self.weights_path, self.device)
        self.model.to(self.device)

        # verify all params on same device
        bad = []
        for n, p in self.model.named_parameters():
            if p.device != torch.device(self.device):
                bad.append((n, p.device))
        if bad:
            print("[WARN] Some params NOT on target device:")
            for n, d in bad[:10]:
                print(" ", n, d)
            # decide fallback
        # freeze entire model for CoOp:
        for p in self.model.parameters():
            p.requires_grad = False

        print("[INFO] Model loaded and frozen.")

    # ---------------------------------------------
    def _init_coop(self, init_prompt: str):
        """
        Initialize context tokens by averaging BERT embeddings of the prompt.
        """
        with torch.no_grad():
            # BERT-space embeddings: [T, D_bert]
            class_token_embs = self.model.get_text_embeddings(init_prompt).to(self.device)
            mean_vec = class_token_embs.mean(dim=0)  # [D]
        learner = CoOpPromptLearner(self.k, mean_vec.shape[0], mean_vec).to(self.device)
        return learner, class_token_embs  # return frozen class tokens too

    # ---------------------------------------------
    def train(self, train_path, train_csv="train.csv",
              init_prompt="a malignant tumor", lr=1e-3, epochs=3,
              save_path="trained_coop_context.pt", num_examples=None):

        # ------------------------
        # Load dataset 
        # ------------------------
        df = load_annotations(os.path.join(train_path, train_csv))
        if num_examples is not None:
            df = df.sample(num_examples)
        image_names = df["image_name"].unique()
        print(f"[INFO] dataset size = {len(image_names)}")

        # ------------------------
        # Init CoOp learner
        # ------------------------
        learner, class_token_embs = self._init_coop(init_prompt)
        class_token_embs = class_token_embs.to(self.device)  # [T_c, D]

        optimizer = torch.optim.Adam([ learner.context ], lr=lr)

        print(f"[INFO] Starting CoOp training (K={self.k})")

        # ------------------------
        # Training Loop
        # ------------------------
        for ep in range(1, epochs + 1):
            losses = []
            np.random.shuffle(image_names)

            for img_name in tqdm(image_names, desc=f"Epoch {ep}"):

                img_path = os.path.join(train_path, img_name)
                if not os.path.exists(img_path):
                    continue

                # --- load image (GroundingDINO format) ---
                image_source, image = load_image(img_path)

                # --- Build CoOp prompt embedding ---
                context = learner()                               # [K, D]
                full_prompt = torch.cat([context, class_token_embs], dim=0)  # [K+T, D]
                full_prompt = full_prompt.unsqueeze(0).to(self.device)            # [1, K+T, D]

                # --- Forward pass using patched GDINO ---
                outputs = self.model(
                    image.unsqueeze(0).to(self.device),           # samples tensor
                    text_embeddings=full_prompt,
                    embeddings_mapped=False                       # map through feat_map
                )

                pred_boxes = outputs["pred_boxes"][0]   # [num_queries, 4]
                pred_logits = outputs["pred_logits"][0] # unused here, but okay

                # scale boxes back
                pred_boxes = pred_boxes * torch.tensor([
                    image_source.shape[1], image_source.shape[0],
                    image_source.shape[1], image_source.shape[0]
                ], device=self.device)

                # --- Ground truth ---
                gt_boxes = get_gt_boxes(df, img_name)

                # --- AP loss ---
                # ap = compute_ap(pred_boxes.detach().cpu().numpy(), gt_boxes)
                # loss_val = 1.0 - ap

                # --- Ensure gt_boxes is torch tensor on same device ---
            if len(gt_boxes) == 0:
                # No GT: encourage low scores (optional); here we push scores down
                pred_scores = outputs["pred_logits"][0].max(dim=1)[0]   # [num_preds]
                target = torch.zeros_like(pred_scores, device=self.device)
                loss = F.binary_cross_entropy_with_logits(pred_scores, target)
            else:
                # convert pred_boxes from cxcywh normalized -> xyxy absolute (or keep normalized, but ensure gt same scale)
                # pred_boxes are normalized in [0,1]; convert to xyxy normalized for IoU (both fine)
                # gt_boxes loaded from CSV (xmin,ymin,xmax,ymax) are absolute pixel coordinates.
                # convert pred_boxes to absolute coordinates to compare:
                h, w = image_source.shape[0], image_source.shape[1]
                # pred_boxes: cx,cy,w,h normalized -> xyxy absolute
                # convert to xyxy normalized? we'll convert gt to normalized to match pred_boxes
                pred_boxes_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")  # still normalized
                # Convert gt_boxes to normalized xyxy in [0,1]
                gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=self.device)  # [G,4] absolute
                # gt is xmin,ymin,xmax,ymax in pixels -> normalize
                gt_boxes_norm = torch.stack([
                    gt_boxes_t[:, 0] / w,
                    gt_boxes_t[:, 1] / h,
                    gt_boxes_t[:, 2] / w,
                    gt_boxes_t[:, 3] / h,
                ], dim=1)  # [G,4] normalized xyxy

                # pred_boxes_xyxy is normalized xyxy in tensor on device
                # compute IoU between pred and GT (both normalized xyxy)
                ious = box_iou(pred_boxes_xyxy, gt_boxes_norm)  # [num_preds, G]

                # For each pred, find max IoU w.r.t any GT
                max_ious, _ = ious.max(dim=1)  # [num_preds]

                iou_pos = 0.5  # positive IoU threshold, tuneable
                # label = 1 if max_iou >= iou_pos else 0
                labels = (max_ious >= iou_pos).float()  # [num_preds] on device

                pred_scores = outputs["pred_logits"][0].max(dim=1)[0]  # [num_preds] raw logits
                # Optionally downweight positives/negatives or sample negatives; simple BCE for now:
                loss = F.binary_cross_entropy_with_logits(pred_scores, labels)

                # loss = torch.tensor(loss_val, device=self.device, dtype=torch.float32)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            print(f"[Epoch {ep}] mean loss = {np.mean(losses):.4f}")

        # ------------------------
        # Save learned context
        # ------------------------
        ensure_dir(os.path.dirname(save_path) or ".")
        torch.save( learner.context.detach().cpu(), save_path )
        print(f"[INFO] Saved CoOp context to: {save_path}")

        return save_path


# ================================================================
# ENTRY POINT
# ================================================================
def main(dataset_path=None, csv_path=None, save_path=None, num_examples=None):
    trainer = CoOpTrainer(device=DEVICE, k_context=8)

    trainer.download_requirements(PATHS["weights"])
    trainer.load_model()

    train_path = dataset_path if dataset_path is not None else PATHS["train_A"]
    train_csv = csv_path if csv_path is not None else "train.csv"
    save_path = save_path if save_path is not None else PATHS["task_2_1_save"] + train_path.replace("/", "_").replace(".", "") + ".pt"

    trainer.train(
        train_path=train_path,
        train_csv=train_csv,
        init_prompt=DEFAULT_PROMPTS["A"][0],
        lr=1e-3,
        epochs=5,
        save_path=save_path,
        num_examples=num_examples
    )


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path     = sys.argv[2] if len(sys.argv) > 2 else None
    save_path    = sys.argv[3] if len(sys.argv) > 3 else None
    num_examples = int(sys.argv[4]) if len(sys.argv) > 4 else None

    main(dataset_path=dataset_path, csv_path=csv_path, save_path=save_path, num_examples=num_examples)
