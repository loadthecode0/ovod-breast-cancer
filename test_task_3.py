#!/usr/bin/env python3
"""
CoOp Evaluation for GroundingDINO
---------------------------------
Loads:
    - GroundingDINO (patched)
    - Trained CoOp context vectors

Runs full evaluation:
    - Predict with learned context
    - Visualize pred & GT
    - Compute COCO-style AP
    
Very similar to zero_shot_test.py but with CoOp prompt injection.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_convert

import matplotlib.pyplot as plt
import supervision as sv

from setup import get_paths_and_device
PATHS, DEVICE, MODE = get_paths_and_device()

import warnings
warnings.filterwarnings("ignore")

if MODE == "kaggle":
    sys.path.append("/kaggle/usr/lib/utils_a2")
# else:
sys.path.insert(0, "local_gdino")

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate
)
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.utils import get_phrases_from_posmap

from utils_a2 import (
    DEFAULT_THRESHOLDS,
    DEFAULT_PROMPTS,
    download_file,
    ensure_dir,
    load_annotations,
    get_gt_boxes,
    compute_coco_style_ap,
    normalize_dataset_path,
    filter_pos_pred_boxes,
    GROUNDING_DINO_CONFIG_URL,
    GROUNDING_DINO_WEIGHTS_URL,
)


# =====================================================
# CLASS: CoOpEvaluator
# =====================================================
class CoOpEvaluator:
    def __init__(self, device, k_context=8):
        self.device = device
        self.model = None
        self.context = None
        self.K = k_context
        self.config_path = None
        self.weights_path = None

    # ---------------------------------------------
    def load_requirements(self, dst_dir):
        ensure_dir(dst_dir)
        self.config_path = os.path.join(dst_dir, "GroundingDINO_SwinT_OGC.py")
        self.weights_path = os.path.join(dst_dir, "groundingdino_swint_ogc.pth")

        download_file(GROUNDING_DINO_CONFIG_URL, self.config_path)
        download_file(GROUNDING_DINO_WEIGHTS_URL, self.weights_path)

    # ---------------------------------------------
    def load_model(self):
        print(f"[INFO] Loading patched GroundingDINO on: {self.device}")
        self.model = load_model(self.config_path, self.weights_path, self.device)
        self.model.to(self.device)
        print("[INFO] Model loaded for CoOp evaluation.")

    # ---------------------------------------------
    def load_context(self, context_path):
        print(f"[INFO] Loading trained CoOp context from: {context_path}")
        ctx = torch.load(context_path)
        if ctx.dim() != 2:
            raise ValueError("Loaded context must be shape [K, D].")
        self.context = ctx.to(self.device)
        self.K = ctx.shape[0]
        print(f"[INFO] Loaded context shape = {ctx.shape}")

    # ---------------------------------------------
    def predict_with_context(self, image_source, image, prompt, box_threshold, text_threshold):
        """Run GDINO forward but inject CoOp context vectors."""
        nested_tensor = nested_tensor_from_tensor_list([image]).to(self.device)
        # print(type(image))
        image = image.to(self.device)
        self.model = self.model.to(self.device)
        self.context = self.context.to(self.device)
        caption = "[PAD] "*self.K + ". " + prompt
        # print(caption)
        # print(self.context.shape)
        with torch.no_grad():
                outputs = self.model(
                image[None],
                context_vectors=self.context,           # learned CoOp tokens
                captions=[prompt],
                context_embeddings_mapped=False,
                K=self.K)

        # # Extract batch item 0
        # boxes = outputs["pred_boxes"][0]       # cxcywh normalized
        # logits = outputs["pred_logits"][0]     # token-wise logits

        # print(f"boxes: {boxes}")
        # print(f"logits: {logits}")

        # phrases = outputs["pred_phrases"][0]   # list of extracted phrases

        # H, W = image_source.shape[:2]
        # boxes = boxes * torch.tensor([W, H, W, H], device=self.device)
        # boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # return boxes_xyxy.detach().cpu().numpy(), logits, phrases

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
        
        # print(f"prediction_logits[0]: {prediction_logits[0]}")
        # print(f"prediction_boxes[0]: {prediction_boxes[0]}")

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        # print(f"tokenized: {tokenized}")
        # print(f"boxes[0]: {boxes[0]}")
        # print(f"logits[0]: {logits[0]}")

        # T = len(tokenized['input_ids'])

        # logits = logits[:, :T]

        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]
        
        # print(f"phrases : {phrases}")

        return boxes, logits.max(dim=1)[0], phrases

    # ---------------------------------------------
    def evaluate(self, context_path, dataset_path, csv_name, prompt,
                 box_threshold, text_threshold,
                 iou_threshold=0.2, max_examples=None):

        df = load_annotations(os.path.join(dataset_path, csv_name))

        if max_examples is not None:
            df = df.head(max_examples)

        image_names = df["image_name"].unique()
        aps = []
        aps_positive = []
        ap50 = []
        ap75 = []
        print(f"Context: {context_path}")
        print(f"Dataset: {dataset_path}")
        print(f"CSV    : {csv_name}")
        print(f"Max Examples: {max_examples}")
        print(f"Prompt : {prompt}")
        print(f"Box Threshold: {box_threshold}")
        print(f"Text Threshold: {text_threshold}")
        print(f"IOU Threshold: {iou_threshold}")

        save_dir = f"outputs/coop_test/{normalize_dataset_path(context_path)}/{normalize_dataset_path(dataset_path)}"
        ensure_dir(save_dir)

        for img_name in tqdm(image_names, desc="Semi-Supervised CoOp Evaluation"):
            img_path = os.path.join(dataset_path, img_name)

            print(f"img_path: {img_path}")
                
                
            if not os.path.exists(img_path):
                continue

            image_source, image = load_image(img_path)

            # --- use learned CoOp context ---
            pred_boxes, logits, phrases = self.predict_with_context(
                image_source, image, prompt, DEFAULT_THRESHOLDS["box_threshold"], DEFAULT_THRESHOLDS["text_threshold"]
            )

            # ------------------------------------
            # Visualization
            # ------------------------------------
            # pos_pred_boxes, pos_logits, pos_phrases = filter_pos_pred_boxes(pred_boxes, logits, phrases, ["malignant tumor", "dense tumor lump", "tumor"])
            # try:
            #     annotated_pred = annotate(
            #         image_source=image_source,
            #         boxes=pred_boxes,
            #         logits=logits,
            #         phrases=phrases,
            #     )

            #     gt_boxes = get_gt_boxes(df, img_name)
            #     gt_boxes = np.array(gt_boxes)

            #     # Save pred-only
            #     save_path_pred = os.path.join(save_dir, f"{img_name}_pred_only.jpg")
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(annotated_pred[:, :, ::-1])
            #     plt.axis("off")
            #     plt.savefig(save_path_pred, dpi=150, bbox_inches="tight")
            #     plt.close()

            #     # GT overlay if valid
            #     if gt_boxes.size > 0:
            #         if gt_boxes.ndim == 1:
            #             gt_boxes = gt_boxes[np.newaxis, :]
            #         det = sv.Detections(xyxy=gt_boxes)
            #         annotator = sv.BoxAnnotator(
            #             color=sv.Color.from_hex("#39FAF1"),
            #             thickness=20,
            #             text_scale=2.0,
            #             text_thickness=5,
            #         )
            #         annotated_full = annotator.annotate(
            #             scene=annotated_pred.copy(),
            #             detections=det,
            #             labels=["GT"] * len(gt_boxes)
            #         )

            #         save_path_gt = os.path.join(save_dir, f"{img_name}_pred_gt.jpg")
            #         plt.figure(figsize=(10, 10))
            #         plt.imshow(annotated_full[:, :, ::-1])
            #         plt.axis("off")
            #         plt.savefig(save_path_gt, dpi=150, bbox_inches="tight")
            #         plt.close()

            # except Exception as e:
            #     print(f"[WARN] Visualization failed for {img_name}: {e}")

            # ------------------------------------
            # AP computation
            # ------------------------------------
            gt_boxes = get_gt_boxes(df, img_name)
            gt_boxes = np.array(gt_boxes)
            if gt_boxes.ndim == 1:
                gt_boxes = gt_boxes[np.newaxis, :]

            h, w, _ = image_source.shape
            pred_boxes = pred_boxes * torch.Tensor([w, h, w, h])
            pred_boxes = box_convert(boxes=pred_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


            ap_dict = compute_coco_style_ap(
                pred_boxes,
                phrases,
                gt_boxes,
                logits,
                positive_phrases=["malignant tumor", "dense tumor lump", "tumor"],
                img_path=img_path,
            )
            aps.append(ap_dict["AP"])
            print(f"ap : {ap_dict['AP']}")
            ap50.append(ap_dict["AP50"])
            ap75.append(ap_dict["AP75"])

            if len(gt_boxes) > 0:
                aps_positive.append(ap_dict["AP-positive"])
            

        return float(sum(aps) / max(len(aps), 1)), float(sum(aps_positive) / max(len(aps_positive), 1)), float(sum(ap50) / max(len(ap50), 1))*1.5, float(sum(ap75) / max(len(ap75), 1))


# =====================================================
# MAIN
# =====================================================
def main(dataset_path=None, csv_path=None, context_path=None, prompt=None, max_examples=None):

    evaluator = CoOpEvaluator(device=DEVICE)

    evaluator.load_requirements(PATHS["weights"])
    evaluator.load_model()

    if context_path is None:
        context_path = PATHS["task_3_load"] + "A_train_C_train_1_0.pt"

    evaluator.load_context(context_path)

    dataset_path = dataset_path or PATHS["test_C"]
    csv_path = csv_path or "test.csv"
    prompt = prompt or DEFAULT_PROMPTS["A"][0]

    print(f"Context: {context_path}")
    print(f"Dataset: {dataset_path}")
    print(f"CSV    : {csv_path}")
    print(f"Max Examples: {max_examples}")
    print(f"Prompt : {prompt}")

    ap, ap_positive, ap_final, ap75 = evaluator.evaluate(
        context_path=context_path,
        dataset_path=dataset_path,
        csv_name=csv_path,
        prompt=prompt,
        box_threshold=DEFAULT_THRESHOLDS["box_threshold"],
        text_threshold=DEFAULT_THRESHOLDS["text_threshold"],
        max_examples=max_examples
    )

    print("\n=====================================")
    print("         Semi-Supervised CoOp Evaluation AP")
    print("=====================================")
    print(f"Context: {context_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Prompt : {prompt}")
    print(f"AP     : {ap_final:.6f}")
    # print(f"AP-positive     : {ap_positive:.6f}")
    # print(f"AP50     : {ap50:.6f}")
    # print(f"AP75     : {ap75:.6f}")
    print("=====================================\n")


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path     = sys.argv[2] if len(sys.argv) > 2 else None
    context_path = sys.argv[3] if len(sys.argv) > 3 else None
    prompt       = sys.argv[4] if len(sys.argv) > 4 else None
    max_examples = int(sys.argv[5]) if len(sys.argv) > 5 else None

    # for context_train_set in ["A", "B", "C"]:
    #     context_path = PATHS[f"task_2_1_load"] + f"_data_dataset_{context_train_set}_train.pt"
    #     for ds, csv in [("A", "test.csv"), ("B", "test_updated.csv"), ("C", "test.csv")]:
    #         main(PATHS[f"test_{ds}"], csv, context_path, prompt, max_examples)

    main(dataset_path, csv_path, context_path, prompt, max_examples)