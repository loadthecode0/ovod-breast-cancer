# --- your local import of patched GroundingDINO inference wrapper ---
# project setup
import sys
from setup import get_paths_and_device
PATHS, DEVICE, MODE = get_paths_and_device()
if MODE == "kaggle":
    sys.path.append("/kaggle/usr/lib/utils_a2")
elif MODE == "local":
    sys.path.insert(0, "local_gdino")   # <--- points to folder that contains a patched groundingdino/


import os

import torch
import numpy as np
from tqdm import tqdm



import torch.nn.functional as F
import torch
from torchvision.ops import box_convert, box_iou
import cv2
import matplotlib.pyplot as plt
import supervision as sv

print(torch.cuda.is_available())

import warnings
warnings.filterwarnings("ignore")

import random
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



from groundingdino.util.inference import load_image, predict, annotate
from groundingdino.util.misc import nested_tensor_from_tensor_list

print(f"[INFO] GroundingDINO path: {sys.path}")

from utils_a2 import (
    DEFAULT_PROMPTS,
    DEFAULT_THRESHOLDS,
    ensure_dir,
    download_file,
    load_annotations,
    get_gt_boxes,
    compute_ap,
    compute_coco_style_ap,
    GROUNDING_DINO_CONFIG_URL,
    GROUNDING_DINO_WEIGHTS_URL,
    normalize_dataset_path,
)


def split_labeled_unlabeled(df, label_ratio=0.1, seed=42):
    np.random.seed(seed)
    all_images = df["image_name"].unique()
    np.random.shuffle(all_images)
    k = int(len(all_images) * label_ratio)
    labeled = all_images[:k]
    unlabeled = all_images[k:]
    return labeled, unlabeled


import torchvision.transforms as T

weak_aug = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.Resize((512, 512)),
    T.ToTensor(),
])

strong_aug = T.Compose([
    T.ToPILImage(),

    # Geometric (mild)
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=7, fill=0),    # medical-safe small rotations

    # Photometric (safe)
    T.RandomApply([
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ], p=0.8),

    # Blur (very common in medical SSL)
    T.RandomApply([
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
    ], p=0.3),

    # Gamma correction (models scanner exposure variations)
    T.RandomApply([
        T.Lambda(lambda img: T.functional.adjust_gamma(img, gamma=np.random.uniform(0.7, 1.3)))
    ], p=0.3),

    T.Resize((512, 512)),
    T.ToTensor(),
])



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
        self.beta = 0.2

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
        # with torch.no_grad():
            # BERT-space embeddings: [T, D_bert]
        class_token_embs = self.model.get_text_embeddings(init_prompt).to(self.device)
        mean_vec = class_token_embs.mean(dim=0)  # [D]
        learner = CoOpPromptLearner(self.k, mean_vec.shape[0], mean_vec).to(self.device)
        return learner, class_token_embs  # return frozen class tokens too


    def _compute_detection_loss(
        self,
        pred_boxes,             # [N,4] normalized cxcywh
        pred_logits,            # [N, T_total] token logits (sparse)
        gt_boxes,               # list of [x1,y1,x2,y2] abs coords
        image_shape,            # (H, W)
        # encoded_text,           # [T_total, D] text embeddings
        input_ids,              # [T_total] token ids (passed in from model)
        positive_phrases_ids,   # set/list of token ids that represent positive phrases
        image,
        caption,
        boxes,
        image_source
    ):
        """
        CoOp-Friendly Phrase-Grounding Loss for Training GroundingDINO

        Components:
        -----------
        1. Dynamic positive-token extraction from input_ids
        2. Phrase contrastive loss (text–query alignment)
        3. Soft box-matching loss using distance-based weights
        """
        # print(f"image shape: {image.shape}")
        H, W = image_shape
        # print(f"H: {H}, W: {W} inside compute_detection_loss")
        N = pred_boxes.shape[0]
        # print(f"pred_boxes: {pred_boxes}") #xyxy
        device = pred_boxes.device
        # print(f"gt_boxes: {gt_boxes}")


        # ============================================================
        # 1. Dynamic identification of positive tokens
        # ============================================================
        # Convert input_ids to list for indexing
        token_ids = input_ids.tolist()
        T = len(token_ids)
        self.T = T

        # print(f"{pred_logits[:, :len(token_ids)][0]}") # non inf
        temperature =1.0
        pred_logits = pred_logits[:, :T]
        # print(f"pred_logits shape: {pred_logits.shape}")
        pred_logits = pred_logits / temperature #make distribution sharper

        # Find indices where the token ID belongs to positive phrases
        # positive_phrases_ids: e.g. {16007, 27881, 13656, 9742, 15116}
        positive_positions = [
            idx for idx, tok in enumerate(token_ids)
            if tok in positive_phrases_ids
        ]

        # non-positive tokens
        non_positive_positions = [
            i for i in range(T) if i not in positive_positions
        ]

        # print(f"positive_positions : {positive_positions}")

        if len(positive_positions) == 0:
            raise ValueError("[ERROR] No positive tokens found in input_ids. Check token IDs.")

        # Extract logits for those token positions: [N, num_positive_tokens]
        positive_logits = pred_logits[:, positive_positions]
        nonpos_logits  = pred_logits[:, non_positive_positions]         # [N, N-P]

        # Average over the token dimension → phrase score per query
        phrase_scores = positive_logits.mean(dim=1)                     # [N]
        nonpos_scores = nonpos_logits.mean(dim=1)    
        neg_logit = nonpos_scores.mean()                       # [1]
        avg_tumor_logit = phrase_scores.mean()
        # print(f"positive_logits shape: {positive_logits.shape}")

        

        # print(f"phrase_scores shape: {phrase_scores.shape}")
        # if np.random.rand() < 0.1:
        #     print(f"phrase_scores : {phrase_scores}")

        if len(gt_boxes) == 0:
            # No tumor → all scores should be negative
            # print(f"debug no GT - {pred_logits[:, :len(token_ids)][0]}")
            print(f"no GT:avg_tumor_logit: {avg_tumor_logit}, neg_logit: {neg_logit}")
            targets_pos = torch.zeros_like(positive_logits)
            # targets_neg = torch.ones_like(nonpos_scores)
            cls_loss = 20.0*F.binary_cross_entropy_with_logits(positive_logits, targets_pos) 
            print(f"no GT: cls_loss : {cls_loss}")
            return cls_loss, {
                "ap": 0.0,
                "num_gt": len(gt_boxes),
                "num_pos": targets_pos.sum().item(),
                "cls_loss": cls_loss.item(),
                "box_loss": 0.0,
                "pos_logit": 0.0,
                "neg_logit": float(neg_logit),
                "avg_tumor_logit": float(avg_tumor_logit),
            }




        # ======================================================
        # 3. Compute closest query to the single GT box
        # ======================================================
        # print(f"debug with positive - {pred_logits[:, :len(token_ids)][0]}")
        # print(f"with GT: avg_tumor_logit: {avg_tumor_logit}")
        # -------- GT: xyxy pixel → normalized --------
        gt_xyxy = torch.tensor(gt_boxes[0], dtype=torch.float32, device=device)  # [4]

        gt_norm = gt_xyxy.clone()
        gt_norm[[0,2]] /= W      # normalize X
        gt_norm[[1,3]] /= H      # normalize Y

        # GT center in normalized coords
        gt_cxcy = box_convert(gt_norm.unsqueeze(0), "xyxy", "cxcywh")[0, :2]   # [2]

        # print(f"gt_cxcy: {gt_cxcy}")
        # -------- PRED: xyxy pixel → normalized --------
        pred_norm = pred_boxes.clone()
        pred_norm[:, [0,2]] /= W
        pred_norm[:, [1,3]] /= H

        # Pred centers (normalized)
        pred_centers = box_convert(pred_norm, "xyxy", "cxcywh")[:, :2]

        # print(f"pred_centers[0]: {pred_centers[0]}")

        # -------- Compute distance in normalized space --------
        # print((pred_centers - gt_cxcy).shape)
        dists = torch.norm(pred_centers - gt_cxcy, dim=1)   # [N]

        # Index of closest query
        best_q = torch.argmin(dists)
        # print(f"best_q: {best_q}")


        # Compute IoU between all pred boxes and GT box
        # print(f"pred_norm.shape: {pred_norm.shape}")
        # print(f"gt_norm.shape: {gt_norm.shape}")
        ious = box_iou(
            pred_norm,         # [N,4]
            gt_norm.unsqueeze(0)            # [1,4]
        ).squeeze(1)            # [N]

        # IoU threshold for being considered "positive"
        pos_indices = torch.nonzero(ious >= 0.1, as_tuple=True)[0]  # [K] queries

        if len(pos_indices) == 0:
            pos_indices = best_q

        # print(f"best_q: {best_q}")
        # print(f"pos_indices: {pos_indices}")
        # ======================================================
        # 4. Classification targets
        # ======================================================
        targets = torch.zeros_like(phrase_scores)
        targets[pos_indices] = 1.0                    # Only the closest query is positive

        cls_pos_loss = F.binary_cross_entropy_with_logits(phrase_scores, targets)
        # negative supervision: suppress non-positive tokens for ALL queries
        cls_neg_loss = F.binary_cross_entropy_with_logits(nonpos_scores, 
                                                          torch.zeros_like(nonpos_scores))
        # contrastive margin: enforce pos > non-pos
        margin = 2.0
        pos_logit = phrase_scores[pos_indices].mean()
        
        print(f"with GT: avg_tumor_logit: {avg_tumor_logit:.4f}, pos_logit: {pos_logit:.4f}, neg_logit: {neg_logit:.4f}")
        contrastive_loss = F.relu(margin - (pos_logit - neg_logit))
        # print(f"contrastive_loss: {contrastive_loss}")
        # ======================================================
        # 5. Box regression for the positive query only
        # ======================================================
        pred_norm_best = pred_norm[best_q].unsqueeze(0)                # [1,4]
        # print(f"pred_norm_best: {pred_norm_best}")
        # gt_box_best = gt_norm.unsqueeze(0), "xyxy", "cxcywh")  # [1,4]
        # print(f"gt_norm: {gt_norm}")
        box_loss = F.l1_loss(pred_norm_best, gt_norm)

        # ======================================================
        # 6. Total loss
        # ======================================================
        print(f"cls_pos_loss: {cls_pos_loss:.4f}, cls_neg_loss: {cls_neg_loss:.4f}, contrastive_loss: {contrastive_loss:.4f}, box_loss: {box_loss:.4f}")
        cls_loss = 50.0 * cls_pos_loss + 5.0 * cls_neg_loss + 5.0 * contrastive_loss
        total_loss = 10.0*(cls_loss + 5.0 * box_loss)



        return total_loss, {
            "ap": 0.0,
            "num_gt": len(gt_boxes),
            "num_pos": targets.sum().item(),
            "cls_loss": cls_loss.item(),
            "box_loss": box_loss.item(),
        }


    # ---------------------------------------------
    def train(self, labeled_path, labeled_csv, unlabeled_path, unlabeled_csv,
              init_prompt="a malignant tumor", lr=1e-3, epochs=3,
              save_path="trained_coop_context.pt", num_examples=None, batch_size=8, lambda_u=1.0):

        tau = 0.25          # pseudo-label confidence threshold
        # lambda_u = 1.0     # unsupervised weight
        warmup_epochs = 5   # λu = 0 for first few epochs

        # ------------------------
        # Load dataset 
        # ------------------------
        # df = load_annotations(os.path.join(labeled_path, labeled_csv))
        # if num_examples is not None:
        #     df = df.sample(num_examples)
        
        # labeled_imgs, unlabeled_imgs = split_labeled_unlabeled(df, label_ratio=0.1)
        df_l = load_annotations(os.path.join(labeled_path, labeled_csv))
        if num_examples is not None:
            df_l = df_l.sample(num_examples)
        df_u = load_annotations(os.path.join(unlabeled_path, unlabeled_csv))

        labeled_imgs = df_l["image_name"].unique()
        unlabeled_imgs = df_u["image_name"].unique()
        np.random.shuffle(unlabeled_imgs)
        unlabeled_imgs = unlabeled_imgs[:int(0.1 * len(unlabeled_imgs))]


        # ------------------------
        # Init CoOp learner
        # ------------------------
        learner, class_token_embs = self._init_coop(init_prompt)
        class_token_embs = class_token_embs.to(self.device)  # [T_c, D]

        optimizer = torch.optim.Adam([ learner.context ], lr=lr)

        print(f"[INFO] Starting CoOp training (K={self.k})")
        # print(f"[INFO] Context shape: {learner.context.shape}")

        # VERIFY feat_map is trainable
        print("[INFO] Checking gradient flow setup...")
        print(f"  Context requires_grad: {learner.context.requires_grad}")
        print(f"  feat_map requires_grad: {self.model.feat_map.weight.requires_grad}")
        # ------------------------
        # Training Loop
        # ------------------------

        # Training Loop
        all_epoch_history = []
        for ep in range(1, epochs + 1):
            epoch_losses = []
            epoch_aps = []
            epoch_metrics = {
                'cls_loss': [],
                'box_loss': [],
                'num_pos': [],
                'pos_logit': [],
                'neg_logit': [],
                'avg_tumor_logit': [],
                'grad_norm': [],
            }
            
            np.random.shuffle(labeled_imgs)
            
            # Create batches
            num_batches = (len(labeled_imgs) + batch_size - 1) // batch_size
            pbar = tqdm(range(num_batches), desc=f"Epoch {ep}/{epochs}")

            for batch_idx in pbar:
                # Get batch of image names
                
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(labeled_imgs))
                batch_image_names = labeled_imgs[start_idx:end_idx]
                current_batch_size = len(batch_image_names)
                # Load batch of images
                batch_images = []
                batch_image_sources = []
                batch_gt_boxes = []
                batch_image_shapes = []
                valid_indices = []
                
                for i, img_name in enumerate(batch_image_names):
                    img_path = os.path.join(labeled_path, img_name)
                    if not os.path.exists(img_path):
                        continue
                    
                    try:
                        image_source, image = load_image(img_path)
                        gt_boxes = get_gt_boxes(df_l, img_name)
                        
                        batch_images.append(image)
                        batch_image_sources.append(image_source)
                        batch_gt_boxes.append(gt_boxes)
                        batch_image_shapes.append((image_source.shape[0], image_source.shape[1]))
                        valid_indices.append(i)
                    except Exception as e:
                        print(f"[WARNING] Error loading {img_name}: {e}")
                        continue
                
                if len(batch_images) == 0:
                    continue

                # Stack images into batch tensor
                # images_tensor = torch.stack(batch_images).to(self.device)  # [B, C, H, W]
                images_nested_tensor = nested_tensor_from_tensor_list(batch_images).to(self.device)
                current_batch_size = len(batch_images)
                
                # Build CoOp prompt (same for all images in batch)
                context = learner()  # [K, D]
                # full_prompt = torch.cat([context, class_token_embs], dim=0)  # [K+T, D]
                # full_prompt = full_prompt.unsqueeze(0).repeat(current_batch_size, 1, 1)  # [B, K+T, D]
                # full_prompt = full_prompt.to(self.device)

                # full_class = class_token_embs.mean(dim=0, keepdim=True)  # collapse to single class embedding

                
                # Forward pass (batched)
                # outputs = self.model(
                #     images_nested_tensor,
                #     text_embeddings=full_prompt,
                #     embeddings_mapped=False
                # )
                # print("context size:", context.shape)
                captions = [init_prompt] * current_batch_size
                # print("captions size:", len(captions))
                # pass only context vectors (they get prepended)
                outputs = self.model(images_nested_tensor,
                                context_vectors=context,            # [K, D_bert] or [B, K, D_bert]
                                captions=captions,
                                # text_embeddings=full_class.unsqueeze(0).repeat(batch_size,1,1),         # optional: class token in BERT space
                                # embeddings_mapped=False,            # class token is BERT-space (we'll map)
                                context_embeddings_mapped=False,    # context is BERT-space
                                K=self.k
                            )
                
                # Get predictions for entire batch
                pred_boxes_batch = outputs["pred_boxes"]    # [B, N, 4]
                pred_logits_batch = outputs["pred_logits"]  # [B, N, K+T]
                # encoded_text_batch = outputs["encoded_text"]  # [B, K+T, D_bert]
                input_ids_batch = outputs["input_ids"]  # [B, K+T]
                # Compute loss for each item in batch
                batch_loss = 0.0
                batch_metrics_accum = {
                    'ap': [],
                    'num_pos': [],
                    'cls_loss': [],
                    'box_loss': [],
                    'raw': []
                }
                
                for b in range(current_batch_size): # loss calculated one by one
                    pred_boxes = pred_boxes_batch[b] 
                    # print(f"image name: {batch_image_names[b]}")   # [N, 4]
                    # print(f"image shape: {batch_images[b].shape}")
                    # print(f"pred_boxes: {pred_boxes}")

                    # # _, h, w = batch_images[b].shape
                    
                    # print(f"preprocessed image shape: {batch_images[b].shape[1:3]}")
                    # print(f"actual image shape: {batch_image_shapes[b]}")
                    h, w = batch_image_shapes[b]
                    boxes = pred_boxes * torch.Tensor([w, h, w, h]).to(self.device)
                    # print(f"boxes: {boxes}")
                    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                    # print(f"xyxy : {xyxy}")
                    pred_logits = pred_logits_batch[b]  # [N, K+T]
                    # encoded_text = encoded_text_batch[b]  # [N, K+T, D_bert]
                    input_ids = input_ids_batch[b]  # [N, K+T]
                    gt_boxes = batch_gt_boxes[b]
                    image_shape = batch_image_shapes[b]

                    
                    # print("class_token_embs size:", class_token_embs.shape)
                    # print("full_prompt size:", full_prompt.shape)
                    # print("pred_logits shape:", pred_logits.shape)
                    
                    # Compute loss for this sample
                    print(f"b : {b}")
                    loss, metrics = self._compute_detection_loss(
                        xyxy,
                        pred_logits,
                        gt_boxes,
                        image_shape,
                        # encoded_text,
                        input_ids,
                        [16007, 27881, 13656, 9742, 15116],
                        batch_images[b],
                        "[PAD] "*self.k + ". " + captions[b],
                        boxes,
                        batch_image_sources[b]
                    )

                    batch_loss += loss

                # ============================
                #  UNLABELED FIXMATCH PART
                # ============================
                # print(f"[INFO] Computing unsupervised loss for {img_name}")
                unsup_loss_total = 0.0
                num_unsup = 0

                if ep >= warmup_epochs:
                    # take ~same batch size from unlabeled pool
                    ul_batch = unlabeled_imgs[batch_idx*batch_size : (batch_idx+1)*batch_size]

                    for img_name in ul_batch:
                        img_path = os.path.join(unlabeled_path, img_name)
                        if not os.path.exists(img_path):
                            continue

                        # weak + strong versions
                        _, img = load_image(img_path)

                        weak_img = weak_aug(img.cpu().numpy().transpose(1,2,0)).unsqueeze(0).to(self.device)
                        strong_img = strong_aug(img.cpu().numpy().transpose(1,2,0)).unsqueeze(0).to(self.device)

                        # prepare prompts
                        context = learner()
                        captions = [init_prompt]

                        # --- weak pass ---
                        out_w = self.model(
                            nested_tensor_from_tensor_list([weak_img.squeeze(0)]).to(self.device),
                            context_vectors=context,
                            captions=captions,
                            context_embeddings_mapped=False,
                            K=self.k
                        )

                        pw_boxes = out_w["pred_boxes"][0]
                        pw_logits = out_w["pred_logits"][0][:, :self.T] #  first i batch, all 900 queries, then T tokens
                        # print(f"pw_logits: {pw_logits[:3]}")

                        # pseudo-labels = predictions with high confidence
                        scores = torch.softmax(pw_logits, dim=1)
                        # print(f"scores: {scores[:3]}")
                        # print(scores.sum(dim=1))
                        mask = scores > tau
                        # print(f"mask.sum(): {mask.sum(dim=1)}")

                        if mask.sum() == 0:
                            print(f"[WARNING] No positive predictions in weak pass for {img_name}")
                            
                            continue

                        # pseudo_boxes = pw_boxes[mask].detach()

                        # --- strong pass ---
                        out_s = self.model(
                            nested_tensor_from_tensor_list([strong_img.squeeze(0)]).to(self.device),
                            context_vectors=context,
                            captions=captions,
                            context_embeddings_mapped=False,
                            K=self.k
                        )

                        # ps_boxes = out_s["pred_boxes"][0]

                        # unsupervised loss = L1 between strong predictions and weak pseudo-labels
                        # (you can use IoU loss instead)
                        # if len(pseudo_boxes) > 0:
                        #     unsup_loss = F.l1_loss(
                        #         ps_boxes[:len(pseudo_boxes)],
                        #         pseudo_boxes,
                        #     )
                        #     unsup_loss_total += unsup_loss
                        #     num_unsup += 1

                        # --- NEW unsupervised loss using cross entropy on logits ---
                        # --- NEW unsupervised loss using thresholded multi-label BCE ---
                        if mask.sum() > 0:
                            print(f"[INFO] Computing unsupervised loss for {img_name}")
                            # Weak logits
                            weak_logits = out_w["pred_logits"][0][:, :self.T]          # [Kpos, T]
                            weak_probs  = torch.softmax(weak_logits, dim=1)

                            # Multi-hot pseudo labels
                            pseudo_labels = (weak_probs > tau).float()            # [Kpos, T]
                            print(f"pseudo_labels: {pseudo_labels[:3]}")
                            # Strong logits
                            strong_logits = out_s["pred_logits"][0][:, :self.T]     # [Kpos, T]
                            print(f"strong_logits: {strong_logits[:3]}")
                            # BCE loss over all tokens
                            unsup_loss = F.binary_cross_entropy_with_logits(
                                strong_logits,
                                pseudo_labels
                            )

                            unsup_loss_total += unsup_loss
                            num_unsup += 1


                
                batch_loss += (lambda_u * unsup_loss_total)
                print(f"loss: {loss}, unsup_loss_total: {unsup_loss_total}")

                # Accumulate metrics
                batch_metrics_accum['ap'].append(metrics['ap'])
                # batch_metrics_accum['num_pred'].append(metrics['num_pred_filtered'])
                batch_metrics_accum['num_pos'].append(metrics['num_pos'])
                batch_metrics_accum['cls_loss'].append(metrics['cls_loss'])
                batch_metrics_accum['box_loss'].append(metrics['box_loss'])
                # batch_metrics_accum['tumor_max'].append(metrics['score_stats']['tumor_max'])
                # batch_metrics_accum['tumor_mean'].append(metrics['score_stats']['tumor_mean'])
            
                # Average loss over batch
                batch_loss = batch_loss / current_batch_size
                
                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()

                # Check gradients
                if learner.context.grad is None:
                    print(f"[ERROR] No gradient at batch {batch_idx}!")
                    continue
                
                grad_norm = learner.context.grad.norm().item()
                if grad_norm < 1e-8:
                    if np.random.rand() < 0.1:
                        print(f"[WARNING] Tiny gradient: {grad_norm:.2e}")
                    continue
                
                # # Clip gradients
                # if grad_norm > 1.0:
                #     torch.nn.utils.clip_grad_norm_([learner.context], max_norm=1.0)
                
                optimizer.step()

                # Log metrics (average over batch)
                epoch_losses.append(batch_loss.item())
                epoch_aps.append(np.mean(batch_metrics_accum['ap']))
                epoch_metrics['cls_loss'].append(np.mean(batch_metrics_accum['cls_loss']))
                epoch_metrics['box_loss'].append(np.mean(batch_metrics_accum['box_loss']))
                # epoch_metrics['num_pred'].append(np.mean(batch_metrics_accum['num_pred']))
                epoch_metrics['num_pos'].append(np.mean(batch_metrics_accum['num_pos']))
                # epoch_metrics['tumor_max'].append(np.mean(batch_metrics_accum['tumor_max']))
                # epoch_metrics['tumor_mean'].append(np.mean(batch_metrics_accum['tumor_mean']))
                # epoch_metrics['pos_logit'].append(np.mean([m["pos_logit"] for m in batch_metrics_accum_list]))
                # epoch_metrics['neg_logit'].append(np.mean([m["neg_logit"] for m in batch_metrics_accum_list]))
                # epoch_metrics['avg_tumor_logit'].append(np.mean([m["avg_tumor_logit"] for m in batch_metrics_accum_list]))
                epoch_metrics['grad_norm'].append(grad_norm)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{batch_loss.item():.4f}",
                    'cls_loss': f"{np.mean(batch_metrics_accum['cls_loss']):.4f}",
                    'box_loss': f"{np.mean(batch_metrics_accum['box_loss']):.4f}",
                    'unsup_loss': f"{unsup_loss_total:.4f}",
                    # 'AP': f"{np.mean(batch_metrics_accum['ap']):.6f}",
                    # 'pred': f"{np.mean(batch_metrics_accum['num_pred']):.1f}",
                    # 'pos': f"{np.mean(batch_metrics_accum['num_pos']):.1f}",
                    'context_grad': f"{learner.context.grad.norm().item():.5f}",
                    # 'tumor_max': f"{np.mean(batch_metrics_accum['tumor_max']):.3f}",
                    # 'tumor_mean': f"{np.mean(batch_metrics_accum['tumor_mean']):.3f}",
                })

            print(f"\n[Epoch {ep}/{epochs}] Summary:")
            print(f"  Total Loss:    {np.mean(epoch_losses):.4f}")
            print(f"  Cls Loss:      {np.mean(epoch_metrics['cls_loss']):.4f}")
            print(f"  Box Loss:     {np.mean(epoch_metrics['box_loss']):.4f}")
            print(f"  Unsup Loss:    {(unsup_loss_total):.4f}")
            print(f"  Mean AP:       {np.mean(epoch_aps):.8f}")
            # print(f"  Avg Preds:     {np.mean(epoch_metrics['num_pred']):.1f}")
            print(f"  Avg Pos:       {np.mean(epoch_metrics['num_pos']):.1f}")
                # print(f"  Tumor Max:     {np.mean(epoch_metrics['tumor_max']):.4f}")
                # print(f"  Tumor Mean:    {np.mean(epoch_metrics['tumor_mean']):.4f}")
            print(f"  Context norm:  {learner.context.norm().item():.3f}")
            
            # Check if gradients are flowing
            if learner.context.grad is not None:
                print(f"  Context grad norm: {learner.context.grad.norm().item():.6f}")
            else:
                print("  WARNING: No gradients for context!")
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
def main(labeled_path=None, labeled_csv=None, unlabeled_path=None, unlabeled_csv=None, save_path=None, num_examples=None, lambda_u=1.0):
    trainer = CoOpTrainer(device=DEVICE, k_context=8)

    trainer.download_requirements(PATHS["weights"])
    trainer.load_model()

        # After loading model
    print("[INFO] Checking feat_map status:")
    print(f"  feat_map requires_grad: {trainer.model.feat_map.weight.requires_grad}")
    print(f"  feat_map device: {trainer.model.feat_map.weight.device}")

    # If it's False, enable it:
    for param in trainer.model.feat_map.parameters():
        param.requires_grad = True

    # train_path = dataset_path if dataset_path is not None else PATHS["train_B"]
    # train_csv = csv_path if csv_path is not None else "train_updated.csv"
    # save_path = save_path if save_path is not None else PATHS["task_2_1_save"] + train_path.replace("/", "_").replace(".", "") + ".pt"

    labeled_path = labeled_path if labeled_path is not None else PATHS["train_B"]
    labeled_csv = labeled_csv if labeled_csv is not None else "train_updated.csv"
    unlabeled_path = unlabeled_path if unlabeled_path is not None else PATHS["train_C"]
    unlabeled_csv = unlabeled_csv if unlabeled_csv is not None else "train.csv"

    id_l = normalize_dataset_path(labeled_path)
    id_u = normalize_dataset_path(unlabeled_path)
    save_path = save_path if save_path is not None else PATHS["task_3_save"] + id_l + "_" + id_u + "_" + str(lambda_u).replace(".", "_") + ".pt"

    trainer.train(
        labeled_path=labeled_path,
        labeled_csv=labeled_csv,
        unlabeled_path=unlabeled_path,
        unlabeled_csv=unlabeled_csv,
        init_prompt=DEFAULT_PROMPTS["B"][0],
        lr=5e-2,
        epochs=30,
        save_path=save_path,
        num_examples=num_examples,
        lambda_u=lambda_u
    )


if __name__ == "__main__":
    labeled_path = sys.argv[1] if len(sys.argv) > 1 else None
    labeled_csv  = sys.argv[2] if len(sys.argv) > 2 else None
    unlabeled_path = sys.argv[3] if len(sys.argv) > 3 else None
    unlabeled_csv = sys.argv[4] if len(sys.argv) > 4 else None
    save_path    = sys.argv[3] if len(sys.argv) > 3 else None
    num_examples = int(sys.argv[4]) if len(sys.argv) > 4 else None
    lambda_u = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

    for lambda_u in [1.0]:
        main(labeled_path=labeled_path, labeled_csv=labeled_csv, unlabeled_path=unlabeled_path, unlabeled_csv=unlabeled_csv, save_path=save_path, num_examples=num_examples, lambda_u=lambda_u)
