"""
Frozen-backbone cosine-similarity classifier (paper §2.3, Table 3).

A lightweight linear head is trained on frozen image embeddings extracted
from the vision encoder.  No gradient flows through the encoder.

Tasks:
  • Leukemia subtype classification  (5 classes: ALL, AML, APML, CLL, CML)
  • Cell-type classification          (up to 13 classes when combined)

Usage::

    clf = FrozenBackboneClassifier(embed_dim=768, num_classes=5)
    clf.fit(train_embeddings, train_labels)
    acc, f1 = clf.evaluate(test_embeddings, test_labels)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Embedding extractor
# ---------------------------------------------------------------------------

def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the frozen vision encoder over a DataLoader and collect embeddings.

    Args:
        model:      BLIP / PEFT model (vision encoder must be present).
        dataloader: Yields batches with 'pixel_values' and 'records'.

    Returns:
        (embeddings, labels) where labels are taken from record['diagnosis_idx']
        or record['cell_type_idx'], whichever is present.  If both, they are
        stacked; callers can pick the task they need.
    """
    model.eval()
    all_embeds: List[np.ndarray] = []
    all_labels: List[Dict] = []

    use_fp16 = device == "cuda"
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)

            base = model.base_model.model if hasattr(model, "base_model") else model
            ctx = torch.amp.autocast("cuda") if use_fp16 else torch.amp.autocast("cpu", enabled=False)
            with ctx:
                vision_out = base.vision_model(pixel_values=pixel_values, return_dict=True)
            cls_embed = vision_out.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeds.append(cls_embed)

            for r in batch["records"]:
                all_labels.append({
                    "diagnosis_idx": r.get("diagnosis_idx", -1),
                    "cell_type_idx": r.get("cell_type_idx", -1),
                })

    embeddings = np.concatenate(all_embeds, axis=0)
    return embeddings, all_labels


# ---------------------------------------------------------------------------
# Lightweight cosine-similarity classifier (logistic regression on L2-normalised
# embeddings, as is standard for frozen-encoder evaluation)
# ---------------------------------------------------------------------------

class FrozenBackboneClassifier:
    """
    Logistic regression probe on top of frozen image embeddings.

    Cosine similarity is approximated by L2-normalising the embeddings before
    fitting a linear classifier (equivalent to cosine-similarity classification
    with a linear head, as described in the paper).
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0) -> None:
        self.max_iter = max_iter
        self.C = C
        self.clf = LogisticRegression(
            max_iter=max_iter, C=C, solver="lbfgs", multi_class="multinomial",
        )

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "FrozenBackboneClassifier":
        X = normalize(embeddings, norm="l2")
        self.clf.fit(X, labels)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        X = normalize(embeddings, norm="l2")
        return self.clf.predict(X)

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        preds = self.predict(embeddings)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return {"accuracy": acc, "f1_weighted": f1}


# ---------------------------------------------------------------------------
# Full evaluation pipeline (Table 3)
# ---------------------------------------------------------------------------

def run_classifier_evaluation(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
    external_loader: Optional[DataLoader] = None,
) -> Dict[str, Dict]:
    """
    Replicate Table 3: train a frozen-backbone probe and evaluate on
    internal (and optionally external) test sets.

    Returns::

        {
          "leukemia_subtype": {
              "internal": {"accuracy": ..., "f1_weighted": ...},
              "external": {...} or None,
          },
          "cell_type": {...},
        }
    """
    print("  Extracting train embeddings ...")
    train_embeds, train_labels = extract_embeddings(model, train_loader, device)

    print("  Extracting test embeddings ...")
    test_embeds, test_labels = extract_embeddings(model, test_loader, device)

    ext_embeds = ext_labels = None
    if external_loader is not None:
        print("  Extracting external embeddings ...")
        ext_embeds, ext_labels = extract_embeddings(model, external_loader, device)

    results: Dict[str, Dict] = {}

    for task, idx_key in [("leukemia_subtype", "diagnosis_idx"), ("cell_type", "cell_type_idx")]:
        train_y = np.array([l[idx_key] for l in train_labels])
        test_y = np.array([l[idx_key] for l in test_labels])

        # Filter out samples with unknown labels (-1)
        train_mask = train_y >= 0
        test_mask = test_y >= 0
        if train_mask.sum() < 10:
            results[task] = {"internal": None, "external": None}
            continue

        clf = FrozenBackboneClassifier()
        clf.fit(train_embeds[train_mask], train_y[train_mask])

        internal_metrics = clf.evaluate(test_embeds[test_mask], test_y[test_mask])

        external_metrics = None
        if ext_embeds is not None and ext_labels is not None:
            ext_y = np.array([l[idx_key] for l in ext_labels])
            ext_mask = ext_y >= 0
            if ext_mask.sum() > 0:
                external_metrics = clf.evaluate(ext_embeds[ext_mask], ext_y[ext_mask])

        results[task] = {"internal": internal_metrics, "external": external_metrics}
        print(
            f"  {task}: acc={internal_metrics['accuracy']:.3f} "
            f"f1={internal_metrics['f1_weighted']:.3f} (internal)"
        )

    return results
