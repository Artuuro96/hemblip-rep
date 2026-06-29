#!/usr/bin/env python3
"""
Evaluation script — reproduces Tables 1, 2, and 3 from the paper.

Usage:
    python scripts/evaluate.py --config configs/hemblip_lora.yaml \
                               --checkpoint outputs/hemblip_lora/best
    python scripts/evaluate.py --config configs/hemblip_lora.yaml \
                               --checkpoint outputs/hemblip_lora/best \
                               --external          # also evaluate on external set
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hemblip_dataset import HemBLIPCollator, build_datasets
from src.evaluation.attribute_extractor import (
    compute_attribute_accuracy,
    compute_attribute_accuracy_from_annotations,
)
from src.evaluation.metrics import compute_all_metrics, generate_predictions
from src.models.classifier import run_classifier_evaluation
from src.models.hemblip import build_hemblip
from src.training.trainer import _auto_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HemBLIP")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--external", action="store_true", help="Include external test set")
    p.add_argument("--device", default=None)
    p.add_argument("--bertscore_model", default="distilbert-base-uncased",
                   help="BERTScore backbone (use roberta-large for paper-accurate numbers)")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or _auto_device()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"  HemBLIP Evaluation — {cfg['model']['name']}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load model + processor
    # ------------------------------------------------------------------
    print("[1/4] Loading model ...")
    model, processor = build_hemblip(
        base_model=cfg["model"]["base_model"],
        lora=cfg["model"].get("lora", False),
        lora_r=cfg["model"].get("lora_r", 16),
        lora_alpha=cfg["model"].get("lora_alpha", 32),
        lora_dropout=cfg["model"].get("lora_dropout", 0.05),
        lora_target_modules=cfg["model"].get("lora_target_modules"),
        checkpoint=args.checkpoint,
    )
    model.to(device)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("\n[2/4] Loading datasets ...")
    train_ds, val_ds, test_ds, external_ds = build_datasets(
        wbcatt_dir=cfg["data"]["wbcatt_dir"],
        leukemia_dir=cfg["data"]["leukemia_dir"],
        processor=processor,
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        max_length=cfg["data"]["max_caption_length"],
        seed=cfg["training"].get("seed", 42),
        external_dir=cfg["data"].get("external_dir") if args.external else None,
    )

    collator = HemBLIPCollator()
    eval_bs = cfg["evaluation"].get("batch_size", 32)

    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator)
    external_loader = (
        DataLoader(external_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator)
        if external_ds is not None else None
    )
    train_loader_clf = DataLoader(
        train_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator
    )

    max_new_tokens = cfg["evaluation"].get("max_new_tokens", 128)
    num_beams = cfg["evaluation"].get("num_beams", 4)

    # ------------------------------------------------------------------
    # Table 1 — Caption generation metrics
    # ------------------------------------------------------------------
    print("\n[3/4] Computing caption generation metrics (Table 1) ...")

    print("  Internal test set:")
    preds_int, refs_int = generate_predictions(
        model, processor, test_loader, device, max_new_tokens, num_beams
    )
    metrics_int = compute_all_metrics(preds_int, refs_int, args.bertscore_model, device)
    print(f"    BLEU={metrics_int['bleu']:.4f}  ROUGE-L={metrics_int['rouge_l']:.4f}  "
          f"BERTScore={metrics_int['bertscore_f1']:.4f}")

    metrics_ext = None
    if external_loader is not None:
        print("  External test set:")
        preds_ext, refs_ext = generate_predictions(
            model, processor, external_loader, device, max_new_tokens, num_beams
        )
        metrics_ext = compute_all_metrics(preds_ext, refs_ext, args.bertscore_model, device)
        print(f"    BLEU={metrics_ext['bleu']:.4f}  ROUGE-L={metrics_ext['rouge_l']:.4f}  "
              f"BERTScore={metrics_ext['bertscore_f1']:.4f}")

    # ------------------------------------------------------------------
    # Table 2 — Morphological attribute accuracy
    # ------------------------------------------------------------------
    print("\n  Morphological attribute accuracy (Table 2):")
    attr_acc_int = compute_attribute_accuracy(preds_int, refs_int)
    _print_attr_table("Internal", attr_acc_int)

    if metrics_ext is not None:
        attr_acc_ext = compute_attribute_accuracy(preds_ext, refs_ext)
        _print_attr_table("External", attr_acc_ext)
    else:
        attr_acc_ext = None

    # ------------------------------------------------------------------
    # Table 3 — Frozen backbone classifier
    # ------------------------------------------------------------------
    print("\n[4/4] Frozen backbone classification (Table 3) ...")
    clf_results = run_classifier_evaluation(
        model=model,
        train_loader=train_loader_clf,
        test_loader=test_loader,
        device=device,
        external_loader=external_loader,
    )

    print("\n  Leukemia subtype classification:")
    _print_clf(clf_results.get("leukemia_subtype", {}))
    print("  Cell-type classification:")
    _print_clf(clf_results.get("cell_type", {}))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = Path(args.checkpoint) / "eval_results.json"
    results = {
        "model": cfg["model"]["name"],
        "checkpoint": args.checkpoint,
        "caption_metrics": {
            "internal": metrics_int,
            "external": metrics_ext,
        },
        "attribute_accuracy": {
            "internal": {k: (None if v != v else v) for k, v in attr_acc_int.items()},
            "external": (
                {k: (None if v != v else v) for k, v in attr_acc_ext.items()}
                if attr_acc_ext else None
            ),
        },
        "classifier": clf_results,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------

def _print_attr_table(split: str, acc: dict) -> None:
    print(f"    [{split}]")
    for attr, val in acc.items():
        bar = f"{val*100:.1f}%" if val == val else "N/A"
        print(f"      {attr:<35} {bar}")


def _print_clf(result: dict) -> None:
    for split in ("internal", "external"):
        m = result.get(split)
        if m:
            print(f"    {split}: acc={m['accuracy']:.3f}  f1={m['f1_weighted']:.3f}")


if __name__ == "__main__":
    main()
