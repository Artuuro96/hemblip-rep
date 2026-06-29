#!/usr/bin/env python3
"""
Main training script for HemBLIP.

Usage:
    python scripts/train_hemblip.py --config configs/hemblip_lora.yaml
    python scripts/train_hemblip.py --config configs/hemblip_full.yaml
    python scripts/train_hemblip.py --config configs/hemblip_lora.yaml --device cuda
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make sure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hemblip_dataset import build_datasets
from src.models.hemblip import build_hemblip
from src.training.trainer import HemBLIPTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train HemBLIP")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detect if omitted)")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"  HemBLIP Training — {cfg['model']['name']}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Build model + processor
    # ------------------------------------------------------------------
    print("[1/3] Building model ...")
    model, processor = build_hemblip(
        base_model=cfg["model"]["base_model"],
        lora=cfg["model"].get("lora", False),
        lora_r=cfg["model"].get("lora_r", 16),
        lora_alpha=cfg["model"].get("lora_alpha", 32),
        lora_dropout=cfg["model"].get("lora_dropout", 0.05),
        lora_target_modules=cfg["model"].get("lora_target_modules"),
        checkpoint=args.resume,
    )

    # ------------------------------------------------------------------
    # 2. Build datasets
    # ------------------------------------------------------------------
    print("\n[2/3] Loading datasets ...")
    train_ds, val_ds, test_ds, external_ds = build_datasets(
        wbcatt_dir=cfg["data"]["wbcatt_dir"],
        leukemia_dir=cfg["data"]["leukemia_dir"],
        processor=processor,
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        max_length=cfg["data"]["max_caption_length"],
        seed=seed,
        external_dir=cfg["data"].get("external_dir"),
    )

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print("\n[3/3] Training ...")
    trainer = HemBLIPTrainer(
        model=model,
        processor=processor,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=cfg["training"]["output_dir"],
        num_epochs=cfg["training"]["num_epochs"],
        batch_size=cfg["training"]["batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
        warmup_steps=cfg["training"].get("warmup_steps", 200),
        patience=cfg["training"].get("patience", 5),
        fp16=cfg["training"].get("fp16", True),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 2),
        num_workers=cfg["training"].get("num_workers", 4),
        compile_model=cfg["training"].get("compile", False),
        device=args.device,
    )

    best_ckpt = trainer.train()
    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")
    print(
        "Run evaluation with:\n"
        f"  python scripts/evaluate.py --config {args.config} --checkpoint {best_ckpt}"
    )


if __name__ == "__main__":
    main()
