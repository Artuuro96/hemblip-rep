#!/usr/bin/env python3
"""
Caption one or more images with a trained HemBLIP checkpoint.

Usage:
    python scripts/predict.py --config configs/hemblip_lora.yaml \
        --checkpoint outputs/hemblip_lora/best \
        --image PBC_dataset_normal_DIB/PBC_dataset_normal_DIB/neutrophil/SNE_1.jpg

    # Multiple images, GPU
    python scripts/predict.py --config configs/hemblip_lora.yaml \
        --checkpoint outputs/hemblip_lora/best \
        --image img1.jpg img2.jpg --device cuda
"""

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hemblip import build_hemblip, generate_captions
from src.training.trainer import _auto_device


def parse_args():
    p = argparse.ArgumentParser(description="Caption images with a trained HemBLIP checkpoint")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", nargs="+", required=True, help="One or more image paths")
    p.add_argument("--device", default=None)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_beams", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or _auto_device()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading checkpoint {args.checkpoint} on {device} ...")
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

    images = [Image.open(p).convert("RGB") for p in args.image]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    captions = generate_captions(
        model, processor, pixel_values,
        max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, device=device,
    )

    for path, caption in zip(args.image, captions):
        print(f"\n{path}\n  -> {caption}")


if __name__ == "__main__":
    main()
