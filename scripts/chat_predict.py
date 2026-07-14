#!/usr/bin/env python3
"""
Small interactive CLI: pick an image, optionally give a text prompt to
condition the caption on, see what HemBLIP generates. Loads the model once
and loops so you can try several images without reloading.

Usage:
    python scripts/chat_predict.py --config configs/hemblip_lora.yaml \
        --checkpoint outputs/hemblip_lora/best --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hemblip import build_hemblip
from src.training.trainer import _auto_device


def parse_args():
    p = argparse.ArgumentParser(description="Interactive HemBLIP captioning")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
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
    model.to(device).eval()
    base = model.base_model.model if hasattr(model, "base_model") else model

    print("\nHemBLIP — escribe 'salir' para terminar.\n")
    while True:
        image_path = input("Ruta de la imagen: ").strip().strip('"')
        if image_path.lower() in ("salir", "exit", "quit", ""):
            break
        if not Path(image_path).exists():
            print(f"  No existe: {image_path}\n")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  No se pudo abrir la imagen: {e}\n")
            continue

        prompt = input("Prompt (opcional, Enter = ninguno): ").strip()

        inputs = processor(images=image, text=prompt or None, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = base.generate(
                **inputs, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
            )
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        print(f"  -> {caption}\n")


if __name__ == "__main__":
    main()
