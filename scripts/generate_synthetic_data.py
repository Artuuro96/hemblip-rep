#!/usr/bin/env python3
"""
Generate a small synthetic dataset for smoke-testing the full pipeline
without needing the real WBCATt / LeukemiaAttri downloads.

Creates:
    data/wbcatt/images/<cell_type>/<id>.jpg
    data/wbcatt/annotations.json
    data/leukemia/images/<diagnosis>/<id>.jpg
    data/leukemia/annotations.json

Each image is a random 224×224 RGB patch with a coloured circle in the
centre to give the model something distinguishable per class.

Usage:
    python scripts/generate_synthetic_data.py --n 200
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path("data")

WBCATT_CELL_TYPES = ["neutrophil", "eosinophil", "basophil", "lymphocyte", "monocyte"]
LEUKEMIA_DIAGNOSES = ["ALL", "AML", "APML", "CLL", "CML"]

WBCATT_ATTRS = {
    "nuclear_shape":       ["round", "kidney", "horseshoe", "multilobulated"],
    "nuclear_color":       ["purple", "light_purple", "dark_purple"],
    "chromatin":           ["coarse", "fine", "open"],
    "lobularity":          [1, 2, 3, 4],
    "cytoplasm_amount":    ["abundant", "moderate", "scant"],
    "cytoplasm_color":     ["pink", "blue", "light_blue"],
    "cytoplasm_texture":   ["granular", "smooth"],
    "granularity":         ["present", "absent"],
    "granularity_texture": ["coarse", "fine"],
    "granularity_color":   ["red", "purple", "dark_purple"],
}

LEUKEMIA_ATTRS = {
    "cell_size":       ["small", "medium", "large"],
    "nuclear_shape":   ["round", "oval", "irregular", "bilobed"],
    "chromatin_texture": ["coarse", "fine", "open", "dense"],
    "nucleoli":        ["visible", "inconspicuous", "not_visible"],
    "cytoplasm_amount": ["abundant", "moderate", "scant"],
    "basophilia":      ["high", "moderate", "low"],
}

CLASS_COLORS = {
    "neutrophil": (220, 180, 255),
    "eosinophil": (255, 200, 150),
    "basophil":   (150, 150, 255),
    "lymphocyte": (200, 255, 200),
    "monocyte":   (255, 230, 180),
    "ALL":  (255, 100, 100),
    "AML":  (255, 150, 50),
    "APML": (200, 50, 200),
    "CLL":  (50, 150, 255),
    "CML":  (100, 200, 100),
}


def make_cell_image(label: str, size: int = 224) -> Image.Image:
    """Create a synthetic blood cell image with a distinguishable colour."""
    rng = random.Random()
    bg = tuple(rng.randint(220, 245) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)

    color = CLASS_COLORS.get(label, (128, 128, 128))
    noise = tuple(min(255, max(0, c + rng.randint(-20, 20))) for c in color)

    r = rng.randint(40, 60)
    cx = size // 2 + rng.randint(-10, 10)
    cy = size // 2 + rng.randint(-10, 10)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=noise)

    # nucleus (darker inner circle)
    nucleus_color = tuple(max(0, c - 80) for c in noise)
    nr = r // 2
    draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=nucleus_color)

    return img


def generate_wbcatt(n_per_class: int, out_dir: Path) -> None:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    annotations = []
    idx = 0
    for cell_type in WBCATT_CELL_TYPES:
        (out_dir / "images" / cell_type).mkdir(exist_ok=True)
        for _ in range(n_per_class):
            image_id = f"wbc_{idx:05d}"
            img = make_cell_image(cell_type)
            img.save(out_dir / "images" / cell_type / f"{image_id}.jpg")

            attrs = {"image_id": image_id, "cell_type": cell_type}
            for attr, choices in WBCATT_ATTRS.items():
                attrs[attr] = random.choice(choices)
            annotations.append(attrs)
            idx += 1

    with open(out_dir / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"  WBCATt: {idx} images → {out_dir}")


def generate_leukemia(n_per_class: int, out_dir: Path) -> None:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    annotations = []
    idx = 0
    for diagnosis in LEUKEMIA_DIAGNOSES:
        (out_dir / "images" / diagnosis).mkdir(exist_ok=True)
        for _ in range(n_per_class):
            image_id = f"leu_{idx:05d}"
            img = make_cell_image(diagnosis)
            img.save(out_dir / "images" / diagnosis / f"{image_id}.jpg")

            attrs = {"image_id": image_id, "diagnosis": diagnosis}
            for attr, choices in LEUKEMIA_ATTRS.items():
                attrs[attr] = random.choice(choices)
            annotations.append(attrs)
            idx += 1

    with open(out_dir / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"  LeukemiaAttri: {idx} images → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40,
                   help="Images per class (default 40 → 200 WBC + 200 leukemia)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Generating synthetic data ({args.n} images per class) ...")
    generate_wbcatt(args.n, DATA_ROOT / "wbcatt")
    generate_leukemia(args.n, DATA_ROOT / "leukemia")
    print("\nDone. Run the pipeline with:")
    print("  python scripts/train_hemblip.py --config configs/hemblip_lora.yaml")


if __name__ == "__main__":
    main()
