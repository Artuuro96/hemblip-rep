#!/usr/bin/env python3
"""
Exploratory data analysis of the HemBLIP caption dataset (WBCAtt attribute
CSVs + the captions generated from them).

Reads the same source the training pipeline uses (`pbc_attr_v1_{train,val,
test}.csv` under --wbcatt_dir) and builds the same per-image captions via
`caption_from_attributes()`, without touching the model/processor — so it's
fast and needs no GPU.

Output (--out_dir, default graphics/eda/)
-------------------------------------------
  class_distribution.png           image count per cell type, by split
  caption_length_distribution.png  caption length in words
  top_words.png                    most frequent caption words
  attribute_distributions.png      value counts for each WBCAtt attribute
  chromatin_by_celltype.png        chromatin_density x cell_type heatmap
  summary.txt                      dataset-level stats (counts, vocab, uniqueness)

Usage
-----
  py scripts/eda_captions.py
  py scripts/eda_captions.py --wbcatt_dir PBC_dataset_normal_DIB --seed 42
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    sys.exit("matplotlib not found. Run: pip install matplotlib>=3.7.0")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.caption_templates import caption_from_attributes
from src.data.wbcatt_attributes import ATTRIBUTE_COLUMNS, has_wbcatt_attributes, load_wbcatt_split

# ── Validated palette (dataviz skill reference palette) ─────────────────────
CAT = {
    "blue":   "#2a78d6",
    "aqua":   "#1baf7a",
    "yellow": "#eda100",
    "green":  "#008300",
    "violet": "#4a3aa7",
    "red":    "#e34948",
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK_PRIMARY   = "#0b0b0b"
INK_MUTED     = "#898781"
GRID          = "#e1e0d9"
SURFACE       = "#fcfcfb"

SPLIT_COLORS = {"train": CAT["blue"], "val": CAT["aqua"], "test": CAT["yellow"]}

_STOPWORDS = {
    "a", "an", "the", "is", "of", "and", "with", "this", "in", "to", "no",
    "are", "showing", "shows", "cell", "cells",
}

GRAPHICS_DIR = Path("graphics") / "eda"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (mirrors _build_wbcatt_attribute_splits, without the Dataset/
# tokenizer machinery — this script never touches the model)
# ─────────────────────────────────────────────────────────────────────────────

def load_all_records(wbcatt_dir: str, seed: int) -> List[Dict]:
    if not has_wbcatt_attributes(wbcatt_dir):
        sys.exit(f"No WBCAtt attribute CSVs found under {wbcatt_dir}")

    rng = random.Random(seed)
    all_records: List[Dict] = []
    for split in ("train", "val", "test"):
        records = load_wbcatt_split(wbcatt_dir, split)
        for r in records:
            r["split"] = split
            r["caption"] = caption_from_attributes(r, rng)
        all_records.extend(records)

    if not any(r["split"] == "val" for r in all_records):
        print("  [eda] no official val split — that's expected, train.py carves "
              "one out at training time; this EDA treats all non-test as 'train'")
    return all_records


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_class_distribution(records: List[Dict], out_path: Path) -> None:
    cell_types = sorted({r["cell_type"] for r in records})
    splits = ["train", "val", "test"]
    counts = {s: [sum(1 for r in records if r["cell_type"] == c and r["split"] == s)
                  for c in cell_types] for s in splits}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cell_types))
    width = 0.25
    for i, s in enumerate(splits):
        offset = (i - 1) * width
        ax.bar(x + offset, counts[s], width, label=s,
               color=SPLIT_COLORS[s], edgecolor=SURFACE, linewidth=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, fontsize=9)
    ax.set_ylabel("Images", fontsize=9, color=INK_PRIMARY)
    ax.set_title("Class distribution by split", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax.legend(fontsize=8, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_caption_length(records: List[Dict], out_path: Path) -> None:
    lengths = np.array([len(r["caption"].split()) for r in records])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=range(lengths.min(), lengths.max() + 2),
            color=SEQ_BLUE[3], edgecolor=SURFACE, linewidth=0.6, zorder=3)

    mean_len = float(lengths.mean())
    ax.axvline(mean_len, color=SEQ_BLUE[6], linewidth=1.2, linestyle="--", zorder=4)
    ax.annotate(f" mean={mean_len:.1f} words", xy=(mean_len, ax.get_ylim()[1] * 0.9),
                fontsize=8, color=SEQ_BLUE[6])

    ax.set_xlabel("Caption length (words)", fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel("Count", fontsize=9, color=INK_PRIMARY)
    ax.set_title("Caption length distribution", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_top_words(records: List[Dict], out_path: Path, top_n: int = 25) -> Counter:
    words: Counter = Counter()
    for r in records:
        tokens = re.findall(r"[a-z]+", r["caption"].lower())
        words.update(t for t in tokens if t not in _STOPWORDS and len(t) > 2)

    top = words.most_common(top_n)
    labels = [w for w, _ in top][::-1]
    values = [c for _, c in top][::-1]

    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(labels))
    ax.barh(y, values, color=SEQ_BLUE[3], edgecolor=SURFACE, linewidth=0.6, zorder=3)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.01, yi, str(v), va="center", fontsize=7, color=INK_MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Occurrences", fontsize=9, color=INK_PRIMARY)
    ax.set_title(f"Top {top_n} caption words", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return words


def plot_attribute_distributions(records: List[Dict], out_path: Path) -> None:
    n = len(ATTRIBUTE_COLUMNS)
    ncols, nrows = 3, -(-n // 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.1 * nrows))
    axes = axes.flatten()

    for i, attr in enumerate(ATTRIBUTE_COLUMNS):
        ax = axes[i]
        counts = Counter(r[attr] for r in records)
        items = counts.most_common()
        labels = [k for k, _ in items]
        values = [v for _, v in items]

        y = np.arange(len(labels))
        ax.barh(y, values, color=SEQ_BLUE[3], edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(attr, fontsize=9, fontweight="bold", color=INK_PRIMARY)
        ax.invert_yaxis()
        _style_axes(ax)
        ax.grid(axis="y", visible=False)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("WBCAtt attribute value distributions", fontsize=12,
                 fontweight="bold", color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_attribute_by_celltype_heatmap(records: List[Dict], attribute: str, out_path: Path) -> None:
    cell_types = sorted({r["cell_type"] for r in records})
    values = sorted({r[attribute] for r in records})

    matrix = np.zeros((len(values), len(cell_types)))
    for r in records:
        i = values.index(r[attribute])
        j = cell_types.index(r["cell_type"])
        matrix[i, j] += 1

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(7, 3 + 0.4 * len(values)))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    for i in range(len(values)):
        for j in range(len(cell_types)):
            v = int(matrix[i, j])
            if v == 0:
                continue
            lum = matrix[i, j] / matrix.max()
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color=SURFACE if lum > 0.55 else INK_PRIMARY)

    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_types, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(values, fontsize=8)
    ax.set_title(f"{attribute} x cell type", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    fig.colorbar(im, ax=ax, label="images", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(records: List[Dict], vocab: Counter, out_path: Path) -> None:
    lengths = [len(r["caption"].split()) for r in records]
    captions = [r["caption"] for r in records]
    n_unique = len(set(captions))

    lines = [
        "HemBLIP Caption Dataset — EDA Summary",
        "=" * 45,
        f"Total images       : {len(records)}",
    ]
    for split in ("train", "val", "test"):
        n = sum(1 for r in records if r["split"] == split)
        lines.append(f"  {split:<5}            : {n}")

    lines += [
        "",
        f"Cell types          : {sorted({r['cell_type'] for r in records})}",
        f"Caption length (words): mean={np.mean(lengths):.1f}  "
        f"median={np.median(lengths):.0f}  min={min(lengths)}  max={max(lengths)}",
        f"Unique captions      : {n_unique} / {len(captions)} "
        f"({100 * n_unique / len(captions):.1f}%)",
        f"Vocabulary size      : {len(vocab)} distinct content words",
        "",
        "Top 10 words:",
    ]
    for w, c in vocab.most_common(10):
        lines.append(f"  {w:<15} {c}")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print("\n" + text)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Exploratory data analysis of HemBLIP captions")
    p.add_argument("--wbcatt_dir", default="PBC_dataset_normal_DIB")
    p.add_argument("--out_dir", default=str(GRAPHICS_DIR))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading records from {args.wbcatt_dir} ...")
    records = load_all_records(args.wbcatt_dir, args.seed)
    print(f"Loaded {len(records)} captioned images.")

    plot_class_distribution(records, out_dir / "class_distribution.png")
    plot_caption_length(records, out_dir / "caption_length_distribution.png")
    vocab = plot_top_words(records, out_dir / "top_words.png")
    plot_attribute_distributions(records, out_dir / "attribute_distributions.png")
    plot_attribute_by_celltype_heatmap(
        records, "chromatin_density", out_dir / "chromatin_by_celltype.png",
    )
    write_summary(records, vocab, out_dir / "summary.txt")

    print(f"\nSaved figures + summary.txt -> {out_dir}/")


if __name__ == "__main__":
    main()
