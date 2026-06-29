#!/usr/bin/env python3
"""
Plot training curves and evaluation metrics for HemBLIP runs.

Reads
-----
  <run_dir>/training_history.json      train_loss / val_loss per epoch
  <run_dir>/best/eval_results.json     caption metrics + classifier metrics
                                       (only present after running evaluate.py)

Output
------
  graphics/<run_name>_curves.png       one file per run (loss + optional metrics)
  graphics/comparison.png              multi-run overlay (when >1 run is given)

Usage
-----
  # After training only (loss curves):
  py scripts/plot_training.py --runs outputs/hemblip_lora

  # After training + evaluate.py:
  py scripts/plot_training.py --runs outputs/hemblip_lora

  # Compare two runs:
  py scripts/plot_training.py --runs outputs/hemblip_lora outputs/hemblip_full

  # Custom checkpoint location:
  py scripts/plot_training.py --runs outputs/hemblip_lora \
      --checkpoints outputs/hemblip_lora/best
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend (safe on all OSes)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    sys.exit("matplotlib not found. Run: pip install matplotlib>=3.7.0")

GRAPHICS_DIR = Path("graphics")

# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_run(run_dir: Path) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Return (history, eval_results) for a run directory."""
    history = _load_json(run_dir / "training_history.json")
    # eval_results lives inside the best checkpoint sub-directory
    eval_res = _load_json(run_dir / "best" / "eval_results.json")
    return history, eval_res


# ──────────────────────────────────────────────────────────────────────────────
# Per-run figure  (loss + metrics if available)
# ──────────────────────────────────────────────────────────────────────────────

def _plot_loss(ax: "plt.Axes", history: Dict, color: str, label_prefix: str = "") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"],
            color=color, linestyle="-", linewidth=2,
            label=f"{label_prefix}train loss")
    ax.plot(epochs, history["val_loss"],
            color=color, linestyle="--", linewidth=2, alpha=0.85,
            label=f"{label_prefix}val loss")

    best_ep  = int(np.argmin(history["val_loss"])) + 1
    best_val = float(np.min(history["val_loss"]))
    ax.axvline(best_ep, color=color, linewidth=0.8, alpha=0.35, linestyle=":")
    ax.scatter([best_ep], [best_val], color=color, s=70, zorder=5)
    ax.annotate(
        f" ep {best_ep}\n {best_val:.4f}",
        xy=(best_ep, best_val),
        fontsize=7, color=color,
        xytext=(4, 4), textcoords="offset points",
    )


def _bar_group(ax: "plt.Axes", labels: List[str], values: List[float],
               title: str, x_labels: List[str], ylim: float = 1.0,
               palette=None) -> None:
    x = np.arange(len(x_labels))
    n = len(labels)
    width = min(0.7 / max(n, 1), 0.25)
    if palette is None:
        palette = plt.cm.Set2(np.linspace(0, 0.8, n))

    for i, (lbl, vals) in enumerate(zip(labels, values)):
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=lbl,
                      color=palette[i], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_single_run(run_dir: Path, history: Dict,
                    eval_res: Optional[Dict], out_path: Path) -> None:
    """Generate the figure for one run."""
    has_eval = eval_res is not None

    if has_eval:
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(
            2, 3, figure=fig, hspace=0.45, wspace=0.4,
            left=0.07, right=0.97, top=0.90, bottom=0.08,
        )
        ax_loss = fig.add_subplot(gs[:, 0])   # full left column
        ax_cap  = fig.add_subplot(gs[0, 1])
        ax_clf  = fig.add_subplot(gs[1, 1])
        ax_f1   = fig.add_subplot(gs[0, 2])
        ax_acc  = fig.add_subplot(gs[1, 2])
    else:
        fig, ax_loss = plt.subplots(figsize=(8, 5))
        ax_cap = ax_clf = ax_f1 = ax_acc = None

    # ── Loss curves ──────────────────────────────────────────────────────────
    color = "#2196F3"
    _plot_loss(ax_loss, history, color)
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Cross-entropy loss", fontsize=10)
    ax_loss.set_title("Training & Validation Loss", fontsize=11, fontweight="bold")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)
    ax_loss.spines[["top", "right"]].set_visible(False)

    # ── Eval panels ──────────────────────────────────────────────────────────
    if has_eval and ax_cap is not None:
        cm_int = eval_res.get("caption_metrics", {}).get("internal", {}) or {}
        cm_ext = eval_res.get("caption_metrics", {}).get("external") or {}

        # Caption metrics (BLEU, ROUGE-L, BERTScore)
        cap_keys   = ["bleu", "rouge_l", "bertscore_f1"]
        cap_labels = ["BLEU-4", "ROUGE-L", "BERTScore F1"]
        cap_int = [cm_int.get(k, 0) or 0 for k in cap_keys]
        cap_ext = [cm_ext.get(k, 0) or 0 for k in cap_keys] if cm_ext else None

        rows_cap = [cap_int]
        lbls_cap = ["Internal"]
        if cap_ext:
            rows_cap.append(cap_ext)
            lbls_cap.append("External")

        _bar_group(ax_cap, lbls_cap, rows_cap,
                   "Caption Metrics (Table 1)", cap_labels,
                   palette=plt.cm.Blues(np.linspace(0.45, 0.75, len(lbls_cap))))

        # Classifier accuracy
        clf = eval_res.get("classifier", {})
        tasks_map = {
            "Leukemia\nsubtype": "leukemia_subtype",
            "Cell\ntype":        "cell_type",
        }
        acc_int = [
            ((clf.get(v) or {}).get("internal") or {}).get("accuracy", 0) or 0
            for v in tasks_map.values()
        ]
        acc_ext = [
            ((clf.get(v) or {}).get("external") or {}).get("accuracy", 0) or 0
            for v in tasks_map.values()
        ]
        f1_int = [
            ((clf.get(v) or {}).get("internal") or {}).get("f1_weighted", 0) or 0
            for v in tasks_map.values()
        ]
        f1_ext = [
            ((clf.get(v) or {}).get("external") or {}).get("f1_weighted", 0) or 0
            for v in tasks_map.values()
        ]

        rows_acc = [acc_int]
        lbls_acc = ["Internal"]
        rows_f1  = [f1_int]
        lbls_f1  = ["Internal"]
        if any(v > 0 for v in acc_ext):
            rows_acc.append(acc_ext)
            lbls_acc.append("External")
            rows_f1.append(f1_ext)
            lbls_f1.append("External")

        _bar_group(ax_acc, lbls_acc, rows_acc,
                   "Classifier Accuracy (Table 3)", list(tasks_map.keys()),
                   palette=plt.cm.Greens(np.linspace(0.45, 0.75, len(lbls_acc))))

        _bar_group(ax_f1, lbls_f1, rows_f1,
                   "Classifier F1-weighted (Table 3)", list(tasks_map.keys()),
                   palette=plt.cm.Oranges(np.linspace(0.45, 0.75, len(lbls_f1))))

        # Attribute accuracy (Table 2)
        attr_acc = (eval_res.get("attribute_accuracy") or {}).get("internal") or {}
        if attr_acc:
            attr_names  = list(attr_acc.keys())
            attr_values = [attr_acc[k] if attr_acc[k] == attr_acc[k] else 0
                           for k in attr_names]
            _bar_group(
                ax_clf, ["Internal"], [attr_values],
                "Morphological Attribute Accuracy (Table 2)",
                [a.replace("_", "\n") for a in attr_names],
                ylim=1.05,
                palette=plt.cm.Purples(np.linspace(0.5, 0.5, 1)),
            )
            ax_clf.tick_params(axis="x", labelsize=7)

    model_name = run_dir.name
    n_ep = len(history["train_loss"])
    best_ep = int(np.argmin(history["val_loss"])) + 1
    subtitle = f"epochs trained: {n_ep}  |  best epoch: {best_ep}"
    if not has_eval:
        subtitle += "  |  run evaluate.py to add metric panels"
    fig.suptitle(f"HemBLIP — {model_name}\n{subtitle}",
                 fontsize=12, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Multi-run comparison figure
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(run_dirs: List[Path],
                    histories: List[Optional[Dict]],
                    eval_results: List[Optional[Dict]],
                    out_path: Path) -> None:
    has_eval = any(e is not None for e in eval_results)
    n = len(run_dirs)
    palette = plt.cm.tab10(np.linspace(0, 0.9, n))

    if has_eval:
        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(
            2, 2, figure=fig, hspace=0.45, wspace=0.35,
            left=0.07, right=0.97, top=0.88, bottom=0.08,
        )
        ax_loss = fig.add_subplot(gs[:, 0])
        ax_cap  = fig.add_subplot(gs[0, 1])
        ax_acc  = fig.add_subplot(gs[1, 1])
    else:
        fig, ax_loss = plt.subplots(figsize=(9, 5))
        ax_cap = ax_acc = None

    # Loss overlay
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Cross-entropy loss", fontsize=10)
    ax_loss.set_title("Loss Comparison", fontsize=11, fontweight="bold")
    ax_loss.grid(alpha=0.3)
    ax_loss.spines[["top", "right"]].set_visible(False)

    for run_dir, hist, color in zip(run_dirs, histories, palette):
        if hist is None:
            continue
        _plot_loss(ax_loss, hist, color, label_prefix=f"{run_dir.name} ")
    ax_loss.legend(fontsize=8)

    if has_eval and ax_cap is not None:
        labels = [r.name for r in run_dirs]
        cap_keys   = ["bleu", "rouge_l", "bertscore_f1"]
        cap_labels = ["BLEU-4", "ROUGE-L", "BERTScore F1"]

        rows_cap = []
        for ev in eval_results:
            cm = (ev or {}).get("caption_metrics", {}).get("internal", {}) or {}
            rows_cap.append([cm.get(k, 0) or 0 for k in cap_keys])

        _bar_group(ax_cap, labels, rows_cap,
                   "Caption Metrics — Internal test (Table 1)", cap_labels,
                   palette=palette)

        tasks_map = {
            "Leukemia\nsubtype": "leukemia_subtype",
            "Cell\ntype":        "cell_type",
        }
        rows_acc, rows_f1 = [], []
        for ev in eval_results:
            clf = (ev or {}).get("classifier", {})
            rows_acc.append([
                ((clf.get(v) or {}).get("internal") or {}).get("accuracy", 0) or 0
                for v in tasks_map.values()
            ])
            rows_f1.append([
                ((clf.get(v) or {}).get("internal") or {}).get("f1_weighted", 0) or 0
                for v in tasks_map.values()
            ])

        # Pack accuracy + F1 side by side per task
        combined_labels = [f"{t}\nAcc" for t in tasks_map] + [f"{t}\nF1" for t in tasks_map]
        rows_combined = [a + f for a, f in zip(rows_acc, rows_f1)]
        _bar_group(ax_acc, labels, rows_combined,
                   "Classifier Metrics (Table 3)", combined_labels,
                   palette=palette)

    fig.suptitle("HemBLIP — Run Comparison", fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(run_dir: Path, history: Optional[Dict],
                   eval_res: Optional[Dict]) -> None:
    print(f"\n-- {run_dir.name} --")
    if history:
        best_ep  = int(np.argmin(history["val_loss"])) + 1
        best_val = float(np.min(history["val_loss"]))
        final_ep = len(history["train_loss"])
        print(f"  Epochs trained : {final_ep}")
        print(f"  Best epoch     : {best_ep}  (val_loss={best_val:.4f})")
    else:
        print("  training_history.json not found")
    if eval_res:
        cm = (eval_res.get("caption_metrics") or {}).get("internal") or {}
        print(f"  BLEU-4         : {cm.get('bleu', 'N/A')}")
        print(f"  ROUGE-L        : {cm.get('rouge_l', 'N/A')}")
        print(f"  BERTScore F1   : {cm.get('bertscore_f1', 'N/A')}")
        for task, key in [("Leukemia subtype", "leukemia_subtype"), ("Cell type", "cell_type")]:
            td = (eval_res.get("classifier") or {}).get(key, {}).get("internal") or {}
            if td:
                print(f"  {task:<17}: acc={td.get('accuracy', 0):.3f}  "
                      f"f1={td.get('f1_weighted', 0):.3f}")
    else:
        print("  eval_results.json not found (run evaluate.py to add metric panels)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot HemBLIP training curves and evaluation metrics"
    )
    p.add_argument(
        "--runs", nargs="+", required=True,
        help="Output directories containing training_history.json "
             "(e.g. outputs/hemblip_lora)"
    )
    p.add_argument(
        "--checkpoints", nargs="*", default=None,
        help="Checkpoint dirs with eval_results.json, one per run. "
             "Defaults to <run>/best."
    )
    args = p.parse_args()

    run_dirs  = [Path(r) for r in args.runs]
    ckpt_dirs = (
        [Path(c) for c in args.checkpoints]
        if args.checkpoints
        else [r / "best" for r in run_dirs]
    )

    histories    = []
    eval_results = []
    for run_dir, ckpt_dir in zip(run_dirs, ckpt_dirs):
        hist = _load_json(run_dir / "training_history.json")
        ev   = _load_json(ckpt_dir / "eval_results.json")
        histories.append(hist)
        eval_results.append(ev)
        _print_summary(run_dir, hist, ev)

    print()

    # Per-run figures
    for run_dir, hist, ev in zip(run_dirs, histories, eval_results):
        if hist is None:
            print(f"  [skip] {run_dir.name} — no training_history.json")
            continue
        out = GRAPHICS_DIR / f"{run_dir.name}_curves.png"
        plot_single_run(run_dir, hist, ev, out)

    # Comparison figure (only when multiple runs)
    if len(run_dirs) > 1:
        valid = [(r, h, e) for r, h, e in zip(run_dirs, histories, eval_results)
                 if h is not None]
        if len(valid) > 1:
            r_list, h_list, e_list = zip(*valid)
            plot_comparison(
                list(r_list), list(h_list), list(e_list),
                GRAPHICS_DIR / "comparison.png",
            )

    print(f"\nAll figures saved in '{GRAPHICS_DIR}/'")


if __name__ == "__main__":
    main()
