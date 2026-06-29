#!/usr/bin/env python3
"""
HemBLIP end-to-end pipeline: train -> evaluate -> plot.
Designed to be left running overnight.

For each config it runs:
  1. Training   (saves best checkpoint + training_history.json)
  2. Evaluation (saves eval_results.json — Tables 1, 2, 3)
  3. Plotting   (saves graphics/<run>_curves.png)

At the end a comparison plot is generated if more than one config was run,
and a plain-text summary is written to graphics/pipeline_summary_<ts>.txt.

Usage
-----
  # LoRA only (recommended first run — ~8-12 h on RTX 3070)
  py scripts/run_pipeline.py

  # LoRA + full fine-tune overnight
  py scripts/run_pipeline.py --configs configs/hemblip_lora.yaml configs/hemblip_full.yaml

  # GPU + paper-accurate BERTScore (roberta-large, slower)
  py scripts/run_pipeline.py --device cuda --bertscore-model roberta-large

  # Skip evaluation / plot if you only want to train
  py scripts/run_pipeline.py --skip-eval
  py scripts/run_pipeline.py --skip-eval --skip-plot
"""

# ── stdlib only at module level — keeps --help instant ────────────────────────
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GRAPHICS_DIR = Path("graphics")


# ─────────────────────────────────────────────────────────────────────────────
# Path bootstrap  (called once inside main, not at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_paths() -> None:
    """
    Add project root to sys.path and remove scripts/ so that
    'import evaluate' finds the HuggingFace package, not scripts/evaluate.py.
    """
    project_root = str(Path(__file__).parent.parent.resolve())
    scripts_dir  = str(Path(__file__).parent.resolve())

    # Remove scripts/ if Python auto-added it (happens when running a .py file)
    while scripts_dir in sys.path:
        sys.path.remove(scripts_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("hemblip_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def _fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(cfg: Dict, device: str, logger: logging.Logger) -> str:
    import math, random as _random
    import numpy as np
    import torch
    from src.data.hemblip_dataset import build_datasets
    from src.models.hemblip import build_hemblip
    from src.training.trainer import HemBLIPTrainer

    seed = cfg["training"].get("seed", 42)
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Building model ...")
    model, processor = build_hemblip(
        base_model=cfg["model"]["base_model"],
        lora=cfg["model"].get("lora", False),
        lora_r=cfg["model"].get("lora_r", 16),
        lora_alpha=cfg["model"].get("lora_alpha", 32),
        lora_dropout=cfg["model"].get("lora_dropout", 0.05),
        lora_target_modules=cfg["model"].get("lora_target_modules"),
    )

    logger.info("Loading datasets ...")
    train_ds, val_ds, test_ds, _ = build_datasets(
        wbcatt_dir=cfg["data"]["wbcatt_dir"],
        leukemia_dir=cfg["data"]["leukemia_dir"],
        processor=processor,
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        max_length=cfg["data"]["max_caption_length"],
        seed=seed,
        external_dir=cfg["data"].get("external_dir"),
    )
    logger.info(
        "Splits — train: %d | val: %d | test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    trainer = HemBLIPTrainer(
        model=model, processor=processor,
        train_ds=train_ds, val_ds=val_ds,
        output_dir=cfg["training"]["output_dir"],
        num_epochs=cfg["training"]["num_epochs"],
        batch_size=cfg["training"]["batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
        warmup_steps=cfg["training"].get("warmup_steps", 200),
        patience=cfg["training"].get("patience", 5),
        fp16=cfg["training"].get("fp16", True),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 2),
        num_workers=cfg["training"].get("num_workers", 2),
        compile_model=cfg["training"].get("compile", False),
        device=device,
    )

    logger.info(
        "Training — max %d epochs | patience=%d | effective batch=%d",
        cfg["training"]["num_epochs"],
        cfg["training"].get("patience", 5),
        cfg["training"]["batch_size"] * cfg["training"].get("gradient_accumulation_steps", 2),
    )
    best_ckpt = trainer.train()
    logger.info("Training complete. Best checkpoint: %s", best_ckpt)
    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    cfg: Dict,
    checkpoint: str,
    device: str,
    bertscore_model: str,
    logger: logging.Logger,
) -> Dict:
    import torch
    from torch.utils.data import DataLoader
    from src.data.hemblip_dataset import HemBLIPCollator, build_datasets
    from src.evaluation.attribute_extractor import compute_attribute_accuracy
    from src.evaluation.metrics import compute_all_metrics, generate_predictions
    from src.models.classifier import run_classifier_evaluation
    from src.models.hemblip import build_hemblip

    seed = cfg["training"].get("seed", 42)

    logger.info("Loading checkpoint: %s", checkpoint)
    model, processor = build_hemblip(
        base_model=cfg["model"]["base_model"],
        lora=cfg["model"].get("lora", False),
        lora_r=cfg["model"].get("lora_r", 16),
        lora_alpha=cfg["model"].get("lora_alpha", 32),
        lora_dropout=cfg["model"].get("lora_dropout", 0.05),
        lora_target_modules=cfg["model"].get("lora_target_modules"),
        checkpoint=checkpoint,
    )
    model.to(device)

    logger.info("Loading datasets for evaluation ...")
    train_ds, val_ds, test_ds, _ = build_datasets(
        wbcatt_dir=cfg["data"]["wbcatt_dir"],
        leukemia_dir=cfg["data"]["leukemia_dir"],
        processor=processor,
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        max_length=cfg["data"]["max_caption_length"],
        seed=seed,
    )

    collator   = HemBLIPCollator()
    eval_bs    = cfg["evaluation"].get("batch_size", 32)
    max_tokens = cfg["evaluation"].get("max_new_tokens", 128)
    num_beams  = cfg["evaluation"].get("num_beams", 4)

    test_loader      = DataLoader(test_ds,  batch_size=eval_bs, shuffle=False, collate_fn=collator)
    train_loader_clf = DataLoader(train_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator)

    # Table 1
    logger.info("Generating captions on test set ...")
    preds, refs = generate_predictions(model, processor, test_loader, device, max_tokens, num_beams)
    metrics = compute_all_metrics(preds, refs, bertscore_model, device)
    logger.info(
        "BLEU=%.4f  ROUGE-L=%.4f  BERTScore=%.4f",
        metrics["bleu"], metrics["rouge_l"], metrics["bertscore_f1"],
    )

    # Table 2
    attr_acc = compute_attribute_accuracy(preds, refs)
    logger.info("Attribute accuracy: %s",
                {k: f"{v:.3f}" if v == v else "N/A" for k, v in attr_acc.items()})

    # Table 3
    logger.info("Frozen-backbone classifier ...")
    clf = run_classifier_evaluation(model=model, train_loader=train_loader_clf,
                                    test_loader=test_loader, device=device)
    for task, key in [("Leukemia subtype", "leukemia_subtype"), ("Cell type", "cell_type")]:
        m = (clf.get(key) or {}).get("internal") or {}
        if m:
            logger.info("%s — acc=%.3f  f1=%.3f", task, m["accuracy"], m["f1_weighted"])

    results = {
        "model": cfg["model"]["name"],
        "checkpoint": checkpoint,
        "caption_metrics": {"internal": metrics, "external": None},
        "attribute_accuracy": {
            "internal": {k: (None if v != v else v) for k, v in attr_acc.items()},
            "external": None,
        },
        "classifier": clf,
    }
    out = Path(checkpoint) / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved -> %s", out)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _load_plot_fns():
    """Lazy-load plot_training module via importlib to keep scripts/ off sys.path."""
    import importlib.util
    import matplotlib
    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location(
        "_plot_training",
        Path(__file__).parent / "plot_training.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_run, mod.plot_single_run, mod.plot_comparison


def run_plotting(run_dir: Path, logger: logging.Logger) -> None:
    try:
        load_run, plot_single_run, _ = _load_plot_fns()
    except Exception as e:
        logger.warning("Plotting unavailable: %s", e)
        return

    history, eval_res = load_run(run_dir)
    if history is None:
        logger.warning("No training_history.json in %s", run_dir)
        return

    out = GRAPHICS_DIR / f"{run_dir.name}_curves.png"
    plot_single_run(run_dir, history, eval_res, out)
    logger.info("Plot saved -> %s", out)


def run_comparison(run_dirs: List[Path], logger: logging.Logger) -> None:
    try:
        load_run, _, plot_comparison = _load_plot_fns()
    except Exception as e:
        logger.warning("Plotting unavailable: %s", e)
        return

    data = [(r, *load_run(r)) for r in run_dirs if r.exists()]
    valid = [(r, h, e) for r, h, e in data if h is not None]
    if len(valid) < 2:
        return
    r_list, h_list, e_list = zip(*valid)
    out = GRAPHICS_DIR / "comparison.png"
    plot_comparison(list(r_list), list(h_list), list(e_list), out)
    logger.info("Comparison plot saved -> %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def _write_summary(
    records: List[Dict],
    elapsed: List[float],
    total: float,
    out_path: Path,
) -> None:
    lines = [
        "HemBLIP Pipeline Summary",
        "=" * 52,
        f"Completed : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total time: {_fmt(total)}",
        "",
    ]
    for rec, el in zip(records, elapsed):
        lines += [
            f"Config  : {rec['config']}",
            f"  Status  : {rec['status']}",
            f"  Runtime : {_fmt(el)}",
        ]
        if rec.get("checkpoint"):
            lines.append(f"  Checkpoint: {rec['checkpoint']}")
        ev = rec.get("eval")
        if ev:
            cm = (ev.get("caption_metrics") or {}).get("internal") or {}
            lines += [
                f"  BLEU-4      : {cm.get('bleu', 'N/A')}",
                f"  ROUGE-L     : {cm.get('rouge_l', 'N/A')}",
                f"  BERTScore F1: {cm.get('bertscore_f1', 'N/A')}",
            ]
            for task, key in [("Leukemia subtype", "leukemia_subtype"),
                               ("Cell type",        "cell_type")]:
                m = (ev.get("classifier") or {}).get(key, {}).get("internal") or {}
                if m:
                    lines.append(
                        f"  {task:<18}: acc={m['accuracy']:.3f}  f1={m['f1_weighted']:.3f}"
                    )
        if rec.get("error"):
            lines.append(f"  Error: {rec['error']}")
        lines.append("")

    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print("\n" + text)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HemBLIP pipeline: train -> evaluate -> plot (leave overnight)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py scripts/run_pipeline.py
  py scripts/run_pipeline.py --device cuda
  py scripts/run_pipeline.py --configs configs/hemblip_lora.yaml configs/hemblip_full.yaml
  py scripts/run_pipeline.py --bertscore-model roberta-large
        """,
    )
    p.add_argument(
        "--configs", nargs="+",
        default=["configs/hemblip_lora.yaml"],
        help="YAML config files to run in sequence (default: hemblip_lora.yaml)",
    )
    p.add_argument(
        "--device", default=None,
        help="cuda | mps | cpu  (auto-detected if omitted)",
    )
    p.add_argument(
        "--bertscore-model", default="distilbert-base-uncased",
        dest="bertscore_model",
        help="BERTScore backbone. Use roberta-large for paper-accurate numbers (slower).",
    )
    p.add_argument("--skip-eval",  action="store_true", help="Skip evaluation step")
    p.add_argument("--skip-plot",  action="store_true", help="Skip plotting step")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Bootstrap paths AFTER argparse (so --help never loads torch)
    _bootstrap_paths()

    # Now it is safe to import heavy libs
    import torch
    import yaml
    from src.training.trainer import _auto_device

    device = args.device or _auto_device()
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = GRAPHICS_DIR / f"pipeline_{ts}.log"
    logger   = _setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("HemBLIP Pipeline  --  %s", ts)
    logger.info("Device     : %s", device)
    logger.info("Configs    : %s", args.configs)
    logger.info("BERTScore  : %s", args.bertscore_model)
    logger.info("Log file   : %s", log_path)
    logger.info("=" * 60)

    if device == "cuda":
        logger.info(
            "GPU : %s  |  VRAM : %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    total_start  = time.time()
    records: List[Dict] = []
    elapsed: List[float] = []
    run_dirs: List[Path] = []

    for cfg_path_str in args.configs:
        t0     = time.time()
        cfg_p  = Path(cfg_path_str)
        record: Dict = {"config": cfg_path_str, "status": "pending",
                        "checkpoint": None, "eval": None, "error": None}

        logger.info("")
        logger.info("=" * 60)
        logger.info("CONFIG: %s", cfg_path_str)
        logger.info("=" * 60)

        try:
            with open(cfg_p) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            record["status"] = "failed"
            record["error"]  = f"Config not found: {cfg_p}"
            logger.error(record["error"])
            records.append(record); elapsed.append(time.time() - t0)
            continue

        run_dir = Path(cfg["training"]["output_dir"])
        run_dirs.append(run_dir)

        # ── 1. Training ───────────────────────────────────────────────────────
        try:
            logger.info("[1/3] TRAINING")
            t1 = time.time()
            ckpt = run_training(cfg, device, logger)
            record["checkpoint"] = ckpt
            logger.info("[1/3] Done in %s", _fmt(time.time() - t1))
        except Exception as exc:
            record.update(status="training_failed", error=str(exc))
            logger.exception("[1/3] Training failed: %s", exc)
            records.append(record); elapsed.append(time.time() - t0)
            continue

        # ── 2. Evaluation ─────────────────────────────────────────────────────
        if not args.skip_eval:
            try:
                logger.info("")
                logger.info("[2/3] EVALUATION")
                t1 = time.time()
                record["eval"] = run_evaluation(cfg, ckpt, device,
                                                args.bertscore_model, logger)
                logger.info("[2/3] Done in %s", _fmt(time.time() - t1))
            except Exception as exc:
                logger.exception("[2/3] Evaluation failed (continuing): %s", exc)
        else:
            logger.info("[2/3] Evaluation skipped (--skip-eval)")

        # ── 3. Plotting ───────────────────────────────────────────────────────
        if not args.skip_plot:
            try:
                logger.info("")
                logger.info("[3/3] PLOTTING")
                t1 = time.time()
                run_plotting(run_dir, logger)
                logger.info("[3/3] Done in %s", _fmt(time.time() - t1))
            except Exception as exc:
                logger.exception("[3/3] Plotting failed: %s", exc)
        else:
            logger.info("[3/3] Plotting skipped (--skip-plot)")

        record["status"] = "success"
        elapsed.append(time.time() - t0)
        logger.info("Config finished in %s", _fmt(elapsed[-1]))
        records.append(record)

    # ── Comparison plot ───────────────────────────────────────────────────────
    if not args.skip_plot and len(run_dirs) > 1:
        try:
            logger.info("")
            logger.info("Generating comparison plot ...")
            run_comparison(run_dirs, logger)
        except Exception as exc:
            logger.exception("Comparison plot failed: %s", exc)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE -- total time: %s", _fmt(total_elapsed))
    logger.info("=" * 60)

    summary_path = GRAPHICS_DIR / f"pipeline_summary_{ts}.txt"
    _write_summary(records, elapsed, total_elapsed, summary_path)
    logger.info("Summary saved -> %s", summary_path)


if __name__ == "__main__":
    main()
