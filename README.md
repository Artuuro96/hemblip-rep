# HemBLIP — Replication

Python replication of **"HemBLIP: A Vision–Language Model for Interpretable Leukemia Cell Morphology Analysis"** (van Logtestijn & Manescu, arXiv:2601.03915, 2026).

## Overview

| Component | Paper | This repo |
|---|---|---|
| Base VLM | BLIP (Salesforce) | `Salesforce/blip-image-captioning-base` |
| Fine-tuning | Full + LoRA on decoder attention | `src/models/hemblip.py` |
| Datasets | WBCATt (10k healthy) + LeukemiaAttri (10k leukemic) | `src/data/` |
| Captions | GPT-4-augmented templates | `src/data/caption_templates.py` |
| Metrics | BLEU-4, ROUGE-L, BERTScore F1 | `src/evaluation/metrics.py` |
| Attribute eval | Regex morphological extractor | `src/evaluation/attribute_extractor.py` |
| Classifier probe | Frozen-backbone cosine-sim head (Table 3) | `src/models/classifier.py` |

## Project Structure

```
hemblip/
├── configs/
│   ├── hemblip_full.yaml     # Full fine-tuning config
│   ├── hemblip_lora.yaml     # LoRA config (recommended)
│   └── medgemma_lora.yaml    # MedGEMMA comparison
├── data/
│   ├── wbcatt/               # WBCATt dataset (see below)
│   ├── leukemia/             # LeukemiaAttri dataset
│   └── external/             # Blood Cell Atlas + Kaggle subset
├── scripts/
│   ├── train_hemblip.py      # Main training entry point
│   ├── evaluate.py           # Full evaluation (Tables 1–3)
│   ├── generate_synthetic_data.py   # Smoke-test without real data
│   └── download_datasets.py  # Download instructions + verification
├── src/
│   ├── data/
│   │   ├── caption_templates.py   # Attribute → natural language
│   │   ├── wbcatt_dataset.py      # WBCATt loader
│   │   ├── leukemia_dataset.py    # LeukemiaAttri loader
│   │   └── hemblip_dataset.py     # Combined Dataset + Collator
│   ├── models/
│   │   ├── hemblip.py         # Model factory (full / LoRA)
│   │   └── classifier.py      # Frozen-backbone probe
│   ├── training/
│   │   └── trainer.py         # Training loop, early stopping
│   └── evaluation/
│       ├── metrics.py         # BLEU, ROUGE-L, BERTScore
│       └── attribute_extractor.py  # Regex attribute accuracy
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick smoke-test (no dataset download needed)

```bash
# Generate 200 synthetic images (20 per class)
python scripts/generate_synthetic_data.py --n 20

# Train HemBLIP LoRA for 2 epochs (fast CPU test)
python scripts/train_hemblip.py --config configs/hemblip_lora.yaml
```

## Dataset Download

```bash
# See full instructions
python scripts/download_datasets.py

# Verify structure after downloading
python scripts/download_datasets.py --verify

# Optional: download Kaggle external subset
python scripts/download_datasets.py --kaggle-external
```

### WBCATt (healthy WBCs)

**Download:** https://github.com/apple2373/wbcatt

Unzip and place contents at `data/wbcatt/`. Expected structure:
```
data/wbcatt/
└── PBC_dataset_normal_DIB/
    ├── neutrophil/
    ├── eosinophil/
    ├── basophil/
    ├── lymphocyte/
    ├── monocyte/
    ├── erythroblast/
    ├── ig/
    └── platelet/
```

### LeukemiaAttri (leukemic WBCs)

**Download:** https://www.kaggle.com/datasets/andrewblayamastephen/leukemiaattri-dataset

```bash
# Automatic (requires Kaggle CLI + credentials)
python scripts/download_datasets.py --kaggle-leukemia

# Or download manually and unzip to data/leukemia/
```

Expected structure:
```
data/leukemia/
└── archive/
    └── Original/
        ├── Benign/    ← healthy lymphocytes
        ├── Early/     ← ALL-L1
        ├── Pre/       ← ALL-L2
        └── Pro/       ← ALL-L3
```

## Training

```bash
# LoRA (recommended — ~0.85% trainable params)
python scripts/train_hemblip.py --config configs/hemblip_lora.yaml

# Full fine-tuning
python scripts/train_hemblip.py --config configs/hemblip_full.yaml

# With GPU
python scripts/train_hemblip.py --config configs/hemblip_lora.yaml --device cuda

# Resume from checkpoint
python scripts/train_hemblip.py --config configs/hemblip_lora.yaml \
    --resume outputs/hemblip_lora/best
```

Key hyperparameters (paper §2.2):
- Optimizer: AdamW, lr = 5×10⁻⁵
- Early stopping on validation loss (patience = 5 epochs)
- LoRA: r=16, α=32, target modules = {query, key, value, dense}

## Evaluation

```bash
# Internal test set (Tables 1 + 2 + 3)
python scripts/evaluate.py \
    --config configs/hemblip_lora.yaml \
    --checkpoint outputs/hemblip_lora/best

# With external validation set
python scripts/evaluate.py \
    --config configs/hemblip_lora.yaml \
    --checkpoint outputs/hemblip_lora/best \
    --external

# Paper-accurate BERTScore (slower)
python scripts/evaluate.py ... --bertscore_model roberta-large
```

Results are saved to `<checkpoint>/eval_results.json`.

## Paper Results (Table 1 reference)

| Model | BLEU (Int.) | ROUGE-L (Int.) | BERTScore (Int.) |
|---|---|---|---|
| HemBLIP Full | 0.24 | 0.42 | 0.83 |
| HemBLIP LoRA | 0.27 | 0.49 | 0.86 |
| MedGEMMA Base | 0.02 | 0.13 | 0.74 |
| MedGEMMA LoRA | **0.31** | **0.52** | **0.87** |

## MedGEMMA comparison

Edit `configs/medgemma_lora.yaml` and run:
```bash
python scripts/train_hemblip.py --config configs/medgemma_lora.yaml
```
Note: MedGEMMA-4B requires a HuggingFace token and ~16 GB VRAM.

## Reference

```bibtex
@article{vanlogtestijn2026hemblip,
  title   = {HemBLIP: A Vision--Language Model for Interpretable Leukemia Cell Morphology Analysis},
  author  = {van Logtestijn, Julie and Manescu, Petru},
  journal = {arXiv preprint arXiv:2601.03915},
  year    = {2026}
}
```
