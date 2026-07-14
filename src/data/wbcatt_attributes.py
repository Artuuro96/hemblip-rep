"""
Loader for the official WBCAtt attribute CSVs (Tsutsui et al., NeurIPS 2023),
which annotate a ~9.3k-image subset of PBC_dataset_normal_DIB with 11
expert-derived morphological attributes per cell:

    pbc_attr_v1_train.csv
    pbc_attr_v1_val.csv    (may be empty — some WBCAtt releases ship no val split)
    pbc_attr_v1_test.csv

Columns: img_name, label, cell_size, cell_shape, nucleus_shape,
nuclear_cytoplasmic_ratio, chromatin_density, cytoplasm_vacuole,
cytoplasm_texture, cytoplasm_colour, granule_type, granule_colour,
granularity, path.

Only 5 of the 8 PBC_dataset_normal_DIB classes are covered by WBCAtt:
neutrophil, eosinophil, basophil, lymphocyte, monocyte.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from src.data.wbcatt_dataset import _resolve_class_root

ATTRIBUTE_COLUMNS = [
    "cell_size", "cell_shape", "nucleus_shape", "nuclear_cytoplasmic_ratio",
    "chromatin_density", "cytoplasm_vacuole", "cytoplasm_texture",
    "cytoplasm_colour", "granule_type", "granule_colour", "granularity",
]


def has_wbcatt_attributes(csv_dir: str) -> bool:
    train_csv = Path(csv_dir) / "pbc_attr_v1_train.csv"
    return train_csv.exists() and train_csv.stat().st_size > 0


def _load_csv(path: Path) -> List[Dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_wbcatt_split(csv_dir: str, split: str) -> List[Dict]:
    """Load one split (train/val/test) of the WBCAtt attribute CSVs and
    resolve each row to an on-disk image path.

    Returns a list of dicts: ``{"image_path", "cell_type", **attributes}``.
    Rows whose image file can't be found are skipped.
    """
    csv_dir_path = Path(csv_dir)
    class_root = _resolve_class_root(csv_dir_path)
    rows = _load_csv(csv_dir_path / f"pbc_attr_v1_{split}.csv")

    records: List[Dict] = []
    missing = 0
    for row in rows:
        cell_type = row["label"].strip().lower()
        image_path = class_root / cell_type / row["img_name"]
        if not image_path.exists():
            missing += 1
            continue
        record = {"image_path": str(image_path), "cell_type": cell_type}
        for col in ATTRIBUTE_COLUMNS:
            record[col] = row[col]
        records.append(record)

    if missing:
        print(f"  [wbcatt-attrs] {split}: {missing} rows referenced missing image files (skipped)")
    if rows:
        print(f"  [wbcatt-attrs] {split}: {len(records)} annotated images loaded")
    return records
