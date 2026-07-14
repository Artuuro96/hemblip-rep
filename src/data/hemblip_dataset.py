"""
Combined HemBLIP dataset: wraps blood-cell images, attaches a natural-
language caption per image, and tokenizes everything for the BLIP
vision-to-text model.

Only the WBCATt/PBC dataset (``wbcatt_dir``) is wired up right now.
``leukemia_dir`` / ``external_dir`` are accepted for interface
compatibility with scripts/evaluate.py and scripts/run_pipeline.py, but
are silently skipped when the directory is absent or ``None`` — so
training works with just the PBC dataset.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.caption_templates import caption_from_attributes, generate_caption
from src.data.wbcatt_attributes import has_wbcatt_attributes, load_wbcatt_split
from src.data.wbcatt_dataset import scan_pbc_dataset

PAD_TOKEN_ID = 0  # BLIP's BERT-style tokenizer pad id


class HemBLIPDataset(Dataset):
    """One entry per image. Caption is chosen once at build time (seeded)
    so val/test captions stay fixed across runs."""

    def __init__(self, records: List[Dict], processor, max_length: int = 128) -> None:
        self.records = records
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        image = Image.open(record["image_path"]).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        tokenized = self.processor.tokenizer(
            record["caption"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
            "record": record,
        }


class HemBLIPCollator:
    """Pads a batch of tokenized samples and stacks pixel values. Masks
    padding positions in `labels` with -100 so they're ignored by the loss."""

    def __init__(self, pad_token_id: int = PAD_TOKEN_ID) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True,
            padding_value=0,
        )
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "records": [item["record"] for item in batch],
        }


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _stratified_split(
    records: List[Dict], val_split: float, test_split: float, rng: random.Random,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split records into train/val/test, preserving class proportions."""
    by_class: Dict[str, List[Dict]] = {}
    for r in records:
        by_class.setdefault(r["cell_type"], []).append(r)

    train, val, test = [], [], []
    for cls_records in by_class.values():
        shuffled = cls_records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = int(n * val_split)
        n_test = int(n * test_split)
        val.extend(shuffled[:n_val])
        test.extend(shuffled[n_val:n_val + n_test])
        train.extend(shuffled[n_val + n_test:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _label_records(records: List[Dict], cell_type_to_idx: Dict[str, int], rng: random.Random) -> None:
    for r in records:
        r["cell_type_idx"] = cell_type_to_idx.get(r["cell_type"], -1)
        r["diagnosis_idx"] = -1  # no leukemia labels available in this loader
        r["caption"] = generate_caption(r["cell_type"], rng)


def _label_records_from_attributes(records: List[Dict], cell_type_to_idx: Dict[str, int], rng: random.Random) -> None:
    for r in records:
        r["cell_type_idx"] = cell_type_to_idx.get(r["cell_type"], -1)
        r["diagnosis_idx"] = -1  # no leukemia labels available in this loader
        r["caption"] = caption_from_attributes(r, rng)


def _build_wbcatt_attribute_splits(
    wbcatt_dir: str, val_split: float, rng: random.Random,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Build train/val/test records from the official WBCAtt attribute CSVs
    (5 annotated classes only), with real per-image captions.

    Uses the official train/test split. WBCAtt ships no val split (or an
    empty one) in this download, so val is carved out of train.
    """
    train_records = load_wbcatt_split(wbcatt_dir, "train")
    val_records = load_wbcatt_split(wbcatt_dir, "val")
    test_records = load_wbcatt_split(wbcatt_dir, "test")

    cell_types = sorted({r["cell_type"] for r in train_records + val_records + test_records})
    cell_type_to_idx = {c: i for i, c in enumerate(cell_types)}
    for split_records in (train_records, val_records, test_records):
        _label_records_from_attributes(split_records, cell_type_to_idx, rng)

    if not val_records:
        train_records, val_records, _ = _stratified_split(train_records, val_split, 0.0, rng)

    return train_records, val_records, test_records


def build_datasets(
    wbcatt_dir: str,
    leukemia_dir: Optional[str],
    processor,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_length: int = 128,
    seed: int = 42,
    external_dir: Optional[str] = None,
) -> Tuple[HemBLIPDataset, HemBLIPDataset, HemBLIPDataset, Optional[HemBLIPDataset]]:
    """Build train/val/test (+ optional external) datasets.

    Only ``wbcatt_dir`` (the PBC_dataset_normal_DIB folder) is required.
    ``leukemia_dir`` is accepted for interface compatibility but is not
    yet supported by this loader — it's skipped with a warning if given.

    If the official WBCAtt attribute CSVs (``pbc_attr_v1_{train,test}.csv``)
    are present under ``wbcatt_dir``, they're used preferentially: real
    per-image captions built from the 11 annotated morphological
    attributes, restricted to the 5 classes WBCAtt covers (neutrophil,
    eosinophil, basophil, lymphocyte, monocyte), using the official
    train/test split. Otherwise falls back to a plain folder scan across
    all 8 classes with generic per-class template captions.
    """
    rng = random.Random(seed)

    if not wbcatt_dir or not Path(wbcatt_dir).exists():
        raise FileNotFoundError(f"wbcatt_dir not found: {wbcatt_dir}")

    if leukemia_dir and Path(leukemia_dir).exists():
        print(f"  [data] leukemia_dir is not supported by this loader yet — skipping: {leukemia_dir}")

    if has_wbcatt_attributes(wbcatt_dir):
        print("  [data] WBCAtt attribute CSVs found — using real per-image captions (5 classes)")
        train_records, val_records, test_records = _build_wbcatt_attribute_splits(wbcatt_dir, val_split, rng)
        cell_type_to_idx = {r["cell_type"]: r["cell_type_idx"] for r in train_records}
    else:
        records = scan_pbc_dataset(wbcatt_dir)
        if not records:
            raise FileNotFoundError(f"No usable images found under wbcatt_dir={wbcatt_dir}")
        cell_types = sorted({r["cell_type"] for r in records})
        cell_type_to_idx = {c: i for i, c in enumerate(cell_types)}
        _label_records(records, cell_type_to_idx, rng)
        train_records, val_records, test_records = _stratified_split(records, val_split, test_split, rng)

    train_ds = HemBLIPDataset(train_records, processor, max_length)
    val_ds = HemBLIPDataset(val_records, processor, max_length)
    test_ds = HemBLIPDataset(test_records, processor, max_length)

    external_ds = None
    if external_dir and Path(external_dir).exists():
        ext_records = scan_pbc_dataset(external_dir)
        _label_records(ext_records, cell_type_to_idx, rng)
        external_ds = HemBLIPDataset(ext_records, processor, max_length)

    print(
        f"  [data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        + (f" external={len(external_ds)}" if external_ds else "")
    )
    return train_ds, val_ds, test_ds, external_ds
