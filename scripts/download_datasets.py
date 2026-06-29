#!/usr/bin/env python3
"""
Dataset download helper.

The WBCATt and LeukemiaAttri datasets are not freely downloadable via direct URL
(both require a request / Kaggle account).  This script:

  1. Prints the exact steps to obtain each dataset.
  2. Verifies the expected directory structure once downloaded.
  3. Optionally downloads the Blood Cell Atlas (Kaggle) subset used for
     external validation.

Usage:
    python scripts/download_datasets.py --verify          # check structure only
    python scripts/download_datasets.py --kaggle-external # download Kaggle subset
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


DATA_ROOT = Path("data")

WBCATT_DIR = DATA_ROOT / "wbcatt"
LEUKEMIA_DIR = DATA_ROOT / "leukemia"
EXTERNAL_DIR = DATA_ROOT / "external"

DOWNLOAD_INSTRUCTIONS = """
==============================================================
  Dataset Download Instructions
==============================================================

1. WBCATt  (healthy WBCs, NeurIPS 2023)
   ─────────────────────────────────────
   Homepage : https://github.com/apple2373/wbcatt
   Steps    :
     a) Follow the download instructions in the README.
     b) Download WBCATt.zip and unzip.
     c) Move the unzipped contents to:  data/wbcatt/
        Expected structure:
          data/wbcatt/
            images/
              neutrophil/  eosinophil/  basophil/
              lymphocyte/  monocyte/
            annotations.json   ← attribute labels per image

2. LeukemiaAttri  (leukemic WBCs, arXiv:2405.10803)
   ─────────────────────────────────────────────────
   Kaggle  : https://www.kaggle.com/datasets/andrewblayamastephen/leukemiaattri-dataset
   Steps   :
     a) Run: python scripts/download_datasets.py --kaggle-leukemia
        (requires kaggle CLI + ~/.kaggle/kaggle.json credentials)
     OR manually download and unzip to data/leukemia/
        Expected structure:
          data/leukemia/
            images/
              ALL/  AML/  APML/  CLL/  CML/  Healthy/
            annotations.json  (or CSV with morphological attribute labels)

3. External validation  (Blood Cell Atlas + Kaggle)
   ──────────────────────────────────────────────────
   Blood Cell Atlas: https://www.mindray.com/en/index.html
     (cited as [25] in the paper; request from Mindray or use equivalent)

   Kaggle collection: "Blood cell images for cancer detection"
     https://www.kaggle.com/datasets/sumithkothwal/blood-cell-images
     Run:  kaggle datasets download sumithkothwal/blood-cell-images
     Unzip to: data/external/images/

==============================================================
"""


def verify_wbcatt() -> bool:
    if not WBCATT_DIR.exists():
        print(f"  [MISSING] {WBCATT_DIR}")
        return False
    img_dir = WBCATT_DIR / "images"
    ann = WBCATT_DIR / "annotations.json"
    ok = True
    if not img_dir.exists():
        print(f"  [MISSING] {img_dir}")
        ok = False
    else:
        n_imgs = sum(1 for _ in img_dir.rglob("*.jpg")) + sum(1 for _ in img_dir.rglob("*.png"))
        print(f"  [OK] {img_dir}  ({n_imgs} images)")
    if not ann.exists():
        alt_csv = WBCATT_DIR / "annotations.csv"
        if alt_csv.exists():
            print(f"  [OK] {alt_csv}  (CSV format)")
        else:
            print(f"  [MISSING] {ann}  (annotations.json or annotations.csv required)")
            ok = False
    else:
        print(f"  [OK] {ann}")
    return ok


def verify_leukemia() -> bool:
    if not LEUKEMIA_DIR.exists():
        print(f"  [MISSING] {LEUKEMIA_DIR}")
        return False
    img_dir = LEUKEMIA_DIR / "images"
    ann = LEUKEMIA_DIR / "annotations.json"
    ok = True
    if not img_dir.exists():
        print(f"  [MISSING] {img_dir}")
        ok = False
    else:
        subtypes = [d.name for d in img_dir.iterdir() if d.is_dir()]
        n_imgs = sum(1 for _ in img_dir.rglob("*.jpg")) + sum(1 for _ in img_dir.rglob("*.png"))
        print(f"  [OK] {img_dir}  ({n_imgs} images, subtypes: {subtypes})")
    if not ann.exists():
        alt_csv = LEUKEMIA_DIR / "annotations.csv"
        if alt_csv.exists():
            print(f"  [OK] {alt_csv}  (CSV format)")
        else:
            print(f"  [MISSING] {ann}")
            ok = False
    else:
        print(f"  [OK] {ann}")
    return ok


def verify_external() -> bool:
    ext_img = EXTERNAL_DIR / "images"
    if not ext_img.exists():
        print(f"  [OPTIONAL, MISSING] {ext_img}  (external validation not available)")
        return True
    n_imgs = sum(1 for _ in ext_img.rglob("*.jpg")) + sum(1 for _ in ext_img.rglob("*.png"))
    print(f"  [OK] {ext_img}  ({n_imgs} images)")
    return True


def _kaggle_download(dataset_id: str, dest_dir: Path, subfolder: str) -> bool:
    """Generic Kaggle dataset downloader. Returns True on success."""
    import subprocess
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dataset_id} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", dataset_id,
         "--path", str(dest_dir), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Error: {result.stderr.strip()}")
        print("  Install kaggle CLI: pip install kaggle")
        print("  Set up credentials: ~/.kaggle/kaggle.json")
        return False
    print(f"  Downloaded to {dest_dir}")
    # Flatten if Kaggle created a subfolder
    nested = dest_dir / subfolder
    if nested.exists() and not (dest_dir / "images").exists():
        nested.rename(dest_dir / "images")
        print(f"  Moved to {dest_dir / 'images'}")
    return True


def try_download_leukemia():
    """Download LeukemiaAttri from Kaggle."""
    try:
        _kaggle_download(
            "andrewblayamastephen/leukemiaattri-dataset",
            LEUKEMIA_DIR,
            subfolder="leukemiaattri-dataset",
        )
    except FileNotFoundError:
        print("  kaggle CLI not found. Install with: pip install kaggle")


def try_download_kaggle_external():
    """Download the Blood Cell Images external subset from Kaggle."""
    try:
        _kaggle_download(
            "sumithkothwal/blood-cell-images",
            EXTERNAL_DIR,
            subfolder="blood-cell-images",
        )
    except FileNotFoundError:
        print("  kaggle CLI not found. Install with: pip install kaggle")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="Check directory structure")
    p.add_argument("--kaggle-leukemia", action="store_true",
                   help="Download LeukemiaAttri from Kaggle")
    p.add_argument("--kaggle-external", action="store_true",
                   help="Download Blood Cell Images external subset from Kaggle")
    args = p.parse_args()

    print(DOWNLOAD_INSTRUCTIONS)

    if args.kaggle_leukemia:
        try_download_leukemia()

    if args.kaggle_external:
        try_download_kaggle_external()

    if args.verify or (not args.kaggle_leukemia and not args.kaggle_external):
        print("Verifying dataset structure:")
        wbc_ok = verify_wbcatt()
        leu_ok = verify_leukemia()
        ext_ok = verify_external()

        if wbc_ok and leu_ok:
            print("\nAll required datasets found. Ready to train!")
        else:
            print("\nSome datasets are missing. Follow the instructions above to download them.")


if __name__ == "__main__":
    main()
