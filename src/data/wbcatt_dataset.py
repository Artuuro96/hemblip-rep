"""
Loader for the PBC normal peripheral blood cell dataset (WBCATt-style),
organised as one sub-folder per cell type::

    <root>/
      basophil/*.jpg
      eosinophil/*.jpg
      erythroblast/*.jpg
      ig/*.jpg
      lymphocyte/*.jpg
      monocyte/*.jpg
      neutrophil/*.jpg
      platelet/*.jpg

Some zip exports nest the class folders one level deeper
(``<root>/<root_name>/<class>/*.jpg``) — ``_resolve_class_root`` auto-detects
and descends into that extra layer so either layout works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image, UnidentifiedImageError

CELL_TYPES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _resolve_class_root(root: Path) -> Path:
    """Descend into a single self-named subdirectory if the class folders
    aren't directly under `root` (common with zip exports, which may also
    leave a junk `__MACOSX` folder alongside the real one)."""
    if any((root / c).is_dir() for c in CELL_TYPES):
        return root
    subdirs = [
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__MACOSX"
    ] if root.is_dir() else []
    if len(subdirs) == 1:
        return _resolve_class_root(subdirs[0])
    return root


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def scan_pbc_dataset(root_dir: str) -> List[Dict]:
    """Scan a PBC_dataset_normal_DIB-style directory and return one record
    per valid image: ``{"image_path": str, "cell_type": str}``.

    Unreadable/corrupt files (e.g. stray macOS ``.`` metadata files that
    survive a zip export) are skipped rather than crashing the DataLoader.
    """
    root = _resolve_class_root(Path(root_dir))
    records: List[Dict] = []
    skipped = 0

    for cell_type in CELL_TYPES:
        class_dir = root / cell_type
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.name.startswith("."):
                skipped += 1
                continue
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            if not _is_valid_image(path):
                skipped += 1
                continue
            records.append({"image_path": str(path), "cell_type": cell_type})

    if skipped:
        print(f"  [wbcatt] Skipped {skipped} unreadable/hidden files")
    print(f"  [wbcatt] Loaded {len(records)} images from {root}")
    return records
