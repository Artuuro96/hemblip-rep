"""
Caption generation for the PBC normal peripheral blood cell dataset
(Acevedo et al.), keyed by cell type.

Two paths, depending on what's available for a given image:

  - `caption_from_attributes`: when the official WBCAtt attribute CSVs are
    present (see src/data/wbcatt_attributes.py), builds a real per-image
    caption from the 11 expert-annotated morphological attributes — this
    is the paper-faithful path.
  - `generate_caption`: fallback for images with no attribute annotation
    (e.g. classes WBCAtt doesn't cover) — samples from a small bank of
    hand-written, class-level template descriptions.

Phrasing intentionally reuses vocabulary understood by
`src/evaluation/attribute_extractor.py` (e.g. "coarse chromatin",
"multi-lobulated nucleus", "small cell size") so downstream attribute
accuracy / caption-metric evaluation stays meaningful.
"""

from __future__ import annotations

import random
from typing import Dict, List

CAPTION_TEMPLATES: Dict[str, List[str]] = {
    "neutrophil": [
        "A neutrophil with a multi-lobulated nucleus, coarse chromatin, "
        "and moderate cytoplasm containing fine pink-lilac granules.",
        "This is a mature neutrophil showing a segmented, multi-lobulated "
        "nucleus with condensed chromatin and pale pink cytoplasm of "
        "medium cell size.",
        "A round neutrophil cell with a multilobulated nucleus and "
        "moderate cytoplasm, typical of the granulocyte lineage.",
    ],
    "eosinophil": [
        "An eosinophil with a bilobed nucleus and abundant cytoplasm "
        "packed with large orange-red granules, medium cell size.",
        "This is an eosinophil showing a bilobed nucleus, coarse "
        "chromatin, and abundant cytoplasm filled with coarse "
        "eosinophilic granules.",
        "A round eosinophil cell with two nuclear lobes and abundant "
        "cytoplasm densely covered in refractile orange granules.",
    ],
    "basophil": [
        "A basophil with an irregular nucleus largely obscured by "
        "abundant coarse dark-purple granules in the cytoplasm.",
        "This is a basophil showing a bilobed nucleus and moderate "
        "cytoplasm overlaid with large basophilic granules.",
        "A round basophil cell of medium cell size with coarse chromatin "
        "and cytoplasm crowded with deep purple granules.",
    ],
    "lymphocyte": [
        "A lymphocyte with a round nucleus occupying most of the cell, "
        "dense clumped chromatin, and scant sky-blue cytoplasm.",
        "This is a small lymphocyte showing a round nucleus with coarse "
        "chromatin and a thin rim of scant cytoplasm, small cell size.",
        "A round lymphocyte cell with a large round nucleus and scant "
        "cytoplasm, no visible granules.",
    ],
    "monocyte": [
        "A monocyte with a kidney-shaped, indented nucleus, fine "
        "chromatin, and abundant blue-gray cytoplasm, large cell size.",
        "This is a large monocyte showing a horseshoe-shaped nucleus and "
        "abundant cytoplasm with fine vacuoles.",
        "An irregular monocyte cell with an indented nucleus, open fine "
        "chromatin, and abundant grayish cytoplasm.",
    ],
    "ig": [
        "An immature granulocyte with a band-shaped, indented nucleus, "
        "moderately coarse chromatin, and moderate granular cytoplasm.",
        "This is an immature granulocyte (metamyelocyte/myelocyte stage) "
        "showing a less-segmented nucleus and moderate cytoplasm with "
        "primary and secondary granules.",
        "An oval immature granulocyte cell with fine to moderate "
        "chromatin and moderate cytoplasm, precursor to the neutrophil.",
    ],
    "erythroblast": [
        "An erythroblast with a small round nucleus showing dense "
        "clumped chromatin and scant deeply basophilic cytoplasm.",
        "This is a nucleated red blood cell precursor (erythroblast) "
        "with condensed round nucleus and scant blue cytoplasm, small "
        "cell size.",
        "A round erythroblast cell with a dense round nucleus and no "
        "cytoplasmic granules, scant cytoplasm.",
    ],
    "platelet": [
        "A platelet, a small anucleate cytoplasmic fragment with fine "
        "purple granules scattered inside, small cell size.",
        "This is a platelet showing an irregular shape, no nucleus, and "
        "scattered fine azurophilic granules.",
        "A small irregular platelet fragment with scant granular "
        "cytoplasm and no visible nucleus.",
    ],
}


def generate_caption(cell_type: str, rng: random.Random) -> str:
    """Return one randomly sampled caption for the given cell type."""
    templates = CAPTION_TEMPLATES.get(cell_type)
    if not templates:
        return f"A {cell_type} cell from a peripheral blood smear."
    return rng.choice(templates)


# ---------------------------------------------------------------------------
# Real per-image captions from WBCAtt's 11 annotated attributes
# ---------------------------------------------------------------------------

_NUCLEUS_SHAPE_PHRASES = {
    "unsegmented-band": "a band-shaped, unsegmented nucleus",
    "unsegmented-round": "a round, unsegmented nucleus",
    "unsegmented-indented": "an indented, unsegmented nucleus",
    "segmented-bilobed": "a bilobed nucleus",
    "segmented-multilobed": "a multi-lobulated nucleus",
    "irregular": "an irregular nucleus",
}

# "Open"/"loose" chromatin and "dense" chromatin are standard hematology
# terms for the two ends of this spectrum.
_CHROMATIN_PHRASES = {
    "densely": "densely packed chromatin",
    "loosely": "open, loosely packed chromatin",
}

# nuclear_cytoplasmic_ratio is inversely related to cytoplasm amount: a low
# N:C ratio means the cytoplasm takes up most of the cell (abundant), a high
# ratio means the nucleus dominates (scant cytoplasm).
_CYTOPLASM_AMOUNT_PHRASES = {
    "low": "abundant cytoplasm",
    "high": "scant cytoplasm",
}

_CELL_SIZE_PHRASES = {"big": "large cell size", "small": "small cell size"}

_CELL_SHAPE_PHRASES = {
    "round": "round overall cell shape",
    "irregular": "irregular overall cell shape",
}


def caption_from_attributes(attrs: Dict[str, str], rng: random.Random) -> str:
    """Build a real per-image caption from WBCAtt's 11 annotated
    morphological attributes (see src/data/wbcatt_attributes.py)."""
    cell_type = attrs["cell_type"]

    nucleus = _NUCLEUS_SHAPE_PHRASES.get(attrs["nucleus_shape"], f"a {attrs['nucleus_shape']} nucleus")
    chromatin = _CHROMATIN_PHRASES.get(attrs["chromatin_density"], f"{attrs['chromatin_density']} chromatin")
    cytoplasm_amount = _CYTOPLASM_AMOUNT_PHRASES.get(attrs["nuclear_cytoplasmic_ratio"], "cytoplasm")
    cell_size = _CELL_SIZE_PHRASES.get(attrs["cell_size"], attrs["cell_size"])
    cell_shape = _CELL_SHAPE_PHRASES.get(attrs["cell_shape"], attrs["cell_shape"])

    sentences = [
        f"This is a {cell_type} with {nucleus}, {chromatin}, and {cytoplasm_amount}.",
        f"The cell shows {cell_size}, {cell_shape}, and {attrs['cytoplasm_colour']} "
        f"cytoplasm with a {attrs['cytoplasm_texture']} texture"
        + (", containing vacuoles" if attrs["cytoplasm_vacuole"] == "yes" else "")
        + ".",
    ]

    if attrs["granularity"] == "yes" and attrs["granule_type"] not in ("nil", ""):
        sentences.append(
            f"Cytoplasmic granules are {attrs['granule_type']} and {attrs['granule_colour']}."
        )
    else:
        sentences.append("No prominent cytoplasmic granules are visible.")

    return " ".join(sentences)
