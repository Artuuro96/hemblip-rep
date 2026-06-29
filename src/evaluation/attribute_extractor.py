"""
Regex-based morphological attribute extractor (paper §2.3).

For each generated caption, we perform controlled string matching to detect
mentions of predefined cytological attributes and compare against ground-truth
attribute labels extracted from the reference caption / annotation.

Attributes tracked:
    cell_type, nuclear_chromatin_texture, cytoplasm_amount,
    diagnosis, nuclear_shape, overall_shape, cell_size
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Attribute patterns
# ---------------------------------------------------------------------------

_CELL_TYPE_PATTERNS = {
    "neutrophil":  r"\bneutrophil",
    "eosinophil":  r"\beosinophil",
    "basophil":    r"\bbasophil(?!ic)",
    "lymphocyte":  r"\blymphocyte",
    "monocyte":    r"\bmonocyte",
}

_DIAGNOSIS_PATTERNS = {
    "ALL":    r"\b(?:ALL|acute\s+lympho(?:blastic|cytic))",
    "AML":    r"\b(?:AML|acute\s+myelogen(?:ous|ic))",
    "APML":   r"\b(?:APM?L|promyelocytic|AML[- ]?M3)",
    "CLL":    r"\b(?:CLL|chronic\s+lympho(?:cytic|blastic))",
    "CML":    r"\b(?:CML|chronic\s+myelogen(?:ous|ic))",
    "healthy": r"\b(?:healthy|normal)\s+(?:cell|white)",
}

_CHROMATIN_PATTERNS = {
    "coarse":   r"\bcoarse\s+chromatin",
    "fine":     r"\bfine\s+chromatin",
    "open":     r"\bopen\s+chromatin",
    "dense":    r"\bdens(?:e|ely)\s+(?:packed\s+)?chromatin",
    "clumped":  r"\bclumped\s+chromatin",
    "stippled": r"\bstippled\s+chromatin",
}

_CYTOPLASM_AMOUNT_PATTERNS = {
    "abundant": r"\babundant\s+cytoplasm",
    "moderate": r"\bmoderate\s+cytoplasm",
    "scant":    r"\bscant(?:y)?\s+cytoplasm",
    "absent":   r"\b(?:no|absent)\s+cytoplasm",
}

_NUCLEAR_SHAPE_PATTERNS = {
    "round":          r"\b(?:round|circular)\s+nucl(?:eus|ear|i)",
    "oval":           r"\boval\s+nucl(?:eus|ear|i)",
    "kidney":         r"\bkidney(?:-shaped)?\s+nucl(?:eus|ear|i)",
    "horseshoe":      r"\bhorseshoe(?:-shaped)?\s+nucl(?:eus|ear|i)",
    "irregular":      r"\birregular\s+nucl(?:eus|ear|i)",
    "bilobed":        r"\bbilobed\s+nucl(?:eus|ear|i)",
    "multilobulated": r"\bmulti(?:-lobulated|lobulated|lobed)\s+nucl(?:eus|ear|i)",
    "indented":       r"\bindented\s+nucl(?:eus|ear|i)",
}

_OVERALL_SHAPE_PATTERNS = {
    "round":     r"\b(?:round|circular)\s+(?:overall\s+)?(?:cell|shape)",
    "oval":      r"\boval\s+(?:overall\s+)?(?:cell|shape)",
    "irregular": r"\birregular\s+(?:overall\s+)?(?:cell|shape)",
}

_CELL_SIZE_PATTERNS = {
    "small":  r"\bsmall\s+(?:cell|size)",
    "medium": r"\bmedium\s+(?:cell|size)",
    "large":  r"\blarge\s+(?:cell|size)",
}

# Map attribute category → pattern dict
ATTRIBUTE_GROUPS: Dict[str, Dict[str, str]] = {
    "cell_type":              _CELL_TYPE_PATTERNS,
    "nuclear_chromatin_texture": _CHROMATIN_PATTERNS,
    "cytoplasm_amount":       _CYTOPLASM_AMOUNT_PATTERNS,
    "diagnosis":              _DIAGNOSIS_PATTERNS,
    "nuclear_shape":          _NUCLEAR_SHAPE_PATTERNS,
    "overall_shape":          _OVERALL_SHAPE_PATTERNS,
    "cell_size":              _CELL_SIZE_PATTERNS,
}


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_attributes(text: str) -> Dict[str, Optional[str]]:
    """
    Search a caption for morphological attribute mentions.

    Returns a dict mapping each attribute group to the *first matched value*
    (or None if no pattern fires).

    Example::

        >>> extract_attributes("This is a neutrophil with coarse chromatin.")
        {'cell_type': 'neutrophil', 'nuclear_chromatin_texture': 'coarse', ...}
    """
    text = text.lower()
    result: Dict[str, Optional[str]] = {}
    for group, patterns in ATTRIBUTE_GROUPS.items():
        found = None
        for value, pat in patterns.items():
            if re.search(pat, text, re.IGNORECASE):
                found = value
                break
        result[group] = found
    return result


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def compute_attribute_accuracy(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute per-attribute accuracy between generated and reference captions.

    Args:
        predictions: List of model-generated caption strings.
        references:  Corresponding ground-truth caption strings.

    Returns:
        Dict mapping attribute group name to accuracy in [0, 1].
        Only samples where the reference mentions the attribute are counted.
    """
    assert len(predictions) == len(references), "Length mismatch"

    correct: Dict[str, int] = {g: 0 for g in ATTRIBUTE_GROUPS}
    total: Dict[str, int] = {g: 0 for g in ATTRIBUTE_GROUPS}

    for pred, ref in zip(predictions, references):
        pred_attrs = extract_attributes(pred)
        ref_attrs = extract_attributes(ref)

        for group in ATTRIBUTE_GROUPS:
            ref_val = ref_attrs[group]
            if ref_val is None:
                continue  # attribute not mentioned in reference → skip
            total[group] += 1
            if pred_attrs[group] == ref_val:
                correct[group] += 1

    accuracy: Dict[str, float] = {}
    for group in ATTRIBUTE_GROUPS:
        if total[group] > 0:
            accuracy[group] = correct[group] / total[group]
        else:
            accuracy[group] = float("nan")

    return accuracy


def compute_attribute_accuracy_from_annotations(
    predictions: List[str],
    annotations: List[Dict],
) -> Dict[str, float]:
    """
    Variant that compares against raw attribute dicts (ground-truth labels)
    rather than reference captions.  The attribute labels are converted to
    their canonical form and matched against pattern-extracted values.
    """
    correct: Dict[str, int] = {g: 0 for g in ATTRIBUTE_GROUPS}
    total: Dict[str, int] = {g: 0 for g in ATTRIBUTE_GROUPS}

    for pred, ann in zip(predictions, annotations):
        pred_attrs = extract_attributes(pred)
        gt_attrs = _annotation_to_canonical(ann)

        for group in ATTRIBUTE_GROUPS:
            gt_val = gt_attrs.get(group)
            if gt_val is None:
                continue
            total[group] += 1
            if pred_attrs[group] == gt_val:
                correct[group] += 1

    return {
        g: (correct[g] / total[g] if total[g] > 0 else float("nan"))
        for g in ATTRIBUTE_GROUPS
    }


def _annotation_to_canonical(ann: Dict) -> Dict[str, Optional[str]]:
    """Map raw annotation dict keys to ATTRIBUTE_GROUPS keys."""
    mapping: Dict[str, Optional[str]] = {}

    # cell_type
    ct = ann.get("cell_type", "")
    if ct:
        mapping["cell_type"] = ct.lower()

    # diagnosis
    diag = ann.get("diagnosis", "")
    if diag:
        mapping["diagnosis"] = diag.upper() if diag.upper() in _DIAGNOSIS_PATTERNS else diag.lower()

    # chromatin
    chrom = ann.get("chromatin_texture") or ann.get("chromatin") or ""
    if chrom:
        mapping["nuclear_chromatin_texture"] = chrom.lower()

    # cytoplasm amount
    ca = ann.get("cytoplasm_amount", "")
    if ca:
        mapping["cytoplasm_amount"] = ca.lower()

    # nuclear shape
    ns = ann.get("nuclear_shape", "")
    if ns:
        mapping["nuclear_shape"] = ns.lower().replace(" ", "_").replace("-", "_")

    # cell size
    cs = ann.get("cell_size", "")
    if cs:
        mapping["cell_size"] = cs.lower()

    return mapping


# ---------------------------------------------------------------------------
# Confusion-matrix helper
# ---------------------------------------------------------------------------

def attribute_confusion(
    predictions: List[str],
    references: List[str],
    group: str,
) -> Dict[Tuple[str, str], int]:
    """
    Return a confusion matrix for a single attribute group as a dict
    mapping (true_label, pred_label) → count.
    """
    matrix: Dict[Tuple[str, str], int] = {}
    values = list(ATTRIBUTE_GROUPS[group].keys())

    for pred, ref in zip(predictions, references):
        ref_val = extract_attributes(ref).get(group)
        pred_val = extract_attributes(pred).get(group)
        if ref_val is None:
            continue
        key = (ref_val, pred_val or "none")
        matrix[key] = matrix.get(key, 0) + 1

    return matrix


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    captions = [
        "This is a neutrophil with coarse chromatin, multi-lobulated nucleus, and moderate cytoplasm.",
        "This image shows a CML case with medium cell size, round nuclear shape, and abundant cytoplasm.",
    ]
    for cap in captions:
        print(cap)
        print(extract_attributes(cap))
        print()
