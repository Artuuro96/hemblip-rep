"""
Métricas de captioning: BLEU-4, ROUGE-L, BERTScore F1.
Optimizado para CUDA (RTX 3070) — BERTScore corre en GPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import evaluate
from bert_score import score as bert_score_fn
from tqdm import tqdm


_bleu_metric = None
_rouge_metric = None


def _get_bleu():
    global _bleu_metric
    if _bleu_metric is None:
        _bleu_metric = evaluate.load("bleu")
    return _bleu_metric


def _get_rouge():
    global _rouge_metric
    if _rouge_metric is None:
        _rouge_metric = evaluate.load("rouge")
    return _rouge_metric


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------

def compute_bleu(predictions: List[str], references: List[str]) -> float:
    refs_wrapped = [[r] for r in references]
    try:
        result = _get_bleu().compute(predictions=predictions, references=refs_wrapped)
        return float(result.get("bleu", 0.0))
    except Exception:
        return 0.0


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    try:
        result = _get_rouge().compute(
            predictions=predictions, references=references, rouge_types=["rougeL"]
        )
        return float(result.get("rougeL", 0.0))
    except Exception:
        return 0.0


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 128,
    device: Optional[str] = None,
) -> float:
    """
    BERTScore F1 en GPU.
    batch_size=128 es seguro en RTX 3070 con distilbert.
    Usa model_type='roberta-large' para números exactos del paper (más lento).
    """
    device = device or _default_device()
    try:
        _, _, F1 = bert_score_fn(
            predictions, references,
            lang="en",
            model_type=model_type,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )
        return float(F1.mean().item())
    except Exception as e:
        print(f"  BERTScore falló: {e}")
        return 0.0


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    bertscore_model: str = "distilbert-base-uncased",
    device: Optional[str] = None,
) -> Dict[str, float]:
    device = device or _default_device()
    assert predictions and len(predictions) == len(references)

    print("  BLEU ...")
    bleu = compute_bleu(predictions, references)
    print("  ROUGE-L ...")
    rouge_l = compute_rouge_l(predictions, references)
    print("  BERTScore ...")
    bscore = compute_bertscore(predictions, references, bertscore_model, device=device)

    return {"bleu": bleu, "rouge_l": rouge_l, "bertscore_f1": bscore}


# ---------------------------------------------------------------------------

def generate_predictions(
    model,
    processor,
    dataloader,
    device: Optional[str] = None,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    repetition_penalty: float = 1.3,
) -> tuple[List[str], List[str]]:
    """
    Genera captions con beam search en GPU.
    Usa autocast fp16 si hay CUDA disponible.
    """
    device = device or _default_device()
    use_fp16 = device == "cuda"

    model.eval()
    predictions: List[str] = []
    references: List[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating captions", unit="batch"):
            pv = batch["pixel_values"].to(device, non_blocking=True)

            base = model.base_model.model if hasattr(model, "base_model") else model
            ctx = torch.amp.autocast("cuda") if use_fp16 else torch.amp.autocast("cpu", enabled=False)
            with ctx:
                out_ids = base.generate(
                    pixel_values=pv,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                )

            preds = processor.batch_decode(out_ids, skip_special_tokens=True)
            predictions.extend(preds)

            ref_ids = batch["input_ids"].clone()
            ref_ids[ref_ids == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(ref_ids, skip_special_tokens=True)
            references.extend(refs)

    return predictions, references
