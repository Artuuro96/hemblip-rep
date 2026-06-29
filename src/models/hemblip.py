"""
HemBLIP model factory.

Two variants (paper §2.2):
  1. HemBLIPFull  — full fine-tuning of BLIP (vision encoder + text decoder)
  2. HemBLIPLoRA  — LoRA on decoder attention layers; vision encoder frozen

Base model: Salesforce/blip-image-captioning-base
  • ViT-B/16 image encoder
  • BERT-like text decoder (cross-attention to image features)

LoRA is applied with the `peft` library.

Usage:
    model, processor = build_hemblip(lora=False)   # full
    model, processor = build_hemblip(lora=True)    # LoRA
    model, processor = build_hemblip(lora=True, checkpoint="outputs/hemblip_lora")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import BlipForConditionalGeneration, BlipProcessor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "Salesforce/blip-image-captioning-base"

# Module name suffixes for PEFT's endswith-matching.
# The vision encoder uses 'self_attn.qkv' (fused), so these suffixes
# exclusively match BERT-style text decoder attention layers.
BLIP_LORA_TARGET_MODULES = [
    "self.query",      # attention.self.query  &  crossattention.self.query
    "self.key",        # attention.self.key    &  crossattention.self.key
    "self.value",      # attention.self.value  &  crossattention.self.value
    "output.dense",    # attention.output.dense & crossattention.output.dense
]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_hemblip(
    base_model: str = DEFAULT_BASE_MODEL,
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    checkpoint: Optional[str] = None,
    freeze_vision: bool = False,
) -> Tuple[nn.Module, BlipProcessor]:
    """
    Build a HemBLIP model and its processor.

    Args:
        base_model:            HuggingFace model id or local path.
        lora:                  If True, apply LoRA to the text decoder.
        lora_r:                LoRA rank.
        lora_alpha:            LoRA scaling factor.
        lora_dropout:          Dropout in LoRA layers.
        lora_target_modules:   Which linear layers to adapt (default: attention).
        checkpoint:            Path to a saved PEFT / full checkpoint directory.
        freeze_vision:         Freeze the ViT vision encoder (always frozen in LoRA variant).

    Returns:
        (model, processor) ready for training or inference.
    """
    processor = BlipProcessor.from_pretrained(base_model)
    model = BlipForConditionalGeneration.from_pretrained(base_model)

    if lora:
        # Vision encoder is frozen in the LoRA variant (paper §2.2)
        for param in model.vision_model.parameters():
            param.requires_grad = False

        target_modules = lora_target_modules or BLIP_LORA_TARGET_MODULES
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            # task_type omitted: SEQ_2_SEQ_LM wraps the full forward and
            # conflicts with transformers 5.x BLIP encoder signature.
        )

        if checkpoint and Path(checkpoint).exists():
            model = PeftModel.from_pretrained(model, checkpoint)
            print(f"  Loaded LoRA checkpoint from {checkpoint}")
        else:
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

    else:
        if freeze_vision:
            for param in model.vision_model.parameters():
                param.requires_grad = False
            print("  Vision encoder frozen (full fine-tune of text decoder only)")

        if checkpoint and Path(checkpoint).exists():
            state = torch.load(Path(checkpoint) / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"  Loaded full checkpoint from {checkpoint}")

    _print_param_stats(model)
    return model, processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_param_stats(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total | {trainable:,} trainable ({100*trainable/total:.2f}%)")


def get_image_embeddings(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Extract frozen vision encoder embeddings for use in the classifier probe.

    Works for both plain BLIP and PEFT-wrapped BLIP.
    """
    base = model.base_model.model if hasattr(model, "base_model") else model
    with torch.no_grad():
        vision_out = base.vision_model(pixel_values=pixel_values, return_dict=True)
    # CLS token (index 0) from the last hidden state
    return vision_out.last_hidden_state[:, 0, :]


def save_model(model: nn.Module, output_dir: str, processor: BlipProcessor) -> None:
    """Save model weights and processor to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out))
    else:
        torch.save(model.state_dict(), str(out / "pytorch_model.bin"))
    processor.save_pretrained(str(out))
    print(f"  Model saved to {out}")


def generate_captions(
    model: nn.Module,
    processor: BlipProcessor,
    pixel_values: torch.Tensor,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    device: str = "cpu",
) -> List[str]:
    """Run beam-search decoding and decode to strings."""
    model.eval()
    with torch.no_grad():
        base = model.base_model.model if hasattr(model, "base_model") else model
        out_ids = base.generate(
            pixel_values=pixel_values.to(device),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    return processor.batch_decode(out_ids, skip_special_tokens=True)
