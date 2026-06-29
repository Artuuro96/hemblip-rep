"""
Merge LoRA adapter weights into the base BLIP model and save a self-contained
checkpoint ready for inference on any device (Mac M1/MPS, CPU, CUDA).

Usage:
    py scripts/export_model.py \
        --checkpoint outputs/hemblip_lora/best_model \
        --output     outputs/hemblip_lora/merged_model

The output directory contains a standard HuggingFace model that can be loaded
with BlipForConditionalGeneration.from_pretrained() without peft installed.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA weights into base BLIP model")
    p.add_argument("--checkpoint", required=True,
                   help="Path to best_model directory (contains adapter_config.json)")
    p.add_argument("--output", required=True,
                   help="Where to save the merged model")
    p.add_argument("--base-model", default=None,
                   help="Base model ID or path (auto-detected from adapter_config.json if omitted)")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    out  = Path(args.output)

    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}")

    import json
    from transformers import BlipForConditionalGeneration, BlipProcessor
    from peft import PeftModel

    # Auto-detect base model from adapter config
    adapter_cfg_path = ckpt / "adapter_config.json"
    if args.base_model:
        base_model_id = args.base_model
    elif adapter_cfg_path.exists():
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model_id = adapter_cfg.get("base_model_name_or_path", "Salesforce/blip-image-captioning-base")
    else:
        base_model_id = "Salesforce/blip-image-captioning-base"

    print(f"Base model : {base_model_id}")
    print(f"Checkpoint : {ckpt}")
    print(f"Output     : {out}")
    print()

    print("Loading base model...")
    base = BlipForConditionalGeneration.from_pretrained(base_model_id)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(ckpt))

    print("Merging weights...")
    model = model.merge_and_unload()
    model.eval()

    print("Saving merged model...")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))

    print("Saving processor...")
    processor = BlipProcessor.from_pretrained(base_model_id)
    processor.save_pretrained(str(out))

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1_000_000
    print(f"\nDone — saved to {out}  ({size_mb:.0f} MB)")
    print()
    print("To run inference on Mac M1:")
    print("  from transformers import BlipForConditionalGeneration, BlipProcessor")
    print("  from PIL import Image")
    print("  import torch")
    print(f'  model = BlipForConditionalGeneration.from_pretrained("{out.name}").to("mps")')
    print(f'  processor = BlipProcessor.from_pretrained("{out.name}")')
    print('  img = Image.open("image.jpg").convert("RGB")')
    print('  inputs = processor(images=img, return_tensors="pt").to("mps")')
    print('  out = model.generate(**inputs, max_new_tokens=128)')
    print('  print(processor.decode(out[0], skip_special_tokens=True))')


if __name__ == "__main__":
    main()
