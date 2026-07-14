#!/usr/bin/env python3
"""
Mini desktop GUI (Tkinter) to try a trained HemBLIP checkpoint: pick an
image, optionally type a prompt to condition the caption on, and see what
the model generates.

Usage:
    python scripts/gui_predict.py --config configs/hemblip_lora.yaml \
        --checkpoint outputs/hemblip_lora/best --device cuda
"""

import argparse
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

import torch
import yaml
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hemblip import build_hemblip
from src.training.trainer import _auto_device

PREVIEW_SIZE = 320


def parse_args():
    p = argparse.ArgumentParser(description="HemBLIP GUI")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_beams", type=int, default=4)
    return p.parse_args()


class HemBLIPApp:
    def __init__(self, root, args):
        self.args = args
        self.device = args.device or _auto_device()
        self.model = None
        self.processor = None
        self.base = None
        self.image_path = None
        self.pil_image = None

        root.title("HemBLIP — probar modelo")
        root.geometry("560x680")
        root.resizable(False, False)
        self.root = root

        self.status_var = tk.StringVar(value=f"Cargando checkpoint en {self.device} ...")
        tk.Label(root, textvariable=self.status_var, fg="gray20").pack(pady=(10, 4))

        self.canvas = tk.Canvas(root, width=PREVIEW_SIZE, height=PREVIEW_SIZE,
                                 bg="#eee", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(pady=6)
        self.canvas_text = self.canvas.create_text(
            PREVIEW_SIZE // 2, PREVIEW_SIZE // 2, text="Sin imagen", fill="#888"
        )

        tk.Button(root, text="Elegir imagen...", command=self.choose_image).pack(pady=4)

        tk.Label(root, text="Prompt (opcional):").pack(pady=(12, 2))
        self.prompt_entry = tk.Entry(root, width=60)
        self.prompt_entry.pack(pady=2)

        self.generate_btn = tk.Button(
            root, text="Generar caption", command=self.on_generate,
            state="disabled", bg="#2d6cdf", fg="white",
        )
        self.generate_btn.pack(pady=12)

        tk.Label(root, text="Resultado:").pack(anchor="w", padx=20)
        self.output = scrolledtext.ScrolledText(root, width=64, height=8, wrap="word")
        self.output.pack(padx=20, pady=6)
        self.output.configure(state="disabled")

        threading.Thread(target=self.load_model, daemon=True).start()

    # ------------------------------------------------------------------

    def load_model(self):
        try:
            with open(self.args.config) as f:
                cfg = yaml.safe_load(f)
            model, processor = build_hemblip(
                base_model=cfg["model"]["base_model"],
                lora=cfg["model"].get("lora", False),
                lora_r=cfg["model"].get("lora_r", 16),
                lora_alpha=cfg["model"].get("lora_alpha", 32),
                lora_dropout=cfg["model"].get("lora_dropout", 0.05),
                lora_target_modules=cfg["model"].get("lora_target_modules"),
                checkpoint=self.args.checkpoint,
            )
            model.to(self.device).eval()
            self.model = model
            self.processor = processor
            self.base = model.base_model.model if hasattr(model, "base_model") else model
            self.root.after(0, self.on_model_ready)
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error cargando modelo: {e}"))

    def on_model_ready(self):
        self.status_var.set(f"Listo — checkpoint cargado en {self.device}")
        if self.image_path:
            self.generate_btn.configure(state="normal")

    # ------------------------------------------------------------------

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Elegir imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return

        self.image_path = path
        self.pil_image = image
        self.show_preview(image)
        if self.model is not None:
            self.generate_btn.configure(state="normal")

    def show_preview(self, image: Image.Image):
        preview = image.copy()
        preview.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        self.tk_image = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(PREVIEW_SIZE // 2, PREVIEW_SIZE // 2, image=self.tk_image)

    # ------------------------------------------------------------------

    def on_generate(self):
        if self.pil_image is None or self.model is None:
            return
        self.generate_btn.configure(state="disabled", text="Generando...")
        self.set_output("")
        prompt = self.prompt_entry.get().strip()
        threading.Thread(target=self.run_generate, args=(prompt,), daemon=True).start()

    def run_generate(self, prompt: str):
        try:
            inputs = self.processor(
                images=self.pil_image, text=prompt or None, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out_ids = self.base.generate(
                    **inputs,
                    max_new_tokens=self.args.max_new_tokens,
                    num_beams=self.args.num_beams,
                )
            caption = self.processor.decode(out_ids[0], skip_special_tokens=True)
            self.root.after(0, lambda: self.set_output(caption))
        except Exception as e:
            self.root.after(0, lambda: self.set_output(f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.generate_btn.configure(
                state="normal", text="Generar caption"
            ))

    def set_output(self, text: str):
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.configure(state="disabled")


def main():
    args = parse_args()
    root = tk.Tk()
    HemBLIPApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
