"""
Training loop para HemBLIP — optimizado para RTX 3070 (CUDA fp16).

Características:
  - AdamW + linear warm-up (paper §2.2)
  - Mixed-precision fp16 con torch.amp (Tensor Cores)
  - Early stopping en validation loss
  - Gradient accumulation
  - DataLoader con num_workers=4 y pin_memory para máximo throughput CUDA
  - cudnn.benchmark=True para convs más rápidas
  - Soporte opcional torch.compile() (PyTorch 2.x, ~10% extra velocidad)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.hemblip_dataset import HemBLIPCollator, HemBLIPDataset
from src.models.hemblip import save_model


def _linear_warmup_schedule(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - progress)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HemBLIPTrainer:
    """
    Trainer optimizado para RTX 3070 con CUDA + fp16.

    Args:
        model:          BLIP / PEFT model.
        processor:      BlipProcessor (para guardar checkpoints).
        train_ds:       HemBLIPDataset de entrenamiento.
        val_ds:         HemBLIPDataset de validación.
        output_dir:     Directorio para checkpoints y logs.
        num_epochs:     Máximo de epochs.
        batch_size:     Batch size por paso (recomendado 8 para LoRA, 4 para full).
        learning_rate:  LR pico (paper: 5e-5).
        weight_decay:   AdamW weight decay.
        warmup_steps:   Pasos de warm-up lineal.
        patience:       Early stopping (epochs sin mejora en val).
        fp16:           Mixed-precision con Tensor Cores (True en RTX 3070).
        gradient_accumulation_steps: Acumular gradientes (batch efectivo = batch_size × grad_accum).
        num_workers:    Workers para DataLoader (4 recomendado en CUDA).
        compile_model:  torch.compile() para ~10% extra velocidad (PyTorch 2.x).
        device:         'cuda' | 'mps' | 'cpu' (auto-detectado si None).
    """

    def __init__(
        self,
        model: nn.Module,
        processor,
        train_ds: HemBLIPDataset,
        val_ds: HemBLIPDataset,
        output_dir: str = "outputs/hemblip",
        num_epochs: int = 20,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        patience: int = 5,
        fp16: bool = True,
        gradient_accumulation_steps: int = 2,
        num_workers: int = 4,
        compile_model: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or _auto_device()
        self.fp16 = fp16 and self.device == "cuda"
        self.grad_accum = gradient_accumulation_steps

        # cudnn.benchmark acelera ops con input size fijo (imágenes 384×384)
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            print(f"  CUDA: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model = model.to(self.device)
        if compile_model and self.device == "cuda":
            print("  Compilando modelo con torch.compile() ...")
            self.model = torch.compile(self.model)

        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience

        # DataLoader con pin_memory + num_workers para máximo throughput GPU
        pin_memory = self.device == "cuda"
        _nw = num_workers if self.device == "cuda" else 0
        collator = HemBLIPCollator()
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collator, num_workers=_nw,
            pin_memory=pin_memory, persistent_workers=(_nw > 0),
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collator, num_workers=_nw,
            pin_memory=pin_memory, persistent_workers=(_nw > 0),
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

        total_steps = (
            math.ceil(len(train_ds) / batch_size) * num_epochs // gradient_accumulation_steps
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: _linear_warmup_schedule(s, warmup_steps, total_steps),
        )

        # GradScaler para fp16 — usa la API nueva torch.amp
        self.scaler = torch.amp.GradScaler("cuda") if self.fp16 else None
        self.history: Dict = {"train_loss": [], "val_loss": []}

        eff_batch = batch_size * gradient_accumulation_steps
        print(f"  Batch efectivo: {eff_batch}  (size={batch_size} × accum={gradient_accumulation_steps})")
        print(f"  fp16: {self.fp16} | workers: {_nw} | steps/epoch: {len(self.train_loader)}")

    # -----------------------------------}----
    # 
    # 

    def train(self) -> str:
        """Entrena hasta max epochs o early stopping. Devuelve ruta al mejor checkpoint."""
        best_val_loss = float("inf")
        epochs_no_improve = 0
        global_step = 0

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self._eval_epoch()
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            vram_str = ""
            if self.device == "cuda":
                used = torch.cuda.max_memory_allocated() / 1e9
                vram_str = f" | VRAM={used:.1f}GB"
                torch.cuda.reset_peak_memory_stats()

            print(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"lr={self.scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.0f}s{vram_str}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_dir = str(self.output_dir / "best")
                save_model(self.model, best_dir, self.processor)
                print(f"  [OK] Mejor val_loss={best_val_loss:.4f} -> {best_dir}")
            else:
                epochs_no_improve += 1
                print(f"  Sin mejora ({epochs_no_improve}/{self.patience})")
                if epochs_no_improve >= self.patience:
                    print("  Early stopping.")
                    break

            global_step += math.ceil(len(self.train_loader) / self.grad_accum)

        self._save_history()
        return str(self.output_dir / "best")

    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for step, batch in enumerate(pbar):
            # non_blocking=True solapa la transferencia CPU→GPU con el cómputo
            pv = batch["pixel_values"].to(self.device, non_blocking=True)
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    out = self.model(
                        pixel_values=pv, input_ids=ids,
                        attention_mask=mask, labels=labels,
                    )
                    loss = out.loss / self.grad_accum
                self.scaler.scale(loss).backward()
            else:
                out = self.model(
                    pixel_values=pv, input_ids=ids,
                    attention_mask=mask, labels=labels,
                )
                loss = out.loss / self.grad_accum
                loss.backward()

            total_loss += loss.item() * self.grad_accum

            if (step + 1) % self.grad_accum == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------

    def _eval_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        ctx = torch.amp.autocast("cuda") if self.fp16 else torch.no_grad()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val", leave=False):
                pv = batch["pixel_values"].to(self.device, non_blocking=True)
                ids = batch["input_ids"].to(self.device, non_blocking=True)
                mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                with ctx:
                    out = self.model(
                        pixel_values=pv, input_ids=ids,
                        attention_mask=mask, labels=labels,
                    )
                total_loss += out.loss.item()
        return total_loss / max(1, len(self.val_loader))

    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
