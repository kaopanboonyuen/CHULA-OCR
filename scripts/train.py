"""
CHULA-OCR Training Script

Author: Teerapong Panboonyuen (Kao), Ph.D.
        C2F Postdoctoral Fellow, Chulalongkorn University
        
Usage:
    python scripts/train.py --config configs/chula_ocr_base.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import CHULA_OCR
from src.dataset import ThaiLandTitleDeedDataset, ThaiOCRTokenizer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_cer(preds: list[str], targets: list[str]) -> float:
    """Character Error Rate."""
    import editdistance
    total_dist, total_len = 0, 0
    for p, t in zip(preds, targets):
        total_dist += editdistance.eval(p, t)
        total_len += len(t)
    return total_dist / max(total_len, 1)


def compute_accuracy(preds: list[str], targets: list[str]) -> float:
    """Character-level accuracy."""
    correct = sum(p == t for p, t in zip(preds, targets))
    return correct / max(len(targets), 1)


# ---------------------------------------------------------------------------
# Cosine LR Scheduler with Warmup
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 eta_min: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.current_step = 0
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        t = self.current_step
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if t <= self.warmup_steps:
                lr = base_lr * t / self.warmup_steps
            else:
                progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CHULAOCRTrainer:
    """
    Full training pipeline for CHULA-OCR.

    Hardware target: 8× NVIDIA A100 (640 GB total)
    Training: 50 epochs, AdamW, cosine LR with warmup, FP16
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = Path(config.get("output_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(config.get("seed", 42))
        self._build_components()

    def _build_components(self):
        cfg = self.config

        # Tokenizer
        self.tokenizer = ThaiOCRTokenizer()

        # Model
        self.model = CHULA_OCR(
            vocab_size=len(self.tokenizer),
            img_size=cfg.get("img_size", 224),
            patch_size=cfg.get("patch_size", 16),
            embed_dim=cfg.get("embed_dim", 768),
            num_decoder_layers=cfg.get("num_decoder_layers", 6),
            num_heads=cfg.get("num_heads", 8),
            ff_dim=cfg.get("ff_dim", 2048),
            max_seq_len=cfg.get("max_seq_len", 256),
            lambda_u=cfg.get("lambda_u", 0.5),
            lambda_h=cfg.get("lambda_h", 0.3),
            alpha_entropy=cfg.get("alpha_entropy", 2.0),
        ).to(self.device)

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"[Trainer] Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )

        # Mixed precision
        self.scaler = GradScaler()

        # Datasets
        data_dir = cfg.get("data_dir", "data/processed")
        self.train_loader = self._make_loader(data_dir, "train", augment=True)
        self.val_loader = self._make_loader(data_dir, "val", augment=False)

        # LR schedule
        total_steps = len(self.train_loader) * cfg.get("epochs", 50)
        warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps, total_steps
        )

        self.epochs = cfg.get("epochs", 50)
        self.best_val_acc = 0.0
        self.patience = cfg.get("patience", 10)
        self.patience_counter = 0

    def _make_loader(self, data_dir: str, split: str, augment: bool) -> DataLoader:
        dataset = ThaiLandTitleDeedDataset(
            data_dir=data_dir,
            split=split,
            tokenizer=self.tokenizer,
            img_size=self.config.get("img_size", 224),
            max_seq_len=self.config.get("max_seq_len", 256),
            augment=augment,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 64),
            shuffle=(split == "train"),
            num_workers=self.config.get("num_workers", 8),
            pin_memory=True,
            collate_fn=ThaiLandTitleDeedDataset.collate_fn,
            drop_last=(split == "train"),
        )

    # ── Training step ────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        metrics = {k: 0.0 for k in ["loss", "loss_ce", "loss_uncertainty", "loss_heuristic"]}
        n_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            tokens = batch["tokens"].to(self.device, non_blocking=True)
            masks = batch["padding_mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                model = self.model.module if hasattr(self.model, "module") else self.model
                out = model(images, tokens, masks)

            self.scaler.scale(out["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            for k in metrics:
                metrics[k] += out[k].item()
            n_batches += 1

        return {k: v / n_batches for k, v in metrics.items()}

    # ── Validation step ──────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.eval()

        all_preds, all_targets = [], []

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"]

            preds, _ = model.predict(images, bos_token=1, eos_token=2)
            pred_texts = [self.tokenizer.decode(p.tolist()) for p in preds]

            all_preds.extend(pred_texts)
            all_targets.extend(labels)

        cer = compute_cer(all_preds, all_targets)
        acc = compute_accuracy(all_preds, all_targets)
        return {"val_cer": cer, "val_acc": acc}

    # ── Main train loop ──────────────────────────────────────────────────

    def train(self):
        print("\n" + "=" * 65)
        print("  CHULA-OCR Training")
        print(f"  Author : Teerapong Panboonyuen (Kao), Ph.D.")
        print(f"  Device : {self.device}")
        print(f"  Epochs : {self.epochs}")
        print("=" * 65 + "\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch()

            elapsed = time.time() - t0
            lr = self.scheduler.get_lr()

            print(
                f"Epoch [{epoch:3d}/{self.epochs}] "
                f"Loss: {train_metrics['loss']:.4f} "
                f"(CE {train_metrics['loss_ce']:.3f} "
                f"U {train_metrics['loss_uncertainty']:.3f} "
                f"H {train_metrics['loss_heuristic']:.3f}) "
                f"| Val Acc: {val_metrics['val_acc']:.4f} "
                f"CER: {val_metrics['val_cer']:.4f} "
                f"| LR: {lr:.2e} "
                f"| {elapsed:.1f}s"
            )

            # Checkpoint
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.patience_counter = 0
                model = self.model.module if hasattr(self.model, "module") else self.model
                model.save_checkpoint(
                    str(self.output_dir / "chula_ocr_best.pth"),
                    config=self.config,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                )
                print(f"  ✅ New best checkpoint saved (val_acc={self.best_val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n⏹ Early stopping at epoch {epoch}.")
                    break

        print(f"\n✅ Training complete. Best val accuracy: {self.best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train CHULA-OCR")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--lambda_u", type=float, default=0.5)
    parser.add_argument("--lambda_h", type=float, default=0.3)
    parser.add_argument("--alpha_entropy", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)
    trainer = CHULAOCRTrainer(config)
    trainer.train()
