"""
CHULA-OCR Evaluation Script

Author: Teerapong Panboonyuen (Kao), Ph.D.
        C2F Postdoctoral Fellow, Chulalongkorn University

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/chula_ocr_best.pth \
        --test_dir data/test/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import CHULA_OCR
from src.dataset import ThaiLandTitleDeedDataset, ThaiOCRTokenizer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    import editdistance
    return editdistance.eval(a, b)


def compute_metrics(preds: list[str], targets: list[str]) -> dict:
    n = len(preds)
    exact_match = sum(p == t for p, t in zip(preds, targets)) / n
    ned = sum(
        edit_distance(p, t) / max(len(t), 1)
        for p, t in zip(preds, targets)
    ) / n
    cer = sum(edit_distance(p, t) for p, t in zip(preds, targets)) / sum(
        max(len(t), 1) for t in targets
    )

    # Precision / Recall (character level)
    total_tp, total_fp, total_fn = 0, 0, 0
    for p, t in zip(preds, targets):
        p_chars = list(p)
        t_chars = list(t)
        tp = sum((c in p_chars) for c in t_chars)
        fp = max(len(p_chars) - tp, 0)
        fn = max(len(t_chars) - tp, 0)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)

    return {
        "accuracy": exact_match,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cer": cer,
        "ned": ned,
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: CHULA_OCR,
    loader: DataLoader,
    tokenizer: ThaiOCRTokenizer,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    uncertainties = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        preds, unc = model.predict(images, return_uncertainty=True)
        pred_texts = [tokenizer.decode(p.tolist()) for p in preds]

        all_preds.extend(pred_texts)
        all_targets.extend(labels)
        if unc is not None:
            uncertainties.append(unc.mean().item())

    metrics = compute_metrics(all_preds, all_targets)
    if uncertainties:
        metrics["mean_uncertainty"] = sum(uncertainties) / len(uncertainties)

    return metrics, all_preds, all_targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate CHULA-OCR")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  CHULA-OCR Evaluation")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    print(f"{'='*55}\n")

    # Load model
    tokenizer = ThaiOCRTokenizer()
    model = CHULA_OCR.from_pretrained(args.checkpoint).to(device)

    # Dataset
    dataset = ThaiLandTitleDeedDataset(
        data_dir=args.test_dir,
        split="test",
        tokenizer=tokenizer,
        img_size=args.img_size,
        max_seq_len=args.max_seq_len,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=ThaiLandTitleDeedDataset.collate_fn,
    )

    t0 = time.time()
    metrics, preds, targets = evaluate(model, loader, tokenizer, device)
    elapsed = time.time() - t0

    print(f"\n{'─'*55}")
    print(f"  Results on {len(targets):,} test samples")
    print(f"{'─'*55}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"  {'elapsed':<25} {elapsed:.2f}s")
    print(f"{'─'*55}\n")

    # Save results
    output = {
        "metrics": metrics,
        "num_samples": len(targets),
        "checkpoint": args.checkpoint,
        "elapsed_seconds": elapsed,
        "predictions": [
            {"pred": p, "target": t, "correct": p == t}
            for p, t in zip(preds, targets)
        ]
    }
    out_path = Path(args.output_dir) / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
