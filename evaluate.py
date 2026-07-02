#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

from src.data import RaceDataset, build_dataloader
from src.models.race_model import RACEModel, AblationConfig
from train import load_config, build_ablation, build_logger

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--ckpt",    default=None, help="指定单个 checkpoint")
    p.add_argument("--ablation", default=None,
                   choices=["wo_region","wo_ocr","wo_relevance",
                            "wo_conflict","wo_gate","wo_align"])
    p.add_argument("--all_ablations", action="store_true",
                   help="遍历全部消融组，输出对比表")
    p.add_argument("--split", default="test", choices=["val","test"])
    p.add_argument("--out_dir", default=None, help="结果 CSV/MD 保存目录")
    return p.parse_args()

@torch.no_grad()
def run_eval(
    model: RACEModel,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch)
        all_logits.append(out["logit"].cpu())
        all_labels.append(batch["label"].cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (logits > 0).astype(int)

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    f1_r = f1_score(labels, preds, pos_label=1, zero_division=0)
    f1_n = f1_score(labels, preds, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    return {
        "Acc":       round(float(acc),  4),
        "F1":        round(float(f1),   4),
        "F1_rumor":  round(float(f1_r), 4),
        "F1_real":   round(float(f1_n), 4),
        "AUC":       round(float(auc),  4),
        "_logits":   logits,
        "_labels":   labels,
        "_preds":    preds,
    }

def print_report(metrics: Dict, tag: str, logger):
    labels  = metrics["_labels"]
    preds   = metrics["_preds"]
    logits  = metrics["_logits"]

    logger.info(f"\n{'='*55}")
    logger.info(f"  [{tag}]  Acc={metrics['Acc']:.4f}  F1={metrics['F1']:.4f}  "
                f"F1_rumor={metrics['F1_rumor']:.4f}  "
                f"F1_real={metrics['F1_real']:.4f}  AUC={metrics['AUC']:.4f}")
    logger.info(f"{'='*55}")

    report = classification_report(
        labels, preds,
        target_names=["real(0)", "rumor(1)"],
        digits=4, zero_division=0,
    )
    logger.info("\n" + report)

    cm = confusion_matrix(labels, preds)
    logger.info(f"Confusion Matrix:\n{cm}")

_ABLATION_TAGS = [
    ("full",          {}),
    ("wo_region",     {"use_region":    False}),
    ("wo_ocr",        {"use_ocr":       False}),
    ("wo_relevance",  {"use_relevance": False}),
    ("wo_conflict",   {"use_conflict":  False}),
    ("wo_gate",       {"use_gate":      False}),
    ("wo_align",      {"use_align":     False}),
]

METRICS_COLS = ["Acc", "F1", "F1_rumor", "F1_real", "AUC"]

def save_comparison(rows: List[Dict], out_dir: Path, dataset: str, split: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{split}_ablation"

    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag"] + METRICS_COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in ["tag"] + METRICS_COLS})

    md_path = out_dir / f"{stem}.md"
    header = "| " + " | ".join(["Model"] + METRICS_COLS) + " |"
    sep    = "| " + " | ".join(["---"] * (len(METRICS_COLS) + 1)) + " |"
    lines  = [header, sep]
    for r in rows:
        cells = [r["tag"]] + [f"{r[c]:.4f}" for c in METRICS_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return csv_path, md_path

def build_model(cfg, ablation: AblationConfig, ckpt_path: Path,
                device: torch.device) -> RACEModel:
    model = RACEModel(
        d_c=cfg.dims.d_c, d_l=cfg.dims.d_l,
        d_g=cfg.dims.d_g, d_f=cfg.dims.d_f,
        d_r=cfg.dims.d_r, gamma=cfg.hparams.gamma,
        delta=cfg.hparams.delta, dropout=0.0,
        ablation=ablation,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    return model

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    out_dir = Path(args.out_dir) if args.out_dir else \
              Path(cfg.paths.log_dir) / cfg.data.dataset

    if not args.all_ablations:
        tag = args.ablation or "full"
        log_path = out_dir / f"eval_{args.split}_{tag}.log"
    else:
        log_path = out_dir / f"eval_{args.split}.log"
    logger   = build_logger(log_path)

    cache_dir = Path(cfg.paths.cache_dir) / cfg.data.dataset
    dataset   = RaceDataset(cache_dir, args.split, d_c=cfg.dims.d_c, logger=logger)
    loader    = build_dataloader(
        dataset, cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers,pin_memory=(device.type == "cuda"),
    )

    ckpt_root = Path(cfg.paths.checkpoint_dir) / cfg.data.dataset

    if not args.all_ablations:
        tag      = args.ablation or "full"
        ablation = build_ablation(cfg, args.ablation)
        ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_root / tag / "best.pt"
        model    = build_model(cfg, ablation, ckpt_path, device)
        metrics  = run_eval(model, loader, device)
        print_report(metrics, tag, logger)
        return

    rows = []
    for tag, ab_kwargs in _ABLATION_TAGS:
        ckpt_path = ckpt_root / tag / "best.pt"
        if not ckpt_path.exists():
            logger.warning(f"checkpoint not found: {ckpt_path}，跳过 {tag}")
            continue
        ablation = AblationConfig(**ab_kwargs)
        model    = build_model(cfg, ablation, ckpt_path, device)
        metrics  = run_eval(model, loader, device)
        print_report(metrics, tag, logger)
        rows.append({"tag": tag, **{k: metrics[k] for k in METRICS_COLS}})

    if rows:
        csv_p, md_p = save_comparison(rows, out_dir, cfg.data.dataset, args.split)
        logger.info(f"\n消融对比表已保存:\n  CSV -> {csv_p}\n  MD  -> {md_p}")
        md_text = md_p.read_text(encoding="utf-8")
        logger.info("\n" + md_text)

if __name__ == "__main__":
    main()