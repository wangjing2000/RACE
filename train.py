#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from src.data import RaceDataset, build_dataloader
from src.data.kfold_dataset import KFoldRaceDataset, get_current_fold
from src.models.race_model import RACEModel, AblationConfig
from src.models.losses import RACELoss
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.meters import MetricBag
from src.utils.seed import set_seed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="checkpoint .pt 路径")
    p.add_argument("--ablation", default=None,
                   choices=["wo_region","wo_ocr","wo_relevance",
                            "wo_conflict","wo_gate","wo_align"],
                   help="单项消融（覆盖 config 中 ablation 设置）")
    return p.parse_args()

def load_config(path: str):
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)

    class Cfg:
        pass

    def _build(d):
        if not isinstance(d, dict):
            return d
        obj = Cfg()
        for k, v in d.items():
            setattr(obj, k, _build(v))
        return obj

    return _build(raw)

def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("race")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def build_ablation(cfg, override: Optional[str]) -> AblationConfig:
    ab_cfg = getattr(cfg, "ablation", None)
    kwargs = {}
    if ab_cfg is not None:
        for k in ("use_region","use_ocr","use_relevance",
                  "use_conflict","use_gate","use_align"):
            if hasattr(ab_cfg, k):
                kwargs[k] = getattr(ab_cfg, k)
    if override is not None:
        flag = override[3:]
        kwargs[f"use_{flag}"] = False
    return AblationConfig(**kwargs)

def train_one_epoch(model, loader, optimizer, loss_fn, device, ablation,
                    epoch, total_epochs, logger, use_tqdm):
    model.train()
    bag = MetricBag()
    all_preds, all_labels = [], []

    desc = f"Epoch {epoch}/{total_epochs} [train]"
    iterator = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True) \
        if (use_tqdm and tqdm) else loader

    for batch in iterator:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        out = model(batch)
        result = loss_fn(
            logit=out["logit"],
            label=batch["label"],
            c_ij=out["c_ij"],
            pair_mask=out["pair_mask"],
            s_c=out["s_c"],
            has_local=out["has_local"],use_align=ablation.use_align,
        )

        optimizer.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        B = batch["label"].shape[0]
        bag.update({k: v.item() for k, v in result.items()}, n=B)

        preds = (out["logit"].detach() > 0).long().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().tolist())

        if use_tqdm and tqdm and hasattr(iterator, "set_postfix"):
            avgs = bag.avg()
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            iterator.set_postfix(
                loss=f"{avgs['loss']:.4f}",
                F1=f"{f1:.4f}",
                L_cls=f"{avgs['L_cls']:.4f}",
            )

    avgs = bag.avg()
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    logger.info(
        f"[train] epoch={epoch}  loss={avgs['loss']:.4f}  "
        f"F1={f1:.4f}  L_cls={avgs['L_cls']:.4f}  "
        f"L_real={avgs['L_real']:.4f}  L_align={avgs['L_align']:.4f}"
    )
    return avgs, f1

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, ablation, epoch, logger):
    model.eval()
    bag = MetricBag()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch)
        result = loss_fn(
            logit=out["logit"],
            label=batch["label"],
            c_ij=out["c_ij"],
            pair_mask=out["pair_mask"],
            s_c=out["s_c"],
            has_local=out["has_local"],
            use_align=ablation.use_align,
        )
        B = batch["label"].shape[0]
        bag.update({k: v.item() for k, v in result.items()}, n=B)
        preds = (out["logit"] > 0).long().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().tolist())

    avgs = bag.avg()
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    logger.info(
        f"[val]   epoch={epoch}  loss={avgs['loss']:.4f}  "
        f"F1={f1:.4f}  Acc={acc:.4f}"
    )
    return avgs, f1, acc

def main():
    args = parse_args()
    cfg = load_config(args.config)
    ablation = build_ablation(cfg, args.ablation)

    tag = args.ablation or "full"
    ckpt_dir = Path(cfg.paths.checkpoint_dir) / cfg.data.dataset / tag
    log_path = Path(cfg.paths.log_dir) / cfg.data.dataset / f"{tag}.log"

    logger = build_logger(log_path)
    logger.info(f"{'='*50}")
    logger.info(f"Run: dataset={cfg.data.dataset}  ablation={tag}")
    logger.info(f"Ablation flags: {ablation}")

    set_seed(cfg.project.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    cache_dir = Path(cfg.paths.cache_dir) / cfg.data.dataset

    current_fold = get_current_fold()
    if current_fold is not None:
        logger.info(f"K-fold mode: fold={current_fold}")
        train_set = KFoldRaceDataset(
            cache_dir, "train", d_c=cfg.dims.d_c, fold=current_fold, logger=logger
        )
        val_set = KFoldRaceDataset(
            cache_dir, "val", d_c=cfg.dims.d_c, fold=current_fold, logger=logger
        )
        tag = f"{tag}_fold{current_fold}"
        ckpt_dir = Path(cfg.paths.checkpoint_dir) / cfg.data.dataset / tag
        log_path = Path(cfg.paths.log_dir) / cfg.data.dataset / f"{tag}.log"
        logger = build_logger(log_path)
    else:
        train_set = RaceDataset(cache_dir, "train", d_c=cfg.dims.d_c, logger=logger)
        val_set   = RaceDataset(cache_dir, "val",   d_c=cfg.dims.d_c, logger=logger)
    train_loader = build_dataloader(
        train_set, cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers,pin_memory=(device.type == "cuda"),
        seed=cfg.project.seed,
    )
    val_loader = build_dataloader(
        val_set, cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = RACEModel(
        d_c=cfg.dims.d_c, d_l=cfg.dims.d_l,
        d_g=cfg.dims.d_g, d_f=cfg.dims.d_f,
        d_r=cfg.dims.d_r, gamma=cfg.hparams.gamma,
        delta=cfg.hparams.delta, dropout=cfg.hparams.dropout,
        ablation=ablation,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)

    pw = None
    if getattr(cfg.train, "use_pos_weight", False):
        pw = torch.tensor([train_set.pos_weight()], device=device)
    loss_fn = RACELoss(
        delta=cfg.hparams.delta,
        beta=cfg.hparams.beta,
        lambda_real=cfg.hparams.lambda_real,
        pos_weight=pw,
    )

    start_epoch = 1
    best_f1 = 0.0
    patience_counter = 0

    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scheduler"] and scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_metric", 0.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}  best_F1={best_f1:.4f}")

    patience = getattr(cfg.train, "patience", 10)
    use_tqdm = getattr(cfg.train, "use_tqdm", True)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            ablation, epoch, cfg.train.epochs, logger, use_tqdm,
        )
        val_avgs, val_f1, val_acc = evaluate(
            model, val_loader, loss_fn, device, ablation, epoch, logger
        )
        scheduler.step()

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            ckpt_dir / "last.pt", epoch,
            model.state_dict(), optimizer.state_dict(),
            scheduler.state_dict(), best_f1,
            {"dataset": cfg.data.dataset, "ablation": tag},
        )
        if is_best:
            save_checkpoint(
                ckpt_dir / "best.pt", epoch,
                model.state_dict(), optimizer.state_dict(),
                scheduler.state_dict(), best_f1,
                {"dataset": cfg.data.dataset, "ablation": tag},
            )
            logger.info(f"  ★ new best  F1={best_f1:.4f}  saved -> {ckpt_dir}/best.pt")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    logger.info(f"Training done. best_val_F1={best_f1:.4f}")

if __name__ == "__main__":
    main()