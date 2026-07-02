#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE

from src.data import RaceDataset, build_dataloader
from src.models.race_model import RACEModel, AblationConfig
from train import load_config, build_logger
from evaluate import build_model, _ABLATION_TAGS

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ts = sub.add_parser("tsne")
    ts.add_argument("--config",  required=True)
    ts.add_argument("--ckpt",    default=None)
    ts.add_argument("--ablation", default=None)
    ts.add_argument("--all_ablations", action="store_true")
    ts.add_argument("--split",   default="train")
    ts.add_argument("--n_samples", type=int, default=2000,
                    help="最多取多少样本做 t-SNE（过多会很慢）")
    ts.add_argument("--out",     default=None)

    hm = sub.add_parser("heatmap")
    hm.add_argument("--config",  required=True)
    hm.add_argument("--ckpt",    default=None)
    hm.add_argument("--ablation", default=None)
    hm.add_argument("--split",   default="test")
    hm.add_argument("--sample_idx", type=int, default=0)
    hm.add_argument("--out",     default=None)

    return p.parse_args()

@torch.no_grad()
def collect_features(
    model: RACEModel,
    loader,
    device: torch.device,
    n_samples: int = 9999,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    model.eval()
    feats_list, labels_list, raw = [], [], []
    total = 0

    for batch in loader:
        if total >= n_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch)
        B = batch["label"].shape[0]
        feats_list.append(out["f_fuse"].cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy())
        raw.append({k: out[k].cpu() for k in ("c_ij","attn","pair_mask","g","s_c")})
        total += B

    feats  = np.concatenate(feats_list, axis=0)[:n_samples]
    labels = np.concatenate(labels_list, axis=0)[:n_samples]
    return feats, labels, raw

def plot_tsne_single(
    feats: np.ndarray,
    labels: np.ndarray,
    ax: plt.Axes,
    title: str,
    seed: int = 66,
):
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, n_iter=1000)
    proj = tsne.fit_transform(feats)

    colors = {0: "#098DF8", 1: "#F9291A"}
    names  = {0: "real", 1: "rumor"}
    for cls in (0, 1):
        mask = labels == cls
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=colors[cls], label=names[cls],
            s=10, alpha=0.7, linewidths=0,
        )
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=2)
    ax.axis("off")

def cmd_tsne(args, cfg, device, logger):
    cache_dir = Path(cfg.paths.cache_dir) / cfg.data.dataset
    dataset   = RaceDataset(cache_dir, args.split, d_c=cfg.dims.d_c, logger=logger)
    loader    = build_dataloader(
        dataset, cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    ckpt_root = Path(cfg.paths.checkpoint_dir) / cfg.data.dataset

    if not args.all_ablations:
        tag       = args.ablation or "full"
        ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_root / tag / "best.pt"
        ablation  = AblationConfig(**({} if tag == "full" else {f"use_{tag[3:]}": False}))
        model     = build_model(cfg, ablation, ckpt_path, device)
        feats, labels, _ = collect_features(model, loader, device, args.n_samples)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plot_tsne_single(feats, labels, ax, f"t-SNE [{tag}]")
        out_path = Path(args.out) if args.out else \
            Path(cfg.paths.log_dir) / cfg.data.dataset / f"tsne_{tag}.png"
        _save(fig, out_path, logger)
        return

    tags_available = []
    for tag, ab_kwargs in _ABLATION_TAGS:
        ckpt_path = ckpt_root / tag / "best.pt"
        if ckpt_path.exists():
            tags_available.append((tag, ab_kwargs, ckpt_path))

    n = len(tags_available)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes_flat  = np.array(axes).flatten()

    for i, (tag, ab_kwargs, ckpt_path) in enumerate(tags_available):
        ablation = AblationConfig(**ab_kwargs)
        model    = build_model(cfg, ablation, ckpt_path, device)
        feats, labels, _ = collect_features(model, loader, device, args.n_samples)
        plot_tsne_single(feats, labels, axes_flat[i], f"t-SNE [{tag}]")
        logger.info(f"  t-SNE done: {tag}")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"t-SNE Feature Comparison — {cfg.data.dataset}", fontsize=13)
    fig.tight_layout()
    out_path = Path(args.out) if args.out else \
               Path(cfg.paths.log_dir) / cfg.data.dataset / "tsne_all_ablations.png"
    _save(fig, out_path, logger)

def _get_sample(raw_batches: List[Dict], idx: int) -> Optional[Dict]:
    cursor = 0
    for batch in raw_batches:
        B = batch["c_ij"].shape[0]
        if cursor + B > idx:
            b = idx - cursor
            return {k: v[b] for k, v in batch.items()}
        cursor += B
    return None

def plot_heatmap(
    c_ij: torch.Tensor,
    attn: torch.Tensor,
    pair_mask: torch.Tensor,
    ax_c: plt.Axes,
    ax_a: plt.Axes,
    title_prefix: str,
):
    c_np = c_ij.numpy().copy()
    a_np = attn.numpy().copy()
    pm   = pair_mask.numpy()

    c_np[~pm] = np.nan
    a_np[~pm] = np.nan

    for ax, data, cmap, title, vmax in [
        (ax_c, c_np, "Reds",   f"{title_prefix} — c_ij (conflict)", 2.0),
        (ax_a, a_np, "Blues",  f"{title_prefix} — attn",None),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="auto",
                       vmin=0, vmax=vmax, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("visual tokens (region | ocr)", fontsize=8)
        ax.set_ylabel("text tokens", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

def cmd_heatmap(args, cfg, device, logger):
    cache_dir = Path(cfg.paths.cache_dir) / cfg.data.dataset
    dataset   = RaceDataset(cache_dir, args.split, d_c=cfg.dims.d_c, logger=logger)
    loader    = build_dataloader(
        dataset, cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    tag       = args.ablation or "full"
    ckpt_root = Path(cfg.paths.checkpoint_dir) / cfg.data.dataset
    ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_root / tag / "best.pt"
    ab_kwargs = {} if tag == "full" else {f"use_{tag[3:]}": False}
    ablation  = AblationConfig(**ab_kwargs)
    model     = build_model(cfg, ablation, ckpt_path, device)

    _, _, raw = collect_features(model, loader, device, n_samples=args.sample_idx + 1)
    sample    = _get_sample(raw, args.sample_idx)
    if sample is None:
        logger.error(f"sample_idx={args.sample_idx} 超出数据集范围")
        return

    c_ij      = sample["c_ij"]
    attn      = sample["attn"]
    pair_mask = sample["pair_mask"]

    if c_ij.shape[0] == 0 or c_ij.shape[1] == 0:
        logger.warning("c_ij 为空（消融导致 N_t=0 或 N_v=0），跳过热图")
        return

    fig, (ax_c, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
    plot_heatmap(c_ij, attn, pair_mask, ax_c, ax_a,
                 title_prefix=f"sample={args.sample_idx} [{tag}]")
    fig.tight_layout()

    out_path = Path(args.out) if args.out else \
               Path(cfg.paths.log_dir) / cfg.data.dataset / \
               f"heatmap_{tag}_s{args.sample_idx}.pdf"
    _save(fig, out_path, logger)

def _save(fig: plt.Figure, out_path: Path, logger):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"图像已保存 -> {out_path}")

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    out_dir  = Path(cfg.paths.log_dir) / cfg.data.dataset
    logger   = build_logger(out_dir / "visualize.log")

    if args.cmd == "tsne":
        cmd_tsne(args, cfg, device, logger)
    elif args.cmd == "heatmap":
        cmd_heatmap(args, cfg, device, logger)

if __name__ == "__main__":
    main()