#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--config", required=True, help="yaml 配置文件路径")

def _add_ablation(p: argparse.ArgumentParser):
    p.add_argument(
        "--ablation", default=None,
        choices=["wo_region","wo_ocr","wo_relevance",
                 "wo_conflict","wo_gate","wo_align"],
    )

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="main.py",
        description="RACE: Relevance-Aware Conflict Estimation for Multimodal Rumor Detection",
    )
    sub = root.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="训练模型")
    _add_common(tr)
    _add_ablation(tr)
    tr.add_argument("--resume", default=None, help="checkpoint .pt 恢复路径")

    ev = sub.add_parser("evaluate", help="测试集评估")
    _add_common(ev)
    _add_ablation(ev)
    ev.add_argument("--ckpt",   default=None)
    ev.add_argument("--split",  default="test", choices=["val","test"])
    ev.add_argument("--all_ablations", action="store_true", help="输出 7 组对比表")
    ev.add_argument("--out_dir", default=None)

    ts = sub.add_parser("tsne", help="t-SNE 融合特征可视化")
    _add_common(ts)
    _add_ablation(ts)
    ts.add_argument("--ckpt",      default=None)
    ts.add_argument("--split",     default="train", choices=["train","val","test"])
    ts.add_argument("--all_ablations", action="store_true")
    ts.add_argument("--n_samples", type=int, default=2000)
    ts.add_argument("--out",       default=None)

    hm = sub.add_parser("heatmap", help="细粒度冲突热图")
    _add_common(hm)
    _add_ablation(hm)
    hm.add_argument("--ckpt",       default=None)
    hm.add_argument("--split",      default="train", choices=["train","val","test"])
    hm.add_argument("--sample_idx", type=int, default=0)
    hm.add_argument("--out",        default=None)

    return root

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.cmd == "train":
        import train as _train
        sys.argv = ["train.py", "--config", args.config]
        if args.resume:
            sys.argv += ["--resume", args.resume]
        if args.ablation:
            sys.argv += ["--ablation", args.ablation]
        _train.main()

    elif args.cmd == "evaluate":
        import evaluate as _eval
        sys.argv = ["evaluate.py", "--config", args.config,
                    "--split", args.split]
        if args.ckpt:          sys.argv += ["--ckpt", args.ckpt]
        if args.ablation:      sys.argv += ["--ablation", args.ablation]
        if args.all_ablations: sys.argv += ["--all_ablations"]
        if args.out_dir:       sys.argv += ["--out_dir", args.out_dir]
        _eval.main()

    elif args.cmd in ("tsne", "heatmap"):
        import visualize as _vis
        sys.argv = ["visualize.py", args.cmd,
                    "--config", args.config, "--split", args.split]
        if args.ckpt:      sys.argv += ["--ckpt", args.ckpt]
        if args.ablation:  sys.argv += ["--ablation", args.ablation]
        if getattr(args, "all_ablations", False):
            sys.argv += ["--all_ablations"]
        if getattr(args, "n_samples", None):
            sys.argv += ["--n_samples", str(args.n_samples)]
        if getattr(args, "sample_idx", None) is not None:
            sys.argv += ["--sample_idx", str(args.sample_idx)]
        if getattr(args, "out", None):
            sys.argv += ["--out", args.out]
        _vis.main()

if __name__ == "__main__":
    main()