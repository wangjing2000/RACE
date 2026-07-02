from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


def generate_kfold_split(
    dataset_dir: str | Path,
    k: int = 10,
    seed: int = 66,
    val_ratio: float = 0.1,
) -> None:
    import numpy as np

    dataset_dir = Path(dataset_dir)
    feat_dir = dataset_dir / "features"

    all_ids = []
    for f in feat_dir.glob("*.npz"):
        all_ids.append(f.stem)

    n_samples = len(all_ids)
    print(f"总样本数: {n_samples}")

    pos_ids = []
    neg_ids = []

    for sid in all_ids:
        npz_path = feat_dir / f"{sid}.npz"
        try:
            with np.load(npz_path) as d:
                if int(d["label"]) == 1:
                    pos_ids.append(sid)
                else:
                    neg_ids.append(sid)
        except Exception:
            neg_ids.append(sid)

    print(f"正样本(rumor): {len(pos_ids)}, 负样本(non-rumor): {len(neg_ids)}")

    rng = random.Random(seed)
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    kfold_dir = dataset_dir / "kfold"
    kfold_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(k):
        pos_fold_size = len(pos_ids) // k
        pos_test_start = fold * pos_fold_size
        pos_test_end = (fold + 1) * pos_fold_size if fold < k - 1 else len(pos_ids)

        neg_fold_size = len(neg_ids) // k
        neg_test_start = fold * neg_fold_size
        neg_test_end = (fold + 1) * neg_fold_size if fold < k - 1 else len(neg_ids)

        test_ids = set(pos_ids[pos_test_start:pos_test_end] + neg_ids[neg_test_start:neg_test_end])

        train_val_pos = [sid for sid in pos_ids if sid not in test_ids]
        train_val_neg = [sid for sid in neg_ids if sid not in test_ids]

        val_pos_size = int(len(train_val_pos) * val_ratio)
        val_neg_size = int(len(train_val_neg) * val_ratio)

        val_ids = set(train_val_pos[:val_pos_size] + train_val_neg[:val_neg_size])
        train_ids = set(train_val_pos[val_pos_size:] + train_val_neg[val_neg_size:])

        split_map: Dict[str, str] = {}
        for sid in all_ids:
            if sid in test_ids:
                split_map[sid] = "test"
            elif sid in val_ids:
                split_map[sid] = "val"
            else:
                split_map[sid] = "train"

        split_path = kfold_dir / f"split_fold{fold}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_map, f, indent=2, ensure_ascii=False)

        print(f"\nFold {fold}:")
        print(f"  Train: {len(train_ids)} ({len(train_ids)-len([s for s in train_ids if s in pos_ids])} non-rumor, {len([s for s in train_ids if s in pos_ids])} rumor)")
        print(f"  Val:   {len(val_ids)} ({len(val_ids)-len([s for s in val_ids if s in pos_ids])} non-rumor, {len([s for s in val_ids if s in pos_ids])} rumor)")
        print(f"  Test:  {len(test_ids)} ({len(test_ids)-len([s for s in test_ids if s in pos_ids])} non-rumor, {len([s for s in test_ids if s in pos_ids])} rumor)")


def get_kfold_split_path(dataset_dir: str | Path, fold: int) -> Path:
    return Path(dataset_dir) / "kfold" / f"split_fold{fold}.json"


def run_kfold_experiment(
    config_path: str,
    k: int = 10,
    ablation: str | None = None,
    resume_fold: int = 0,
) -> None:
    import subprocess
    import sys

    config_path = Path(config_path)

    for fold in range(resume_fold, k):
        print(f"\n{'='*60}")
        print(f"Running Fold {fold}/{k-1}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "main.py", "train",
            "--config", str(config_path),
        ]

        if ablation:
            cmd += ["--ablation", ablation]

        env = os.environ.copy()
        env["RACE_KFOLD_FOLD"] = str(fold)

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"Fold {fold} failed with exit code {result.returncode}")
            break


def collect_kfold_results(
    dataset_name: str,
    k: int = 10,
    ablation: str | None = None,
) -> Dict[str, dict]:
    import numpy as np

    tag = ablation or "full"
    results: Dict[str, list] = {
        "f1": [],
        "acc": [],
        "loss": [],
    }

    for fold in range(k):
        ckpt_dir = Path("checkpoints") / dataset_name / f"{tag}_fold{fold}"
        best_ckpt = ckpt_dir / "best.pt"

        if best_ckpt.exists():
            try:
                import torch
                ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                if "best_metric" in ckpt:
                    results["f1"].append(ckpt["best_metric"])
            except Exception as e:
                print(f"[fold {fold}] checkpoint读取失败，改为从日志读取: {type(e).__name__}")

        log_path = Path("logs") / dataset_name / f"{tag}_fold{fold}.log"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "[val]" in line and "F1=" in line and "Acc=" in line:
                        parts = line.split()
                        for part in parts:
                            if part.startswith("F1="):
                                f1_val = float(part.split("=")[1])
                                if len(results["f1"]) <= fold:
                                    results["f1"].append(f1_val)
                            elif part.startswith("Acc="):
                                acc_val = float(part.split("=")[1])
                                if len(results["acc"]) <= fold:
                                    results["acc"].append(acc_val)
                            elif part.startswith("loss="):
                                loss_val = float(part.split("=")[1])
                                if len(results["loss"]) <= fold:
                                    results["loss"].append(loss_val)

    summary = {}
    for key, values in results.items():
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "all": values,
                "max": float(max(values)),
                "min": float(min(values)),
            }

    return summary


def run_kfold_test(
    dataset_name: str,
    k: int = 10,
    ablation: str | None = None,
    resume_fold: int = 0,
) -> Dict[str, dict]:
    import numpy as np
    import torch
    import logging
    from pathlib import Path

    import sys
    if str(Path(__file__).resolve().parents[2]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.kfold_dataset import KFoldRaceDataset
    from src.data import build_dataloader
    from src.models.race_model import RACEModel, AblationConfig
    from src.utils.config import load_config as _load_config
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

    tag = ablation or "full"
    config_path = Path("configs") / f"{dataset_name.lower()}.yaml"

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    cache_dir = Path("cache") / dataset_name
    ckpt_root = Path(cfg_dict["paths"]["checkpoint_dir"]) / dataset_name
    d_c = int(cfg_dict["dims"]["d_c"])
    d_l = int(cfg_dict["dims"]["d_l"])
    d_g = int(cfg_dict["dims"]["d_g"])
    d_f = int(cfg_dict["dims"]["d_f"])
    d_r = int(cfg_dict["dims"]["d_r"])
    gamma = float(cfg_dict["hparams"]["gamma"])
    delta = float(cfg_dict["hparams"]["delta"])
    batch_size = int(cfg_dict["train"]["batch_size"])
    num_workers = int(cfg_dict["train"]["num_workers"])

    ab_kwargs = {"use_region": True, "use_ocr": True, "use_relevance": True,
                 "use_conflict": True, "use_gate": True, "use_align": True}
    if ablation:
        ab_kwargs[f"use_{ablation[3:]}"] = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    log_dir = Path(cfg_dict["paths"]["log_dir"]) / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"kfold_test_{tag}.log"

    logger = logging.getLogger(f"kfold_test_{dataset_name}_{tag}")
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"开始 K折测试评估：dataset={dataset_name}, tag={tag}, device={device}")
    logger.info(f"config={config_path}")

    fold_results: Dict[str, list] = {
        "Acc": [], "F1": [], "F1_rumor": [], "F1_real": [], "AUC": [],
        "Pre_rumor": [], "Pre_real": [], "Rec_rumor": [], "Rec_real": [],
    }

    for fold in range(resume_fold, k):
        ckpt_path = ckpt_root / f"{tag}_fold{fold}" / "best.pt"
        if not ckpt_path.exists():
            logger.warning(f"[fold {fold}] checkpoint不存在：{ckpt_path}，跳过")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold}/{k-1}")
        logger.info(f"{'='*60}")

        try:
            test_set = KFoldRaceDataset(
                cache_dir=cache_dir, split="test", d_c=d_c,
                fold=fold, logger=logger,
            )
        except Exception as e:
            logger.warning(f"[fold {fold}] 加载测试集失败：{e}，跳过")
            continue

        test_loader = build_dataloader(
            test_set, batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )

        try:
            model = RACEModel(
                d_c=d_c, d_l=d_l, d_g=d_g, d_f=d_f, d_r=d_r,
                gamma=gamma, delta=delta, dropout=0.0,
                ablation=AblationConfig(**ab_kwargs),
            ).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            model.eval()
        except Exception as e:
            logger.warning(f"[fold {fold}] 加载模型失败：{e}，跳过")
            continue

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch)
                all_logits.append(out["logit"].cpu())
                all_labels.append(batch["label"].cpu())

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = 1 / (1 + np.exp(-logits))
        preds = (logits > 0).astype(int)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        f1_r = f1_score(labels, preds, pos_label=1, zero_division=0)
        f1_n = f1_score(labels, preds, pos_label=0, zero_division=0)
        pre_r = precision_score(labels, preds, pos_label=1, zero_division=0)
        pre_n = precision_score(labels, preds, pos_label=0, zero_division=0)
        rec_r = recall_score(labels, preds, pos_label=1, zero_division=0)
        rec_n = recall_score(labels, preds, pos_label=0, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")

        fold_results["Acc"].append(acc)
        fold_results["F1"].append(f1)
        fold_results["F1_rumor"].append(f1_r)
        fold_results["F1_real"].append(f1_n)
        fold_results["AUC"].append(auc)
        fold_results["Pre_rumor"].append(pre_r)
        fold_results["Pre_real"].append(pre_n)
        fold_results["Rec_rumor"].append(rec_r)
        fold_results["Rec_real"].append(rec_n)

        logger.info(
            f"[fold {fold}] Acc={acc:.4f} F1={f1:.4f} "
            f"F1_rumor={f1_r:.4f} F1_real={f1_n:.4f} AUC={auc:.4f} "
            f"Pre_rumor={pre_r:.4f} Pre_real={pre_n:.4f} "
            f"Rec_rumor={rec_r:.4f} Rec_real={rec_n:.4f}"
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"K折测试汇总（共 {len(fold_results['F1'])} 折）")
    logger.info(f"{'='*60}")

    summary = {}
    for metric, values in fold_results.items():
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values))
            summary[metric] = {
                "mean": mean, "std": std,
                "all": values,
                "max": float(max(values)),
                "min": float(min(values)),
            }
            logger.info(
                f"{metric}: {mean*100:.2f}% ± {std*100:.2f}%  "
                f"(range: {min(values)*100:.2f}% ~ {max(values)*100:.2f}%)"
            )
            logger.info(f"  各折: {[f'{v*100:.2f}%' for v in values]}")

    logger.info(f"\n日志已保存：{log_path}")

    import json
    result_json = {
        "dataset": dataset_name,
        "tag": tag,
        "k": k,
        "n_folds": len(fold_results["F1"]),
        "metrics": {k: {"mean": v["mean"], "std": v["std"], "all": v["all"]}
                     for k, v in summary.items()},
    }
    json_path = log_dir / f"kfold_test_{tag}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总 JSON 已保存：{json_path}")

    return summary