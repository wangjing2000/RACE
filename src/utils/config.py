from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class _AttrDict(dict):
    def __init__(self, data: Dict[str, Any] | None = None):
        super().__init__()
        data = data or {}
        for key, value in data.items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return _AttrDict(value)
        if isinstance(value, list):
            return [_AttrDict._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"配置项 '{name}' 不存在，请检查 config.yaml"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

class Config:
    def __init__(self, raw: Dict[str, Any], config_path: Path):
        self._raw = copy.deepcopy(raw)
        self.config_path = Path(config_path)
        self.root = PROJECT_ROOT
        self._cfg = _AttrDict(raw)

    def __getattr__(self, name: str) -> Any:
        cfg = self.__dict__.get("_cfg")
        if cfg is None:
            raise AttributeError(name)
        return getattr(cfg, name)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._raw)

    def _abs(self, *parts: str) -> Path:
        return (self.root.joinpath(*parts)).resolve()

    def path(self, key: str) -> Path:
        return self._abs(self._cfg.paths[key])

    def dataset_names(self) -> List[str]:
        return list(self._cfg.datasets.keys())

    def language(self, dataset: str) -> str:
        self._assert_dataset(dataset)
        return self._cfg.datasets[dataset].language

    def dataset_dir(self, dataset: str) -> Path:
        self._assert_dataset(dataset)
        return self._abs(self._cfg.paths.data_dir, dataset)

    def json_path(self, dataset: str) -> Path:
        self._assert_dataset(dataset)
        return self.dataset_dir(dataset) / self._cfg.datasets[dataset].json_name

    def images_dir(self, dataset: str) -> Path:
        self._assert_dataset(dataset)
        return self.dataset_dir(dataset) / self._cfg.datasets[dataset].images_dirname

    def cache_dir(self, dataset: str) -> Path:
        return self._abs(self._cfg.paths.cache_dir, dataset)

    def clip_path(self, dataset: str) -> Path:
        lang = self.language(dataset)
        return self._abs(self._cfg.paths.pretrained_dir,
                         self._cfg.models.clip[lang])

    def ner_path(self, dataset: str) -> Path:
        lang = self.language(dataset)
        return self._abs(self._cfg.paths.pretrained_dir,
                         self._cfg.models.ner[lang])

    def faster_rcnn_path(self) -> Path:
        return self._abs(self._cfg.paths.pretrained_dir,
                         self._cfg.models.faster_rcnn)

    def paddleocr_paths(self, dataset: str) -> Dict[str, Path]:
        lang = self.language(dataset)
        base = self._abs(self._cfg.paths.pretrained_dir, "paddleocr")
        sub = self._cfg.models.paddleocr[lang]
        return {
            "det_model_dir": base / sub.det,
            "rec_model_dir": base / sub.rec,
            "cls_model_dir": base / sub.cls,
        }

    def ablation(self, name: str) -> _AttrDict:
        if name not in self._cfg.ablation:
            raise KeyError(
                f"消融组 '{name}' 不存在，可选: {list(self._cfg.ablation.keys())}"
            )
        return self._cfg.ablation[name]

    def ablation_names(self) -> List[str]:
        return list(self._cfg.ablation.keys())

    def _assert_dataset(self, dataset: str) -> None:
        if dataset not in self._cfg.datasets:
            raise KeyError(
                f"数据集 '{dataset}' 未在 config.yaml 定义，"
                f"可选: {self.dataset_names()}"
            )

    def validate_pretrained(self, dataset: str) -> List[str]:
        missing: List[str] = []
        checks = {
            "CLIP": self.clip_path(dataset),
            "NER": self.ner_path(dataset),
            "FasterRCNN": self.faster_rcnn_path(),
        }
        checks.update(
            {f"OCR/{k}": v for k, v in self.paddleocr_paths(dataset).items()}
        )
        for name, p in checks.items():
            if not p.exists():
                missing.append(f"{name}: {p}")
        return missing

def load_config(config_path: str | Path = "config.yaml") -> Config:
    path = Path(config_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("config.yaml 解析结果不是字典，请检查格式")

    return Config(raw, path)