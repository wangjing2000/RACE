from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_run_log_path(log_dir: Path, dataset: str, experiment: str) -> Path:
    run_dir = Path(log_dir) / dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f"{experiment}.log"

def setup_logger(
    name: str,
    log_file: Path,
    resume: bool = False,
    level: int = logging.INFO,
    to_console: bool = True,
) -> logging.Logger:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    file_mode = "a" if (resume and log_file.exists()) else "w"
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    if file_mode == "a":
        logger.info("=" * 60)
        logger.info("=== 恢复训练，续接已有日志 ===")

    return logger

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)