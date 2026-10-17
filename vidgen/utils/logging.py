"""Structured experiment logging with JSON-L and console formatting."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    """Experiment logger writing to console and JSON-lines logfile."""

    def __init__(self, log_dir: Optional[str | Path] = None, name: str = "vidgen"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Console handler with clean formatting
        c_handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        c_handler.setFormatter(fmt)
        self.logger.addHandler(c_handler)

        self.jsonl_file = None
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.jsonl_file = log_dir / "metrics.jsonl"
            f_handler = logging.FileHandler(log_dir / "train.log")
            f_handler.setFormatter(fmt)
            self.logger.addHandler(f_handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log structured scalar metrics to console and JSON-lines."""
        payload = {"step": step, **metrics}
        line = " | ".join(f"{k}: {v:.4e}" if isinstance(v, float) else f"{k}: {v}" for k, v in payload.items())
        self.logger.info(line)

        if self.jsonl_file is not None:
            with open(self.jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
