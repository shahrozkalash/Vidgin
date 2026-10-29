"""Part 7: structured JSONL logging."""

from __future__ import annotations

import json
import time


class StructuredLogger:
    """Append-only JSONL logger. Every record gets a step and a timestamp."""

    def __init__(self, path: str, echo: bool = True):
        self.path = path
        self.echo = echo
        self._fh = open(path, "a")

    def log(self, record: dict, echo: bool | None = None) -> dict:
        """Append ``record``; ``echo`` overrides the logger's console echo
        for this one record (probe records print their own summary)."""
        record = dict(record)
        record.setdefault("timestamp", time.time())
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        if self.echo if echo is None else echo:
            step = record.get("step", "?")
            msg = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in record.items()
                if k in ("loss", "recon", "kl", "lr", "grad_norm")
            )
            print(f"[step {step}] {msg}", flush=True)
        return record

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
