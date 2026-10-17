"""Unit tests for StructuredLogger and metric serialization."""

import json
from pathlib import Path
from vidgen.utils.logging import StructuredLogger


def test_structured_logger(tmp_path):
    """Verify structured logger writes to JSONL properly."""
    log_dir = tmp_path / "logs"
    logger = StructuredLogger(log_dir=log_dir, name="test_logger")

    logger.info("Starting test experiment")
    logger.log_metrics(step=1, metrics={"loss": 0.4521, "lr": 1e-4})
    logger.log_metrics(step=2, metrics={"loss": 0.3210, "lr": 1e-4})

    jsonl_file = log_dir / "metrics.jsonl"
    assert jsonl_file.exists()

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 2
    assert lines[0]["step"] == 1
    assert lines[0]["loss"] == pytest_approx(0.4521)
    assert lines[1]["step"] == 2
    assert lines[1]["loss"] == pytest_approx(0.3210)


def pytest_approx(val):
    from pytest import approx
    return approx(val, rel=1e-4)
