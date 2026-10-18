"""Part 7: structured logger."""

import json

from vidgen.logging import StructuredLogger


def test_structured_logger(tmp_path):
    p = str(tmp_path / "train.jsonl")
    with StructuredLogger(p) as log:
        log.log({"step": 0, "loss": 1.234, "lr": 1e-4, "grad_norm": 0.9})
        log.log({"step": 1, "loss": 1.100, "lr": 1e-4, "grad_norm": 0.8})
    rows = [json.loads(line) for line in open(p)]
    assert len(rows) == 2
    assert rows[0]["step"] == 0 and rows[1]["step"] == 1
    assert rows[0]["loss"] == 1.234
    assert "timestamp" in rows[0]
