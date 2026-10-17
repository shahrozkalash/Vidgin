"""Pytest configuration and custom markers."""

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring a CUDA GPU (automatically skipped if CUDA is absent)",
    )


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA GPU not available in this environment")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
