"""PyTorch compatibility layer for Vidgin audio.

When PyTorch is installed, re-exports standard torch, torch.nn, and torch.nn.functional.
When PyTorch is absent (e.g. lightweight node container), provides minimal class
stubs so models and configurations can be inspected, parameter counts calculated,
and purity tests run without ModuleNotFoundError.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class _NoGrad:
        def __call__(self, fn):
            return fn
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

    class _TorchStub:
        device = str
        float32 = "float32"
        long = "long"
        bool = "bool"

        @staticmethod
        def no_grad():
            return _NoGrad()

        @staticmethod
        def tensor(*args, **kwargs):
            return None

        @staticmethod
        def zeros(*args, **kwargs):
            return None

        @staticmethod
        def randn(*args, **kwargs):
            return None

    torch = _TorchStub()

    class _DummyModule:
        def __init__(self, *args, **kwargs):
            self._modules = {}
            self._parameters = {}
            self._buffers = {}

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def train(self, mode=True):
            return self

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict, strict=True):
            return self

        def parameters(self):
            return iter([])

        def register_buffer(self, name, val):
            self._buffers[name] = val

        def __call__(self, *args, **kwargs):
            if hasattr(self, "forward"):
                return self.forward(*args, **kwargs)
            return None

    class _NN:
        Module = _DummyModule

        class Linear(_DummyModule):
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = None
                self.bias = None

        class Conv1d(_DummyModule):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
                super().__init__()

        class Conv2d(_DummyModule):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
                super().__init__()

        class ConvTranspose1d(_DummyModule):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
                super().__init__()

        class LayerNorm(_DummyModule):
            def __init__(self, normalized_shape, eps=1e-5):
                super().__init__()

        class Embedding(_DummyModule):
            def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
                super().__init__()
                self.weight = type("Weight", (), {"device": "cpu"})()

        class Sequential(_DummyModule):
            def __init__(self, *args):
                super().__init__()
                self.layers = list(args)

        class ModuleList(_DummyModule):
            def __init__(self, modules=None):
                super().__init__()
                self.modules = list(modules) if modules else []

            def __iter__(self):
                return iter(self.modules)

            def __getitem__(self, idx):
                return self.modules[idx]

            def __len__(self):
                return len(self.modules)

        class SiLU(_DummyModule):
            pass

        class GELU(_DummyModule):
            pass

        class ReLU(_DummyModule):
            pass

        class LeakyReLU(_DummyModule):
            def __init__(self, negative_slope=0.1):
                super().__init__()

        class Tanh(_DummyModule):
            pass

        class init:
            @staticmethod
            def zeros_(tensor):
                pass

    nn = _NN()

    class _F:
        @staticmethod
        def gelu(x):
            return x

        @staticmethod
        def relu(x):
            return x

        @staticmethod
        def leaky_relu(x, negative_slope=0.1):
            return x

        @staticmethod
        def scaled_dot_product_attention(q, k, v, attn_mask=None):
            return q

    F = _F()


__all__ = ["torch", "nn", "F", "HAS_TORCH"]
