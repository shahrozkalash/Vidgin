"""Log-mel spectrogram utilities for Vidgin audio models.

Standard audio parameters across Vidgin:
- Sample rate: 22050 Hz
- Mel channels: 80
- FFT size: 1024
- Hop length: 256
- Window length: 1024 (Hann window)
- Frequency range: 0 Hz - 8000 Hz

Implemented in pure PyTorch / math with no external audio libraries required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 0.0
    f_max: float = 8000.0
    clip_val: float = 1e-5


DEFAULT_AUDIO_CFG = AudioConfig()


def hz_to_mel(frequencies: float | torch.Tensor) -> float | torch.Tensor:
    """Convert Hz to Mel scale using standard Slaney formula."""
    if torch is not None and isinstance(frequencies, torch.Tensor):
        return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
    return 2595.0 * math.log10(1.0 + frequencies / 700.0)


def mel_to_hz(mels: float | torch.Tensor) -> float | torch.Tensor:
    """Convert Mel scale back to Hz."""
    if torch is not None and isinstance(mels, torch.Tensor):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def build_mel_basis(
    sr: int = 22050,
    n_fft: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Construct triangular Mel filterbank matrix [n_mels, n_fft // 2 + 1].

    Self-contained pure PyTorch implementation matching standard librosa / torchaudio
    Slaney filterbank with triangle area normalization.
    """
    if torch is None:
        raise RuntimeError("torch is required to construct mel basis tensor")

    num_freq_bins = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sr / 2.0, num_freq_bins, device=device)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max if f_max is not None else sr / 2.0)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz_points = mel_to_hz(mel_points)

    filterbank = torch.zeros(n_mels, num_freq_bins, device=device)

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # Up slope
        up_mask = (fft_freqs >= left) & (fft_freqs <= center)
        if (center - left) > 0:
            filterbank[i, up_mask] = (fft_freqs[up_mask] - left) / (center - left)

        # Down slope
        down_mask = (fft_freqs >= center) & (fft_freqs <= right)
        if (right - center) > 0:
            filterbank[i, down_mask] = (right - fft_freqs[down_mask]) / (right - center)

        # Slaney area normalization: 2.0 / (right - left)
        enorm = 2.0 / torch.clamp(right - left, min=1e-5)
        filterbank[i] *= enorm

    return filterbank


_MEL_BASIS_CACHE: dict[tuple, torch.Tensor] = {}


def get_cached_mel_basis(
    sr: int = 22050,
    n_fft: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    dev_key = str(device) if device else "cpu"
    key = (sr, n_fft, n_mels, f_min, f_max, dev_key)
    if key not in _MEL_BASIS_CACHE:
        _MEL_BASIS_CACHE[key] = build_mel_basis(
            sr=sr, n_fft=n_fft, n_mels=n_mels, f_min=f_min, f_max=f_max, device=device
        )
    return _MEL_BASIS_CACHE[key]


def wav_to_mel(
    waveform: torch.Tensor,
    cfg: AudioConfig = DEFAULT_AUDIO_CFG,
    mel_basis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert 1D/2D audio waveform in [-1.0, 1.0] to log-mel spectrogram [B, n_mels, frames].

    Args:
        waveform: Float tensor [B, T] or [T]
        cfg: AudioConfig parameters
        mel_basis: Optional precomputed filterbank [n_mels, n_fft // 2 + 1]

    Returns:
        Log-mel spectrogram [B, n_mels, frames]
    """
    if torch is None:
        raise RuntimeError("torch is required for wav_to_mel")

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    device = waveform.device
    window = torch.hann_window(cfg.win_length, device=device)

    # Center-padded STFT
    # waveform: [B, T] -> stft: [B, n_fft // 2 + 1, frames]
    stft = torch.stft(
        waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitudes = torch.abs(stft)  # [B, freq_bins, frames]

    if mel_basis is None:
        mel_basis = get_cached_mel_basis(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            device=device,
        )
    else:
        mel_basis = mel_basis.to(device)

    # [n_mels, freq_bins] @ [B, freq_bins, frames] -> [B, n_mels, frames]
    mel_spectrogram = torch.matmul(mel_basis, magnitudes)
    log_mel = torch.log(torch.clamp(mel_spectrogram, min=cfg.clip_val))
    return log_mel


def mel_to_linear(
    log_mel: torch.Tensor,
    cfg: AudioConfig = DEFAULT_AUDIO_CFG,
    mel_basis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Approximate linear STFT magnitude from log-mel spectrogram via pseudo-inverse."""
    if torch is None:
        raise RuntimeError("torch is required for mel_to_linear")

    device = log_mel.device
    if mel_basis is None:
        mel_basis = get_cached_mel_basis(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            device=device,
        )
    else:
        mel_basis = mel_basis.to(device)

    mel_spec = torch.exp(log_mel)
    # Pseudo-inverse of mel basis: [freq_bins, n_mels]
    pinv = torch.pinverse(mel_basis)
    linear_mag = torch.matmul(pinv, mel_spec)
    return torch.clamp(linear_mag, min=0.0)


def mel_to_wav_griffin_lim(
    log_mel: torch.Tensor,
    cfg: AudioConfig = DEFAULT_AUDIO_CFG,
    n_iters: int = 32,
) -> torch.Tensor:
    """Reconstruct audio waveform from log-mel spectrogram using Griffin-Lim.

    Useful as a fallback or sanity preview when vocoder weights are not yet loaded.
    """
    if torch is None:
        raise RuntimeError("torch is required for Griffin-Lim")

    device = log_mel.device
    linear_mag = mel_to_linear(log_mel, cfg=cfg)
    window = torch.hann_window(cfg.win_length, device=device)

    # Initialize with random angles
    angles = torch.exp(2j * math.pi * torch.rand_like(linear_mag))
    complex_spec = linear_mag * angles

    for _ in range(n_iters):
        # ISTFT
        wav = torch.istft(
            complex_spec,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            center=True,
        )
        # STFT
        stft = torch.stft(
            wav,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        angles = stft / torch.clamp(torch.abs(stft), min=1e-8)
        complex_spec = linear_mag * angles

    final_wav = torch.istft(
        complex_spec,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
    )
    return torch.clamp(final_wav, -1.0, 1.0)


def normalize_mel(mel: torch.Tensor, min_db: float = -11.51, max_db: float = 2.0) -> torch.Tensor:
    """Normalize log-mel from natural range [min_db, max_db] to [-1.0, 1.0] for diffusion."""
    norm = 2.0 * (mel - min_db) / (max_db - min_db) - 1.0
    return torch.clamp(norm, -1.0, 1.0)


def denormalize_mel(mel_norm: torch.Tensor, min_db: float = -11.51, max_db: float = 2.0) -> torch.Tensor:
    """Denormalize diffusion output [-1.0, 1.0] back to natural log-mel range."""
    return ((mel_norm + 1.0) * 0.5) * (max_db - min_db) + min_db
