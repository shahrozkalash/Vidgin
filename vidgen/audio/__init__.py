"""Vidgin audio package.

Vidgin's OWN audio brain:
- mel.py: log-mel spectrogram utilities (22050 Hz, 80 mel channels)
- sound_model.py: text-conditioned DiT audio diffusion model (30-80M parameters)
- vocoder.py: in-repo HiFi-GAN mel -> waveform vocoder
- voice_model.py: single-speaker text-to-mel transformer TTS model

100% in-repo code, zero external models, zero downloaded weights.
"""

from __future__ import annotations

from vidgen.audio.mel import (
    AudioConfig,
    build_mel_basis,
    wav_to_mel,
    mel_to_linear,
    mel_to_wav_griffin_lim,
    normalize_mel,
    denormalize_mel,
)
from vidgen.audio.sound_model import (
    AudioDiT,
    AudioDiTConfig,
    count_sound_params,
)
from vidgen.audio.vocoder import (
    HiFiGANGenerator,
    VocoderConfig,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    count_vocoder_params,
)
from vidgen.audio.voice_model import (
    VidginVoiceTTS,
    VoiceModelConfig,
    count_voice_params,
)

__all__ = [
    "AudioConfig",
    "build_mel_basis",
    "wav_to_mel",
    "mel_to_linear",
    "mel_to_wav_griffin_lim",
    "normalize_mel",
    "denormalize_mel",
    "AudioDiT",
    "AudioDiTConfig",
    "count_sound_params",
    "HiFiGANGenerator",
    "VocoderConfig",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "count_vocoder_params",
    "VidginVoiceTTS",
    "VoiceModelConfig",
    "count_voice_params",
]
