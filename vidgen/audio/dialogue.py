"""Timed dialogue tracks for videos, voiced by Vidgin's own TTS + vocoder.

A video gets one or more dialogue LINES, each with a start time in
seconds. :func:`build_speech_track` lays them on a single track the
length of the video:

* lines never overlap (one voice): a line that would start inside the
  previous one is pushed back to just after it (``shifted``);
* a line longer than its slot (until the next line's requested start or
  the video's end) is re-synthesized FASTER through the voice model's
  duration scale, never below ``min_speed_scale`` so speech stays
  intelligible. If it still overruns, it is NOT cut mid-sentence for the
  next line — that line is pushed back instead. Only the end of the video
  cuts speech (with a short fade);
* every placement decision is returned in a report, so nothing is
  silently dropped or truncated (the old single-line path cut overlong
  speech at the video's end without saying so).

The synthesizer is passed in as a callable, so the placement logic is
independent of model loading (and testable without trained checkpoints).

Line syntax (``--line`` / dialogue files): ``"START: TEXT"``, e.g.
``"1.5: Where are we going?"``. Dialogue files hold one line per row;
blank rows and rows starting with ``#`` are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

SAMPLE_RATE = 22050
LINE_GAP_SEC = 0.15       # silence kept between consecutive lines
FADE_SEC = 0.02           # fade-out applied to a cut line


@dataclass(frozen=True)
class DialogueLine:
    start: float
    text: str


def parse_dialogue_line(spec: str) -> DialogueLine:
    """Parse ``"START: TEXT"`` (START in seconds, >= 0)."""
    head, sep, text = spec.partition(":")
    try:
        start = float(head.strip()) if sep else float("nan")
    except ValueError:
        start = float("nan")
    text = text.strip()
    if not sep or not np.isfinite(start) or start < 0 or not text:
        raise ValueError(
            f"bad dialogue line {spec!r}: expected 'START: TEXT' with START "
            f"in seconds >= 0, e.g. '1.5: Where are we going?'")
    return DialogueLine(start, text)


def load_dialogue_file(path: str) -> list[DialogueLine]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for lineno, row in enumerate(f, start=1):
            row = row.strip()
            if not row or row.startswith("#"):
                continue
            try:
                lines.append(parse_dialogue_line(row))
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return lines


def build_speech_track(lines: list[DialogueLine],
                       synthesize: Callable[[str, float], np.ndarray],
                       target_duration: float,
                       sample_rate: int = SAMPLE_RATE,
                       min_speed_scale: float = 0.6
                       ) -> tuple[np.ndarray, list[dict]]:
    """Place every line on a track of ``target_duration`` seconds.

    ``synthesize(text, duration_scale) -> float32 waveform``; a
    duration_scale below 1 speaks faster. Returns ``(track, report)``.
    """
    if not 0 < min_speed_scale <= 1:
        raise ValueError(f"min_speed_scale must be in (0, 1], got {min_speed_scale}")
    total = max(1, int(round(target_duration * sample_rate)))
    track = np.zeros(total, dtype=np.float32)
    ordered = sorted(lines, key=lambda ln: ln.start)
    report = []
    cursor = 0                      # first free sample after the last line
    gap = int(round(LINE_GAP_SEC * sample_rate))
    for i, line in enumerate(ordered):
        want = int(round(line.start * sample_rate))
        start = max(want, cursor)
        nxt = (int(round(ordered[i + 1].start * sample_rate))
               if i + 1 < len(ordered) else total)
        slot = min(total, nxt - gap) - start      # time before the next line
        room = total - start                       # time before the video ends
        entry = {"text": line.text, "requested_start": line.start,
                 "start": start / sample_rate, "shifted": start != want,
                 "speed_scale": 1.0, "truncated_sec": 0.0, "skipped": False}
        if room <= 0:
            entry.update(skipped=True, end=entry["start"])
            report.append(entry)
            continue
        wav = np.asarray(synthesize(line.text, 1.0), dtype=np.float32)
        fit = min(slot, room)
        if len(wav) > fit:
            scale = max(min_speed_scale, max(fit, 1) / len(wav))
            if scale < 1.0:
                wav = np.asarray(synthesize(line.text, scale), dtype=np.float32)
                entry["speed_scale"] = round(scale, 3)
        if len(wav) > room:                        # only the video end cuts
            entry["truncated_sec"] = round((len(wav) - room) / sample_rate, 3)
            wav = wav[:room].copy()
            fade = min(len(wav), int(FADE_SEC * sample_rate))
            if fade:
                wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        track[start:start + len(wav)] = wav
        cursor = start + len(wav) + gap
        entry["end"] = (start + len(wav)) / sample_rate
        report.append(entry)
    return track, report


def describe_report(report: list[dict]) -> list[str]:
    """Human-readable warnings for anything that did not go as written."""
    notes = []
    for r in report:
        tag = f"line at {r['requested_start']:.2f}s ({r['text'][:40]!r})"
        if r["skipped"]:
            notes.append(f"{tag}: SKIPPED, no time left in the video")
            continue
        if r["shifted"]:
            notes.append(f"{tag}: moved to {r['start']:.2f}s to avoid "
                         f"overlapping the previous line")
        if r["speed_scale"] < 1.0:
            notes.append(f"{tag}: spoken faster (duration x"
                         f"{r['speed_scale']}) to fit its slot")
        if r["truncated_sec"] > 0:
            notes.append(f"{tag}: last {r['truncated_sec']:.2f}s cut — it "
                         f"runs past the end of the video")
    return notes


def load_voice_synthesizer(voice_ckpt: str, vocoder_ckpt: str,
                           device: str = "cpu"
                           ) -> Callable[[str, float], np.ndarray]:
    """Load the voice TTS + vocoder ONCE; return ``synthesize(text, scale)``."""
    import torch

    from vidgen.audio import CHECKPOINT_CLASSES
    from vidgen.audio.vocoder import HiFiGANGenerator, VocoderConfig
    from vidgen.audio.voice_model import VidginVoiceTTS, VoiceModelConfig
    from vidgen.checkpoint import load_checkpoint

    dev = torch.device(device)
    vp = load_checkpoint(voice_ckpt, map_location=str(dev),
                         safe_classes=CHECKPOINT_CLASSES)
    voice = VidginVoiceTTS(vp.get("cfg", VoiceModelConfig())).to(dev)
    voice.load_state_dict(vp["model"])
    voice.eval()
    gp = load_checkpoint(vocoder_ckpt, map_location=str(dev),
                         safe_classes=CHECKPOINT_CLASSES)
    vocoder = HiFiGANGenerator(gp.get("cfg", VocoderConfig())).to(dev)
    vocoder.load_state_dict(gp["generator"])
    vocoder.eval()

    @torch.no_grad()
    def synthesize(text: str, duration_scale: float = 1.0) -> np.ndarray:
        mel = voice.synthesize(text, duration_scale=duration_scale,
                               device=str(dev))           # [1, 80, frames]
        return vocoder(mel).squeeze().float().cpu().numpy()

    return synthesize
