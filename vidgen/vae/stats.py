"""Measured VAE latent statistics and reconstruction quality.

Phase 3 of the targeted improvement programme (2026-10-22). The DiT
trains on VAE latents multiplied by ``VidgenConfig.latent_scale``,
which used to be a hand-set constant (1.0) that was never measured
against real data. Diffusion/flow training works best when the latents
it sees are roughly unit-variance, so the honest workflow is:

1. train (or warm-start) the VAE,
2. MEASURE the latent distribution it actually produces on the target
   dataset with :func:`measure_latent_stats`,
3. set ``latent_scale = 1 / global_std`` (the report's
   ``recommended_latent_scale``) via ``train.py --latent-scale`` — the
   value is stored in the checkpoint config, so sampling and
   continuation decode with the same scale automatically.

The same pass measures reconstruction quality (MSE / PSNR of
decode(encode(x)) against x) so a VAE that cannot reconstruct its
inputs is caught BEFORE an expensive DiT run is built on its latents.
All metrics are plain statistics — no learned evaluator involved.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def measure_latent_stats(vae, dataset, max_clips: int = 256,
                         batch_size: int = 8, device: str = "cpu") -> dict:
    """Latent distribution + reconstruction quality over dataset clips.

    ``vae`` is a CausalVAE (eval mode is enforced). ``dataset`` yields
    ``(clip, caption)`` with clip float32 [3, T, H, W] in [-1, 1], the
    FolderClipDataset / MovingMNISTDataset contract. Statistics are
    accumulated streaming, so memory stays at one batch.
    """
    vae.eval()
    n = min(len(dataset), int(max_clips))
    if n <= 0:
        raise ValueError("dataset is empty; nothing to measure")

    sum_c = None       # per-channel sum of mu
    sumsq_c = None     # per-channel sum of mu^2
    count_c = 0        # elements per channel
    mse_sum = 0.0
    mse_count = 0

    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        x = torch.stack([dataset[i][0] for i in idx]).to(device)
        mu, _ = vae.encode(x)
        flat = mu.double().permute(1, 0, 2, 3, 4).reshape(mu.shape[1], -1)
        if sum_c is None:
            sum_c = flat.sum(dim=1)
            sumsq_c = (flat * flat).sum(dim=1)
        else:
            sum_c += flat.sum(dim=1)
            sumsq_c += (flat * flat).sum(dim=1)
        count_c += flat.shape[1]

        recon = vae.decode(mu)
        mse_sum += float(((recon - x) ** 2).sum().double())
        mse_count += x.numel()

    mean_c = sum_c / count_c
    var_c = (sumsq_c / count_c) - mean_c ** 2
    var_c = torch.clamp(var_c, min=0.0)
    std_c = var_c.sqrt()

    global_mean = float(mean_c.mean())
    global_var = float(var_c.mean() + (mean_c ** 2).mean() - global_mean ** 2)
    global_std = math.sqrt(max(global_var, 0.0))
    mse = mse_sum / mse_count
    # Peak signal for data in [-1, 1] is 2, so PSNR = 10*log10(4 / MSE).
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(4.0 / mse)

    return {
        "clips_measured": n,
        "latent_channels": int(mu.shape[1]),
        "mean_per_channel": [round(float(v), 6) for v in mean_c],
        "std_per_channel": [round(float(v), 6) for v in std_c],
        "channel_std_min": round(float(std_c.min()), 6),
        "channel_std_max": round(float(std_c.max()), 6),
        "global_mean": round(global_mean, 6),
        "global_std": round(global_std, 6),
        "recommended_latent_scale": (round(1.0 / global_std, 6)
                                     if global_std > 0 else None),
        "recon_mse": round(mse, 8),
        "recon_psnr_db": (round(psnr, 3) if math.isfinite(psnr) else None),
    }


@torch.no_grad()
def save_recon_preview(vae, clip: torch.Tensor, path: str,
                       device: str = "cpu") -> str:
    """Side-by-side PNG: input frames (top) vs reconstruction (bottom).

    ``clip`` is one dataset sample [3, T, H, W] in [-1, 1]. Frames are
    laid out left-to-right; the image is scaled x4 with nearest
    neighbour so small clips stay inspectable.
    """
    from PIL import Image

    vae.eval()
    x = clip.unsqueeze(0).to(device)
    mu, _ = vae.encode(x)
    recon = vae.decode(mu)[0].clamp(-1, 1)
    src = clip.clamp(-1, 1)

    def to_frame(t: torch.Tensor) -> "Image.Image":
        arr = ((t.cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).astype("uint8")
        return Image.fromarray(arr)

    frames_in = [to_frame(src[:, f]) for f in range(src.shape[1])]
    frames_out = [to_frame(recon[:, f]) for f in range(recon.shape[1])]
    w, h = frames_in[0].size
    sheet = Image.new("RGB", (w * len(frames_in), h * 2), (24, 24, 24))
    for i, (a, b) in enumerate(zip(frames_in, frames_out)):
        sheet.paste(a, (i * w, 0))
        sheet.paste(b, (i * w, h))
    sheet = sheet.resize((sheet.width * 4, sheet.height * 4),
                         Image.NEAREST)
    sheet.save(path)
    return path
