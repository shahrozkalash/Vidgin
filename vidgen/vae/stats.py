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


def ssim_index(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> float:
    """Mean SSIM between two clip batches [B, 3, T, H, W] in [-1, 1].

    Standard SSIM (Wang et al.) on grayscale frames with a uniform
    window — implemented in pure PyTorch so no skimage dependency is
    introduced. SSIM complements PSNR: it tracks structural similarity
    (contrast/structure terms) rather than raw pixel error, so it
    notices blur that MSE tolerates. Neither metric measures perceived
    quality; both are reported, and the side-by-side previews remain
    the ground truth for eyeballs. 1.0 = identical.
    """
    import torch.nn.functional as F

    a = (x.float().mean(dim=1) * 0.5 + 0.5)  # [B, T, H, W] in [0, 1]
    b = (y.float().mean(dim=1) * 0.5 + 0.5)
    bb, tt, h, w = a.shape
    a = a.reshape(bb * tt, 1, h, w)
    b = b.reshape(bb * tt, 1, h, w)
    pad = window // 2
    mu_a = F.avg_pool2d(F.pad(a, [pad] * 4, mode="reflect"), window, stride=1)
    mu_b = F.avg_pool2d(F.pad(b, [pad] * 4, mode="reflect"), window, stride=1)
    var_a = F.avg_pool2d(F.pad(a * a, [pad] * 4, mode="reflect"), window, stride=1) - mu_a ** 2
    var_b = F.avg_pool2d(F.pad(b * b, [pad] * 4, mode="reflect"), window, stride=1) - mu_b ** 2
    cov = F.avg_pool2d(F.pad(a * b, [pad] * 4, mode="reflect"), window, stride=1) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(ssim.mean())


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
    ssim_sum = 0.0
    ssim_batches = 0
    nonfinite_latents = 0   # mu elements that are NaN/Inf (broken VAE)
    nonfinite_recon = 0

    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        x = torch.stack([dataset[i][0] for i in idx]).to(device)
        mu, _ = vae.encode(x)
        nonfinite_latents += int((~torch.isfinite(mu)).sum())
        flat = mu.double().permute(1, 0, 2, 3, 4).reshape(mu.shape[1], -1)
        if sum_c is None:
            sum_c = flat.sum(dim=1)
            sumsq_c = (flat * flat).sum(dim=1)
        else:
            sum_c += flat.sum(dim=1)
            sumsq_c += (flat * flat).sum(dim=1)
        count_c += flat.shape[1]

        recon = vae.decode(mu)
        nonfinite_recon += int((~torch.isfinite(recon)).sum())
        mse_sum += float(((recon - x) ** 2).sum().double())
        mse_count += x.numel()
        if torch.isfinite(recon).all() and torch.isfinite(x).all():
            ssim_sum += ssim_index(recon, x)
            ssim_batches += 1

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
        "recon_ssim": (round(ssim_sum / ssim_batches, 4)
                       if ssim_batches else None),
        "latent_finite": nonfinite_latents == 0,
        "nonfinite_latent_values": nonfinite_latents,
        "nonfinite_recon_values": nonfinite_recon,
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


@torch.no_grad()
def save_recon_mp4(vae, clip: torch.Tensor, path: str, fps: int = 8,
                   device: str = "cpu") -> str:
    """Side-by-side MP4: original (left) vs reconstruction (right).

    A PNG strip shows frames; a playing side-by-side video shows what
    numbers hide — temporal distortion, flicker introduced by the VAE,
    motion smearing. Frames are concatenated along the width, so the
    output is [T, H, 2W, 3]. Written via vidgen.video_io (frame count
    verified on read-back).
    """
    import numpy as np

    from vidgen.video_io import write_mp4

    vae.eval()
    x = clip.unsqueeze(0).to(device)
    mu, _ = vae.encode(x)
    recon = vae.decode(mu)[0].clamp(-1, 1).cpu()
    src = clip.clamp(-1, 1).cpu()
    # [3,T,H,W] -> [T,H,W,3]
    a = src.permute(1, 2, 3, 0).numpy()
    b = recon.permute(1, 2, 3, 0).numpy()
    side = np.concatenate([a, b], axis=2)
    write_mp4(side, path, fps=fps)
    return path
