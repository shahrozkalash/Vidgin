"""Automated Quick Diagnosis Engine for Generative Video Models (Part 10).

Goal:
Quickly diagnose visual generation failure modes and map them directly to their
underlying architectural, hyperparameter, or training root causes.

Diagnosed Failure Modes & Rules (from Wan Video DiT Specification):
1. Blurry output:
   - Root Causes: Undertrained model (<10k steps), too few sampling steps (<25 steps), or weak VAE codebook/latent reconstruction.
   - Prescriptions: Increase sampling steps to 30-50, train further, verify VAE roundtrip PSNR > 32dB.

2. Temporal Flickering:
   - Root Causes: VAE chunk borders (temporal overlap < 4 frames), or insufficient temporal training data / causal convolution issues.
   - Prescriptions: Enable chunk overlap blending in VAE decode (min 4-frame overlap window), increase video ratio in dataset curriculum.

3. Wrong motion for the prompt:
   - Root Causes: Weak or short captions, low CFG guidance scale (< 3.0), or poor text conditioning.
   - Prescriptions: Increase CFG scale to 4.5 - 7.0, run prompts through prompt rewriter (umT5 / Qwen), check cross-attention gradients.

4. Washed-out colors / oversaturated burns at high CFG:
   - Root Causes: CFG guidance scale too high (> 7.5), causing latent dynamic range explosion and clipping.
   - Prescriptions: Lower CFG guidance scale to 4.5 - 6.0, apply dynamic thresholding / latent rescale clamping.

5. Static / Still video (negligible motion):
   - Root Causes: Low dynamic degree, too many static image clips in training without motion filtering, or insufficient temporal RoPE frequency.
   - Prescriptions: Apply optical flow / motion filtering (minimum flow magnitude > 0.5) to training bucket, adjust timestep shift s=5.0.

6. Gray or brown amorphous blobs:
   - Root Causes: Inverted latent normalization (wrong mean/std values) or sign error in sampler Euler step: x = x + (sigma_next - sigma_i) * v.
   - Prescriptions: Verify Euler update sign (as sigma falls, sigma_next - sigma_i is negative, moving toward data), check latent mean=0, std=1.0.

7. Numerical instability / NaN output:
   - Root Causes: FlashAttention bf16 overflow, learning rate too high, or unclipped gradients.
   - Prescriptions: Apply grad clipping at 1.0, verify RMSNorm on Q and K, check adamw betas (0.9, 0.95).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class DiagnosticFinding:
    severity: str  # 'CRITICAL' | 'WARNING' | 'INFO' | 'OPTIMAL'
    issue: str
    likely_causes: List[str]
    prescriptions: List[str]
    affected_metrics: Dict[str, Any]


class VideoDiagnosisEngine:
    """Evaluates video metric profiles and sampling configs to provide instant diagnosis."""

    @staticmethod
    def diagnose(
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        symptoms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Diagnose model and generation behavior based on metrics, hyperparams, and observed symptoms."""
        metrics = metrics or {}
        config = config or {}
        symptoms = symptoms or []

        findings: List[DiagnosticFinding] = []

        # Extract parameters
        steps = config.get("num_inference_steps", config.get("steps", 30))
        cfg_scale = config.get("guidance_scale", config.get("cfg", 5.0))
        training_steps = config.get("training_steps", 20000)
        chunk_overlap = config.get("chunk_overlap_frames", 4)
        solver = config.get("solver", "euler")

        # Extract metric values
        img_qual = metrics.get("imaging_quality", 0.85)
        motion_smooth = metrics.get("motion_smoothness", 0.92)
        flicker_score = metrics.get("temporal_flickering", 0.95)
        dynamic_deg = metrics.get("dynamic_degree", 0.70)
        text_align = metrics.get("text_alignment", 0.75)
        vae_psnr = metrics.get("vae_psnr_db", 34.0)

        # 1. Blurry Output Diagnosis
        is_blurry = "blurry" in symptoms or img_qual < 0.65 or vae_psnr < 28.0
        if is_blurry:
            causes = []
            remedies = []
            if steps < 25:
                causes.append(f"Too few inference sampling steps (currently {steps} < 25).")
                remedies.append(f"Increase sampling steps to 30-50 for high-frequency convergence.")
            if training_steps < 5000:
                causes.append(f"Undertrained model checkpoint ({training_steps} steps). High frequencies converge late.")
                remedies.append("Train checkpoint further past 15k steps with shifted logit-normal schedule.")
            if vae_psnr < 30.0:
                causes.append(f"Sub-optimal VAE reconstruction fidelity ({vae_psnr} dB < 30 dB).")
                remedies.append("Verify VAE causal 3D decode and ensure correct channel scaling factors.")

            if not causes:
                causes.append("Insufficient high-frequency detail in latent representation.")
                remedies.append("Increase sampling steps to 40 and test a higher-order ODE solver (Heun or DPM-Solver++).")

            findings.append(DiagnosticFinding(
                severity="WARNING" if img_qual > 0.50 else "CRITICAL",
                issue="Blurry / Soft Video Output",
                likely_causes=causes,
                prescriptions=remedies,
                affected_metrics={"imaging_quality": img_qual, "steps": steps, "vae_psnr_db": vae_psnr},
            ))

        # 2. Temporal Flickering Diagnosis
        is_flickering = "flicker" in symptoms or flicker_score < 0.80 or chunk_overlap < 2
        if is_flickering:
            causes = []
            remedies = []
            if chunk_overlap < 4:
                causes.append(f"Insufficient VAE temporal chunk overlap ({chunk_overlap} frames < 4 frames minimum).")
                remedies.append("Set chunk_overlap_frames to 4 and apply cosine cross-fade blending between temporal chunks.")
            causes.append("Insufficient temporal dataset conditioning or lack of causal attention masking.")
            remedies.append("Check causal 3D attention masks to ensure future frames do not leak into past latent states.")

            findings.append(DiagnosticFinding(
                severity="WARNING",
                issue="Temporal Flickering / Chunk Boundary Glitches",
                likely_causes=causes,
                prescriptions=remedies,
                affected_metrics={"temporal_flickering": flicker_score, "chunk_overlap": chunk_overlap},
            ))

        # 3. Wrong Motion for Prompt / Semantic Misalignment
        is_wrong_motion = "wrong_motion" in symptoms or text_align < 0.60 or cfg_scale < 3.0
        if is_wrong_motion:
            causes = []
            remedies = []
            if cfg_scale < 3.5:
                causes.append(f"CFG guidance scale is too low ({cfg_scale} < 3.5), allowing model to drift towards generic priors.")
                remedies.append("Increase CFG guidance scale to 4.5 - 6.5.")
            causes.append("Sparse or uninformative prompt captions in training dataset.")
            remedies.append("Enrich captions using a multimodal LLM rewriter (umT5 / Qwen) before conditioning.")
            causes.append("Classifier-Free Guidance text drop rate improperly configured.")
            remedies.append("Ensure 10% text dropout during training so unconditional null embeddings train properly.")

            findings.append(DiagnosticFinding(
                severity="WARNING",
                issue="Wrong Motion / Semantic Prompt Deviation",
                likely_causes=causes,
                prescriptions=remedies,
                affected_metrics={"text_alignment": text_align, "cfg_scale": cfg_scale},
            ))

        # 4. Washed-Out Colors / High-CFG Burn
        is_burned_colors = "washed_out" in symptoms or cfg_scale > 7.5
        if is_burned_colors:
            causes = [
                f"Guidance scale ({cfg_scale}) is excessively high (>7.5), over-amplifying latent gradients.",
                "Latent vectors exceed dynamic range [-1.0, 1.0], saturating VAE color channels.",
            ]
            remedies = [
                "Lower CFG guidance scale to 4.0 - 6.0 (Wan default recommended: 5.0).",
                "Enable dynamic thresholding or latent percentile clamping in the sampler pipeline.",
            ]
            findings.append(DiagnosticFinding(
                severity="CRITICAL" if cfg_scale > 10.0 else "WARNING",
                issue="Washed-Out / Oversaturated Color Burn",
                likely_causes=causes,
                prescriptions=remedies,
                affected_metrics={"cfg_scale": cfg_scale},
            ))

        # 5. Static / Still Video (Lack of Motion)
        is_static = "static_video" in symptoms or dynamic_deg < 0.25
        if is_static:
            causes = [
                f"Dynamic degree is critically low ({dynamic_deg} < 0.25).",
                "Training dataset contains too many still images or static tripod clips without motion filtering.",
                "Timestep shift parameter s is too low, spending too little compute on coarse motion stages.",
            ]
            remedies = [
                "Filter training dataset using optical flow to enforce minimum average motion magnitude.",
                "Ensure timestep shift s=5.0 is applied during sampling and training.",
                "Promote curriculum stages 2 and 3 with multi-frame dynamic sequences.",
            ]
            findings.append(DiagnosticFinding(
                severity="WARNING",
                issue="Static Video (Lack of Kinematic Motion)",
                likely_causes=causes,
                prescriptions=remedies,
                affected_metrics={"dynamic_degree": dynamic_deg},
            ))

        # 6. Gray or Brown Blobs
        is_blobs = "gray_blobs" in symptoms
        if is_blobs:
            causes = [
                "Latent channel normalization mismatch (mean != 0 or std != 1.0 in VAE latent space).",
                "Sign error in Flow Matching Euler update step: x = x + (sigma_next - sigma_i) * v.",
            ]
            remedies = [
                "Verify Euler solver: since v = eps - x0, step moves towards x0 as sigma decreases (sigma_next - sigma_i is negative).",
                "Verify 16-channel latent scaling factors: latents = (raw - mean) / std.",
            ]
            findings.append(DiagnosticFinding(
                severity="CRITICAL",
                issue="Gray or Brown Amorphous Blobs",
                likely_causes=causes,
                prescriptions=remedies,
                affected_metrics={"latent_normalization": "CHECK_REQUIRED"},
            ))

        # If no issues detected: Healthy state
        if not findings:
            findings.append(DiagnosticFinding(
                severity="OPTIMAL",
                issue="All Generation Parameters & Metrics Nominal",
                likely_causes=["Balanced CFG guidance, sufficient sampling steps, and calibrated VAE."],
                prescriptions=["Model is performing optimally according to VBench criteria."],
                affected_metrics=metrics,
            ))

        return {
            "num_issues": sum(1 for f in findings if f.severity in ("CRITICAL", "WARNING")),
            "status": "PASS" if all(f.severity in ("INFO", "OPTIMAL") for f in findings) else "ACTION_REQUIRED",
            "findings": [
                {
                    "severity": f.severity,
                    "issue": f.issue,
                    "likely_causes": f.likely_causes,
                    "prescriptions": f.prescriptions,
                    "affected_metrics": f.affected_metrics,
                }
                for f in findings
            ],
        }
