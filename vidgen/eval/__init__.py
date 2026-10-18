"""VidGen Evaluation & VBench Suite (Part 10).

Exports:
- Validation Prompts: VALIDATION_PROMPTS_CATALOG, get_all_validation_prompts, get_prompts_by_category, get_sample_grid_prompts
- VBench & Generative Metrics: evaluate_vbench_dimensions, calculate_psnr, calculate_ssim, calculate_lpips,
  calculate_motion_smoothness, calculate_dynamic_degree, calculate_temporal_flickering,
  calculate_subject_consistency, calculate_background_consistency, calculate_imaging_quality,
  calculate_aesthetic_score, calculate_text_video_alignment, calculate_fvd, evaluate_vae_reconstruction
- Memorization Check: VideoMemorizationDetector, extract_video_embedding
- Quick Diagnosis: VideoDiagnosisEngine, DiagnosticFinding
- Human A/B Testing: HumanABEvaluationSuite, CheckpointProfile
"""

from vidgen.eval.validation_prompts import (
    ValidationPrompt,
    VALIDATION_PROMPTS_CATALOG,
    get_all_validation_prompts,
    get_prompts_by_category,
    get_sample_grid_prompts,
    export_prompts_as_dict,
)
from vidgen.eval.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    calculate_motion_smoothness,
    calculate_dynamic_degree,
    calculate_temporal_flickering,
    calculate_subject_consistency,
    calculate_background_consistency,
    calculate_imaging_quality,
    calculate_aesthetic_score,
    calculate_text_video_alignment,
    evaluate_vbench_dimensions,
    calculate_fvd,
    evaluate_vae_reconstruction,
)
from vidgen.eval.memorization import (
    VideoMemorizationDetector,
    extract_video_embedding,
)
from vidgen.eval.diagnostics import (
    VideoDiagnosisEngine,
    DiagnosticFinding,
)
from vidgen.eval.human_ab import (
    HumanABEvaluationSuite,
    CheckpointProfile,
)

__all__ = [
    "ValidationPrompt",
    "VALIDATION_PROMPTS_CATALOG",
    "get_all_validation_prompts",
    "get_prompts_by_category",
    "get_sample_grid_prompts",
    "export_prompts_as_dict",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_lpips",
    "calculate_motion_smoothness",
    "calculate_dynamic_degree",
    "calculate_temporal_flickering",
    "calculate_subject_consistency",
    "calculate_background_consistency",
    "calculate_imaging_quality",
    "calculate_aesthetic_score",
    "calculate_text_video_alignment",
    "evaluate_vbench_dimensions",
    "calculate_fvd",
    "evaluate_vae_reconstruction",
    "VideoMemorizationDetector",
    "extract_video_embedding",
    "VideoDiagnosisEngine",
    "DiagnosticFinding",
    "HumanABEvaluationSuite",
    "CheckpointProfile",
]
