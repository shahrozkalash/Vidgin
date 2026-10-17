"""VidGen Data Package: Curation, Manifests, Moving MNIST, Dataset, Buckets, Precomputation."""

from vidgen.data.manifest import (
    MANIFEST_COLUMNS,
    create_empty_manifest,
    read_manifest,
    write_manifest,
    resume_or_init_manifest,
)
from vidgen.data.moving_mnist import MovingMNISTGenerator
from vidgen.data.curation import (
    CurationConfig,
    compute_laplacian_sharpness,
    compute_motion_score,
    detect_black_borders,
    compute_dhash,
    hamming_distance,
    estimate_aesthetic_score,
    run_stage1_index,
    run_stage2_scenecut,
    run_stage3_technical_filters,
    run_stage4_quality_filters,
    run_stage5_deduplication,
    run_stage6_captioning,
    run_stage7_statistics,
    run_curation_pipeline_end_to_end,
)
from vidgen.data.dataset import CuratedVideoDataset
from vidgen.data.buckets import AspectRatioBucketSampler
from vidgen.data.precompute import precompute_latents_and_embeddings

__all__ = [
    "MANIFEST_COLUMNS",
    "create_empty_manifest",
    "read_manifest",
    "write_manifest",
    "resume_or_init_manifest",
    "MovingMNISTGenerator",
    "CurationConfig",
    "compute_laplacian_sharpness",
    "compute_motion_score",
    "detect_black_borders",
    "compute_dhash",
    "hamming_distance",
    "estimate_aesthetic_score",
    "run_stage1_index",
    "run_stage2_scenecut",
    "run_stage3_technical_filters",
    "run_stage4_quality_filters",
    "run_stage5_deduplication",
    "run_stage6_captioning",
    "run_stage7_statistics",
    "run_curation_pipeline_end_to_end",
    "CuratedVideoDataset",
    "AspectRatioBucketSampler",
    "precompute_latents_and_embeddings",
]
