"""Targeted Improvements Suite (Part 12: Beating Wan).

Exports:
- 12.1 Long Video: build_block_causal_mask, KVCache, BlockCausalAttentionWrapper, RolloutTrajectoryTrainer
- 12.2 Cheaper Attention: WindowedSpacetimeAttention, HybridSpacetimeBlock, estimate_token_and_flop_savings
- 12.3 Few-Step & CFG: CFGDistillationLoss, FastSinglePassSampler
- 12.4 Text Understanding: StructuredPromptPlanner, get_prompt_following_test_cases
- 12.5 Reference Conditioning: InContextReferenceDiT
- 12.6 Representation Alignment: RepresentationAlignmentLoss, ShiftedTimestepSampler
- 12.7 Joint Audio-Video: JointAudioVideoAttention
- Ablation Benchmark: AblationRunner
"""

from vidgen.improvements.block_causal import (
    build_block_causal_mask,
    KVCache,
    BlockCausalAttentionWrapper,
    RolloutTrajectoryTrainer,
)
from vidgen.improvements.windowed_attention import (
    WindowedSpacetimeAttention,
    HybridSpacetimeBlock,
    estimate_token_and_flop_savings,
)
from vidgen.improvements.cfg_distill import (
    CFGDistillationLoss,
    FastSinglePassSampler,
)
from vidgen.improvements.prompt_planner import (
    StructuredPromptPlanner,
    get_prompt_following_test_cases,
)
from vidgen.improvements.in_context_ref import (
    InContextReferenceDiT,
)
from vidgen.improvements.rep_alignment import (
    RepresentationAlignmentLoss,
    ShiftedTimestepSampler,
)
from vidgen.improvements.joint_audio import (
    JointAudioVideoAttention,
)
from vidgen.improvements.ablation_runner import (
    AblationRunner,
)

__all__ = [
    "build_block_causal_mask",
    "KVCache",
    "BlockCausalAttentionWrapper",
    "RolloutTrajectoryTrainer",
    "WindowedSpacetimeAttention",
    "HybridSpacetimeBlock",
    "estimate_token_and_flop_savings",
    "CFGDistillationLoss",
    "FastSinglePassSampler",
    "StructuredPromptPlanner",
    "get_prompt_following_test_cases",
    "InContextReferenceDiT",
    "RepresentationAlignmentLoss",
    "ShiftedTimestepSampler",
    "JointAudioVideoAttention",
    "AblationRunner",
]
