"""VidGen Post-Training and Extensions Suite (Part 11).

Exports:
- LoRA: LoRALinear, inject_lora_into_dit, merge_lora_weights, unmerge_lora_weights, get_lora_state_dict, load_lora_weights, count_parameters_summary
- Image-to-Video (I2V): WanImageToVideoDiT, convert_t2v_to_i2v, I2VConditioningCollator
- Dual Experts: Wan22DualExperts, create_dual_experts_from_base, compute_dual_expert_training_loss
- Distillation: RectifiedFlowReflow, FewStepFlowSolver
- Long Video: ChunkedOverlapGenerator
- Safety & Provenance: NSFWPromptFilter, VideoSafetyChecker, C2PAProvenanceWatermark
"""

from vidgen.extensions.lora import (
    LoRALinear,
    inject_lora_into_dit,
    merge_lora_weights,
    unmerge_lora_weights,
    get_lora_state_dict,
    load_lora_weights,
    count_parameters_summary,
)
from vidgen.extensions.i2v import (
    WanImageToVideoDiT,
    convert_t2v_to_i2v,
    I2VConditioningCollator,
)
from vidgen.extensions.dual_experts import (
    Wan22DualExperts,
    create_dual_experts_from_base,
    compute_dual_expert_training_loss,
)
from vidgen.extensions.distillation import (
    RectifiedFlowReflow,
    FewStepFlowSolver,
)
from vidgen.extensions.long_video import (
    ChunkedOverlapGenerator,
)
from vidgen.extensions.safety import (
    NSFWPromptFilter,
    VideoSafetyChecker,
    C2PAProvenanceWatermark,
)

__all__ = [
    "LoRALinear",
    "inject_lora_into_dit",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "get_lora_state_dict",
    "load_lora_weights",
    "count_parameters_summary",
    "WanImageToVideoDiT",
    "convert_t2v_to_i2v",
    "I2VConditioningCollator",
    "Wan22DualExperts",
    "create_dual_experts_from_base",
    "compute_dual_expert_training_loss",
    "RectifiedFlowReflow",
    "FewStepFlowSolver",
    "ChunkedOverlapGenerator",
    "NSFWPromptFilter",
    "VideoSafetyChecker",
    "C2PAProvenanceWatermark",
]
