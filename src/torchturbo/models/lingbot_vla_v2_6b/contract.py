"""LingBot-VLA-V2 field, module and hook mapping onto the TurboModel contract.

The values are parsed by :mod:`capabilities` from the checkpoint config, the
official V2 training YAML, the Qwen3-VL backbone config and the safetensors
headers. This module records, for every parsed value, which TorchTurbo
contract surface consumes it, and which module path the adapter must expose.

The mapping is deliberately declarative: the T2 adapter and the T5 weight
converter read these constants instead of re-deriving paths from the
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Checkpoint module tree. Every path below is a prefix in the safetensors
# index of revision 11c703b; the fused expert tensors are the only tensors
# that are not plain nn.Linear weights.
MODEL_ROOT = "model"
BACKBONE = "model.qwenvl_with_expert.qwenvl.model"
BACKBONE_TEXT = f"{BACKBONE}.language_model"
VISION_TOWER = f"{BACKBONE}.visual"
VISION_PROJECTOR = f"{VISION_TOWER}.merger"
VISION_DEEPSTACK = f"{VISION_TOWER}.deepstack_merger_list"
EXPERT = "model.qwenvl_with_expert.qwen_expert.model"
EXPERT_NORM = f"{EXPERT}.norm"
BACKBONE_NORM = f"{BACKBONE_TEXT}.norm"
ACTION_STATE_PROJ = "model.state_proj"
ACTION_IN_PROJ = "model.action_in_proj"
ACTION_OUT_PROJ = "model.action_out_proj"
ACTION_TIME_MLP_IN = "model.action_time_mlp_in"
ACTION_TIME_MLP_OUT = "model.action_time_mlp_out"

MODULE_PATHS: dict[str, str] = {
    "backbone_text": BACKBONE_TEXT,
    "backbone_norm": BACKBONE_NORM,
    "vision_tower": VISION_TOWER,
    "vision_projector": VISION_PROJECTOR,
    "vision_deepstack": VISION_DEEPSTACK,
    "action_expert": EXPERT,
    "action_expert_norm": EXPERT_NORM,
    "state_proj": ACTION_STATE_PROJ,
    "action_in_proj": ACTION_IN_PROJ,
    "action_out_proj": ACTION_OUT_PROJ,
    "action_time_mlp_in": ACTION_TIME_MLP_IN,
    "action_time_mlp_out": ACTION_TIME_MLP_OUT,
}

# Auxiliary distillation heads built only when model.align_params is non-empty.
# The real_robot checkpoint contains them, but the first SFT adaptation keeps
# align_params empty and the weight converter skips these keys.
OPTIONAL_MODULE_PATHS: dict[str, str] = {
    "depth_align_head": "model.depth_align_head",
    "future_depth_align_head": "model.future_depth_align_head",
    "current_video_align_head": "model.current_video_align_head",
    "future_video_align_head": "model.future_video_align_head",
    "current_shared_task_proj": "model.current_shared_task_proj",
    "future_shared_task_proj": "model.future_shared_task_proj",
}

# ParameterGroup values are only consumed by FreezePolicy. The projector is
# nested inside the vision tower in this checkpoint, so the two groups overlap
# by design and freezing is idempotent.
PARAMETER_GROUP_PATHS: dict[str, tuple[str, ...]] = {
    "vision_tower": (VISION_TOWER,),
    "multi_modal_projector": (VISION_PROJECTOR, VISION_DEEPSTACK),
    "language_model": (
        BACKBONE_TEXT,
        EXPERT,
        ACTION_STATE_PROJ,
        ACTION_IN_PROJ,
        ACTION_OUT_PROJ,
        ACTION_TIME_MLP_IN,
        ACTION_TIME_MLP_OUT,
    ),
}

# Media tensors that survive packed-input transforms. The Qwen3-VL processor
# emits image_grid_thw; the videos entries exist for the future-frame path.
MULTIMODAL_INPUT_KEYS: tuple[str, ...] = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
)

# VLA batches are not causal-LM batches: the engine's packed contract fields
# plus the VLA-specific state/action tensors.
PACKED_INPUT_KEYS: tuple[str, ...] = (
    "input_ids",
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "max_length_q",
    "max_length_k",
)
VLA_INPUT_KEYS: tuple[str, ...] = (
    "images",
    "image_grid_thw",
    "lang_tokens",
    "state",
    "actions",
    "actions_mask",
)


@dataclass(frozen=True)
class ConfigField:
    """One config value and the TurboModel surface that consumes it.

    Attributes:
        source: Fixture name (``hf_config``, ``official_config``,
            ``backbone_config``, ``checkpoint_summary``) or ``default`` for
            a value that the official code only supplies as a dataclass
            default.
        path: Dotted path inside the source document.
        capability: Dotted attribute path on
            :class:`capabilities.LingbotVLAV2Capabilities`.
        consumer: TorchTurbo contract surface that consumes the value.
        transform: Normalization needed before comparing source and parsed
            value. ``None`` compares the raw values (tuples compare as lists).
        default_value: Value used when ``source`` is ``default``.
    """

    source: str
    path: str
    capability: str
    consumer: str
    transform: str | None = None
    default_value: Any = None


CONFIG_FIELDS: tuple[ConfigField, ...] = (
    ConfigField("hf_config", "vlm_family", "vlm_family", "config class selection"),
    ConfigField("official_config", "model.config_key", "config_key", "AutoConfig registration"),
    ConfigField(
        "official_config",
        "model.moe_implementation",
        "action_expert.moe_implementation",
        "token-MoE block factory (fused on GPU, eager on CPU)",
    ),
    ConfigField(
        "official_config",
        "model.adanorm_time",
        "action_expert.adanorm_time",
        "action-expert AdaNorm time embedding",
    ),
    ConfigField("default", "DataArguments.max_seq_len", "sequence.max_seq_len", "packed sequence limit", default_value=2048),
    ConfigField("default", "DataArguments.img_size", "multimodal.image_size", "image processor and collator", default_value=256),
    ConfigField("official_config", "data.cameras", "multimodal.cameras", "camera order in the VLA batch", "tuple"),
    ConfigField(
        "official_config",
        "data.joints",
        "multimodal.joint_state_dim",
        "state_proj input width and normalized joint layout",
        "sum_joint_dims",
    ),
    ConfigField("official_config", "train.use_moe", "action_expert.use_moe", "action-expert MLP selection"),
    ConfigField(
        "official_config",
        "train.token_moe_layers",
        "action_expert.moe_layers",
        "MoE layer indices",
        "tuple",
    ),
    ConfigField(
        "official_config",
        "train.token_num_experts",
        "action_expert.num_experts",
        "router gate width",
    ),
    ConfigField("official_config", "train.token_top_k", "action_expert.top_k", "router top-k"),
    ConfigField(
        "official_config",
        "train.token_moe_intermediate_size",
        "action_expert.routed_intermediate_size",
        "routed expert FFN width",
    ),
    ConfigField(
        "official_config",
        "train.token_shared_intermediate_size",
        "action_expert.shared_intermediate_size",
        "shared expert FFN width",
    ),
    ConfigField(
        "official_config",
        "train.router_activation",
        "action_expert.router_activation",
        "router score activation",
    ),
    ConfigField(
        "official_config",
        "train.routed_scaling_factor",
        "action_expert.routed_scaling_factor",
        "routed output scale",
    ),
    ConfigField(
        "official_config",
        "train.use_shared_expert_gate",
        "action_expert.use_shared_expert_gate",
        "shared expert gating",
    ),
    ConfigField(
        "official_config",
        "train.bias_update_speed",
        "action_expert.bias_update_speed",
        "router correction bias update rate",
    ),
    ConfigField("official_config", "train.action_dim", "action_head.action_dim", "flow-matching loss slice"),
    ConfigField(
        "official_config",
        "train.max_action_dim",
        "action_head.max_action_dim",
        "action_in_proj/action_out_proj widths",
    ),
    ConfigField(
        "official_config",
        "train.max_state_dim",
        "action_head.max_state_dim",
        "state_proj input width",
    ),
    ConfigField("default", "MyTrainingArguments.chunk_size", "action_head.chunk_size", "action horizon", default_value=50),
    ConfigField("default", "MyTrainingArguments.loss_type", "action_head.loss_type", "loss selection", default_value="fm"),
    ConfigField(
        "official_config",
        "train.tokenizer_max_length",
        "sequence.tokenizer_max_length",
        "text truncation length",
    ),
    ConfigField("official_config", "train.optimizer", "training.optimizer", "engine optimizer (muon reference)"),
    ConfigField("official_config", "train.use_compile", "training.use_compile", "compile policy"),
    ConfigField(
        "official_config",
        "train.enable_gradient_checkpointing",
        "training.enable_gradient_checkpointing",
        "activation checkpointing policy",
    ),
    ConfigField("official_config", "train.enable_fp32", "training.enable_fp32", "mixed-precision policy"),
    ConfigField(
        "official_config",
        "train.attention_implementation",
        "training.attention_implementation",
        "text attention backend",
    ),
    ConfigField(
        "default",
        "MyTrainingArguments.vit_attn_implementation",
        "training.vit_attention_implementation",
        "vision attention backend",
        default_value="flash_attention_2",
    ),
    ConfigField(
        "official_config",
        "train.ulysses_parallel_size",
        "training.ulysses_parallel_size",
        "SP/CP policy (must stay 1)",
    ),
    ConfigField(
        "official_config",
        "train.data_parallel_mode",
        "training.data_parallel_mode",
        "parallel plan selection",
    ),
    ConfigField("checkpoint_summary", "weight_dtype", "weights.dtype", "checkpoint dtype check"),
    ConfigField(
        "checkpoint_summary",
        "total_size_bytes",
        "weights.total_size_bytes",
        "checkpoint size check",
    ),
    ConfigField(
        "backbone_config",
        "text_config.hidden_size",
        "backbone.hidden_size",
        "backbone projections and parameter groups",
    ),
    ConfigField(
        "backbone_config",
        "text_config.num_hidden_layers",
        "backbone.num_hidden_layers",
        "layer plan and FSDP wrapping",
    ),
    ConfigField(
        "backbone_config",
        "text_config.num_attention_heads",
        "backbone.num_attention_heads",
        "attention reshape",
    ),
    ConfigField(
        "backbone_config",
        "text_config.num_key_value_heads",
        "backbone.num_key_value_heads",
        "GQA key/value width",
    ),
    ConfigField("backbone_config", "vision_config.depth", "backbone.vision_num_layers", "vision tower plan"),
    ConfigField(
        "backbone_config",
        "vision_config.spatial_merge_size",
        "backbone.vision_spatial_merge_size",
        "image token count and merger width",
    ),
)


@dataclass(frozen=True)
class HookContract:
    """One TurboModel/TurboVLMModel hook and its LingBot-VLA-V2 behavior.

    Attributes:
        hook: Method name on :class:`torchturbo.models.base.TurboModel` or
            :class:`torchturbo.models.base.TurboVLMModel`.
        behavior: What the adapter must do for this checkpoint.
        owner: Task that implements the hook.
    """

    hook: str
    behavior: str
    owner: str


CONTRACT_HOOKS: tuple[HookContract, ...] = (
    HookContract(
        "get_norm",
        "Return the action-expert final norm (model.qwenvl_with_expert.qwen_expert.model.norm), "
        "not the backbone norm: the flow-matching loss path ends there.",
        "T2",
    ),
    HookContract(
        "get_lm_head",
        "Return None. Both lm_head modules are deleted because use_lm_head is false.",
        "T2",
    ),
    HookContract(
        "get_output_embeddings",
        "Return None, matching get_lm_head; the language embedding is tied to the "
        "input embedding and the expert embedding is deleted.",
        "T2",
    ),
    HookContract(
        "init_context",
        "Force fp32 on the wrapper and nested Qwen configs, select eager/SDPA attention "
        "on CPU and keep the vision tower on its own attention implementation.",
        "T2",
    ),
    HookContract(
        "get_default_parallel_plan",
        "Return an FSDP2 plan with EP=1 and SP=1; shard-16 and bf16 are GPU-only reference "
        "settings, not CPU acceptance criteria.",
        "T3",
    ),
    HookContract(
        "get_sp_attention_config",
        "Do not use Ulysses/CP: the workload is 2048 tokens and ulysses_parallel_size is 1. "
        "Expose the backbone text config only for a future SP path.",
        "T2",
    ),
    HookContract(
        "register_sp_attention_interface",
        "No-op at SP=1; the Qwen3-VL text attention must not be replaced by the Ulysses "
        "interface.",
        "T3",
    ),
    HookContract(
        "on_parallelized",
        "No-op at EP=1. Reserve the fused expert Shard(0) split interface "
        "(get_parallel_plan ep_plan on experts.gate_proj/up_proj/down_proj).",
        "T3",
    ),
    HookContract(
        "on_materialized",
        "Initialize rotary inverse frequencies and any grid cache on the real device; "
        "the CPU run keeps fp32 buffers.",
        "T2",
    ),
    HookContract(
        "get_moe_runtime_inputs",
        "Build the padding mask from cu_seq_lens_q; the token-MoE routes every unpadded "
        "token, so no EP all-to-all metadata is required.",
        "custom-moe",
    ),
    HookContract(
        "get_sequence_parallel_input_slicing_spec",
        "Declare the media-fusion boundary (after images are merged into text tokens), but "
        "keep SP=1 so the spec is inactive in L0-L2.",
        "T2/T3",
    ),
    HookContract(
        "on_prepare_inputs",
        "Normalize the packed VLA batch and preserve pixel_values/image_grid_thw; reject "
        "attention_mask batches per the packed contract.",
        "T6",
    ),
    HookContract(
        "get_parameter_groups",
        "Expose vision_tower (visual), multi_modal_projector (merger/deepstack) and "
        "language_model (backbone text + action expert + action head).",
        "T2",
    ),
    HookContract(
        "get_optimizer_recipe",
        "Provide a muon matrix split for GPU reference; the CPU run only smoke-tests "
        "construction.",
        "T3/T9",
    ),
    HookContract(
        "get_weight_converter",
        "Map HF keys including fused experts [num_experts, ...] and the action head; "
        "skip optional align-head keys when align_params is empty.",
        "T5",
    ),
    HookContract(
        "get_flops_class",
        "Not required: the adaptation has no throughput/latency target.",
        "none",
    ),
)

CONTRACT_CONSTANTS: dict[str, Any] = {
    "base_model_prefix": "model",
    "is_diffusion_model": False,
    "use_lm_head": False,
    "tie_word_embeddings": True,
    "num_action_expert_layers": 36,
    "num_backbone_layers": 36,
    "sp_size": 1,
    "ep_size": 1,
    "cpu_attn_implementation": "eager",
    "gpu_attn_implementation": "flex_cached",
    "cpu_moe_implementation": "eager",
    "gpu_moe_implementation": "fused",
}


def resolve_path(root: Any, path: str) -> Any:
    """Resolve a dotted path through mappings and attributes.

    Args:
        root: Parsed JSON/YAML document or capability object.
        path: Dotted path such as ``train.token_top_k``.

    Returns:
        The value stored at ``path``.

    Raises:
        KeyError: If a mapping key is missing.
        AttributeError: If an object attribute is missing.
    """
    value = root
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value[part]
        else:
            value = getattr(value, part)
    return value


__all__ = [
    "BACKBONE",
    "BACKBONE_NORM",
    "BACKBONE_TEXT",
    "CONFIG_FIELDS",
    "CONTRACT_CONSTANTS",
    "CONTRACT_HOOKS",
    "ConfigField",
    "EXPERT",
    "EXPERT_NORM",
    "HookContract",
    "MODEL_ROOT",
    "MODULE_PATHS",
    "MULTIMODAL_INPUT_KEYS",
    "OPTIONAL_MODULE_PATHS",
    "PACKED_INPUT_KEYS",
    "PARAMETER_GROUP_PATHS",
    "VISION_DEEPSTACK",
    "VISION_PROJECTOR",
    "VISION_TOWER",
    "VLA_INPUT_KEYS",
    "resolve_path",
]
