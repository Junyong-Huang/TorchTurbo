"""Capability parsing for LingBot-VLA-V2 6B.

The Hugging Face checkpoint ``robbyant/lingbot-vla-v2-6b`` does not ship a
complete architecture config: its ``config.json`` only declares
``{"vlm_family": "qwen3_vl"}``. Architecture values live in three places:

1. the official ``lingbot-vla-v2`` repository defaults and training YAML,
2. the Qwen3-VL backbone config referenced by ``model.tokenizer_path``,
3. the safetensors tensor shapes (layer counts, expert counts, action dims).

This module merges those sources into one immutable capability object and
cross-checks the config against the checkpoint shapes, so a wrong field name
or a stale YAML value fails loudly instead of silently building a mismatched
module tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

MODEL_KEY = "lingbot_vla_v2_6b"
HF_REPO_ID = "robbyant/lingbot-vla-v2-6b"
HF_REVISION = "11c703bf6a5c1f45b3b69168482da11fdbba53d7"
OFFICIAL_REPO_COMMIT = "be969b8fd117fb70550c5d4bf4bc328211b5b1b6"
BACKBONE_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
BACKBONE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

SUPPORTED_VLM_FAMILIES = ("qwen3_vl",)
SUPPORTED_MOE_IMPLEMENTATIONS = ("eager", "fused")
SUPPORTED_ROUTER_ACTIVATIONS = ("sigmoid", "softmax")

# The number of flow-matching integration steps is hard-coded in
# LingbotVLAConfig.__init__ (``self.num_steps = 10``); it has no YAML field.
FLOW_MATCHING_NUM_STEPS = 10

# Defaults reproduced from the official repository. A training YAML only
# carries the values a run overrides, so absent fields must fall back here.
# Sources: lingbotvla/utils/arguments.py (DataArguments) and
# tasks/vla/train_lingbotvla.py (MyTrainingArguments).
DATA_DEFAULTS: dict[str, Any] = {
    "max_seq_len": 2048,
    "img_size": 256,
}

TRAIN_DEFAULTS: dict[str, Any] = {
    "tokenizer_max_length": 48,
    "action_dim": 7,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "chunk_size": 50,
    "loss_type": "fm",
    "action_fp32": False,
    "use_moe": False,
    "token_moe_layers": None,
    "token_num_experts": 32,
    "token_top_k": 1,
    "token_moe_intermediate_size": 256,
    "token_shared_intermediate_size": 256,
    "router_activation": "softmax",
    "routed_scaling_factor": 1.0,
    "use_shared_expert_gate": True,
    "bias_update_speed": 0.001,
    "expert_hidden_size": 768,
    "expert_intermediate_size": 2752,
    "action_num_attention_heads": 32,
    "action_num_key_value_heads": 8,
    "action_head_dim": 128,
    "attention_implementation": "flex",
    "vit_attn_implementation": "flash_attention_2",
    "use_compile": False,
    "enable_gradient_checkpointing": False,
    "enable_fp32": False,
    "ulysses_parallel_size": 1,
    "data_parallel_mode": "fsdp2",
    "optimizer": "adamw",
}

_EMBED_TOKENS = "model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens.weight"
_BACKBONE_LAYER = "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.{}"
_VISION_PATCH = "model.qwenvl_with_expert.qwenvl.model.visual.patch_embed.proj.weight"
_VISION_BLOCK = "model.qwenvl_with_expert.qwenvl.model.visual.blocks.0.{}"
_VISION_MERGER = "model.qwenvl_with_expert.qwenvl.model.visual.merger.{}"
_EXPERT_LAYER = "model.qwenvl_with_expert.qwen_expert.model.layers.0.{}"


class CapabilityParseError(ValueError):
    """Raised when a config source is missing or contradicts the checkpoint."""


@dataclass(frozen=True)
class BackboneCapabilities:
    """Qwen3-VL vision-language backbone."""

    family: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    vision_hidden_size: int
    vision_num_layers: int
    vision_num_heads: int
    vision_patch_size: int
    vision_temporal_patch_size: int
    vision_spatial_merge_size: int
    vision_out_hidden_size: int
    deepstack_visual_indexes: tuple[int, ...]


@dataclass(frozen=True)
class ActionExpertCapabilities:
    """The 36-layer token-MoE action expert (not the dense VLM backbone)."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    use_moe: bool
    moe_layers: tuple[int, ...]
    num_experts: int
    top_k: int
    router_activation: str
    routed_scaling_factor: float
    moe_implementation: str
    routed_intermediate_size: int
    shared_intermediate_size: int
    use_shared_expert_gate: bool
    bias_update_speed: float
    adanorm_time: bool


@dataclass(frozen=True)
class ActionHeadCapabilities:
    """Flow-matching action head.

    ``max_action_dim`` is the checkpoint projection width (55) and fixes the
    shape of ``action_in_proj``/``action_out_proj``/``state_proj``.
    ``action_dim`` is the number of loss-bearing dimensions taken from the run
    config; it can be smaller than ``max_action_dim`` for a low-dimension
    embodiment (for example 8 actions padded to 55) and only slices the loss.
    """

    action_dim: int
    max_action_dim: int
    max_state_dim: int
    chunk_size: int
    num_steps: int
    loss_type: str
    action_fp32: bool
    proj_width: int


@dataclass(frozen=True)
class MultiModalCapabilities:
    """Camera views and padded robot state."""

    cameras: tuple[str, ...]
    image_size: int
    num_images: int
    joint_state_dim: int
    state_padding_dim: int


@dataclass(frozen=True)
class SequenceCapabilities:
    """Sequence-length knobs; the workload is short-sequence, not 128k."""

    max_seq_len: int
    tokenizer_max_length: int
    backbone_max_position_embeddings: int
    tokenizer_model_max_length: int


@dataclass(frozen=True)
class TrainingCapabilities:
    """Training-side knobs that the CPU adaptation must downgrade."""

    optimizer: str
    use_compile: bool
    enable_gradient_checkpointing: bool
    enable_fp32: bool
    attention_implementation: str
    vit_attention_implementation: str
    ulysses_parallel_size: int
    data_parallel_mode: str


@dataclass(frozen=True)
class WeightCapabilities:
    """Checkpoint facts read from the safetensors headers."""

    dtype: str
    total_size_bytes: int
    revision: str


@dataclass(frozen=True)
class LingbotVLAV2Capabilities:
    """Normalized capability report for ``lingbot_vla_v2_6b``."""

    model_key: str
    hf_repo_id: str
    hf_revision: str
    vlm_family: str
    backbone: BackboneCapabilities
    action_expert: ActionExpertCapabilities
    action_head: ActionHeadCapabilities
    multimodal: MultiModalCapabilities
    sequence: SequenceCapabilities
    training: TrainingCapabilities
    weights: WeightCapabilities


def load_capabilities(fixture_dir: str | Path) -> LingbotVLAV2Capabilities:
    """Load and parse the capability fixtures stored for this model.

    Args:
        fixture_dir: Directory holding ``hf_config.json``,
            ``official_train_config.yaml``, ``backbone_config.json``,
            ``tokenizer_config.json`` and ``checkpoint_summary.json``.

    Returns:
        The parsed capability object.
    """
    fixture_dir = Path(fixture_dir)

    def _json(name: str) -> Mapping[str, Any]:
        return json.loads((fixture_dir / name).read_text(encoding="utf-8"))

    official_config = yaml.safe_load(
        (fixture_dir / "official_train_config.yaml").read_text(encoding="utf-8")
    )
    return parse_capabilities(
        hf_config=_json("hf_config.json"),
        official_config=official_config,
        backbone_config=_json("backbone_config.json"),
        tokenizer_config=_json("tokenizer_config.json"),
        checkpoint_summary=_json("checkpoint_summary.json"),
    )


def parse_capabilities(
    *,
    hf_config: Mapping[str, Any],
    official_config: Mapping[str, Any],
    backbone_config: Mapping[str, Any],
    tokenizer_config: Mapping[str, Any],
    checkpoint_summary: Mapping[str, Any],
) -> LingbotVLAV2Capabilities:
    """Merge the four config sources and validate them against tensor shapes.

    Args:
        hf_config: Contents of the checkpoint ``config.json``.
        official_config: Parsed LingBot-VLA-V2 training YAML.
        backbone_config: Qwen3-VL backbone ``config.json``.
        tokenizer_config: Checkpoint ``tokenizer_config.json``.
        checkpoint_summary: Layer counts and tensor shapes read from the
            safetensors headers.

    Returns:
        The parsed capability object.

    Raises:
        CapabilityParseError: If a required field is missing or any config
            value contradicts the checkpoint tensor shapes.
    """
    vlm_family = _require(hf_config, "vlm_family", "HF config.json")
    if vlm_family not in SUPPORTED_VLM_FAMILIES:
        raise CapabilityParseError(
            f"Unsupported vlm_family {vlm_family!r}; expected one of "
            f"{SUPPORTED_VLM_FAMILIES}."
        )

    model_section = _section(official_config, "model")
    config_key = model_section.get("config_key")
    if config_key is not None and config_key != "LingbotVLAV2Config":
        raise CapabilityParseError(
            f"Unsupported model.config_key {config_key!r}; the V2 checkpoint "
            "requires LingbotVLAV2Config."
        )

    moe_implementation = _require(model_section, "moe_implementation", "official config model")
    if moe_implementation not in SUPPORTED_MOE_IMPLEMENTATIONS:
        raise CapabilityParseError(
            f"Unsupported model.moe_implementation {moe_implementation!r}; "
            f"expected one of {SUPPORTED_MOE_IMPLEMENTATIONS}."
        )

    train = _merged_section(official_config, "train", TRAIN_DEFAULTS)
    data = _merged_section(official_config, "data", DATA_DEFAULTS)
    tensors = _section(checkpoint_summary, "tensors")
    layer_counts = _section(checkpoint_summary, "layer_counts")

    backbone = _parse_backbone(backbone_config, tensors, layer_counts)
    action_expert = _parse_action_expert(
        train, tensors, layer_counts, moe_implementation,
        adanorm_time=bool(model_section.get("adanorm_time", False)),
    )
    action_head = _parse_action_head(train, tensors, action_expert.hidden_size)
    multimodal = _parse_multimodal(data, train)
    sequence = _parse_sequence(data, train, backbone, tokenizer_config)

    training = TrainingCapabilities(
        optimizer=str(train["optimizer"]),
        use_compile=bool(train["use_compile"]),
        enable_gradient_checkpointing=bool(train["enable_gradient_checkpointing"]),
        enable_fp32=bool(train["enable_fp32"]),
        attention_implementation=str(train["attention_implementation"]),
        vit_attention_implementation=str(train["vit_attn_implementation"]),
        ulysses_parallel_size=int(train["ulysses_parallel_size"]),
        data_parallel_mode=str(train["data_parallel_mode"]),
    )
    weights = WeightCapabilities(
        dtype=str(_require(checkpoint_summary, "weight_dtype", "checkpoint summary")),
        total_size_bytes=int(_require(checkpoint_summary, "total_size_bytes", "checkpoint summary")),
        revision=str(_require(checkpoint_summary, "revision", "checkpoint summary")),
    )

    return LingbotVLAV2Capabilities(
        model_key=MODEL_KEY,
        hf_repo_id=HF_REPO_ID,
        hf_revision=HF_REVISION,
        vlm_family=str(vlm_family),
        backbone=backbone,
        action_expert=action_expert,
        action_head=action_head,
        multimodal=multimodal,
        sequence=sequence,
        training=training,
        weights=weights,
    )


def _parse_backbone(
    backbone_config: Mapping[str, Any],
    tensors: Mapping[str, Sequence[int]],
    layer_counts: Mapping[str, int],
) -> BackboneCapabilities:
    text = _section(backbone_config, "text_config")
    vision = _section(backbone_config, "vision_config")

    hidden_size = int(_require(text, "hidden_size", "backbone text_config"))
    num_heads = int(_require(text, "num_attention_heads", "backbone text_config"))
    num_kv_heads = int(_require(text, "num_key_value_heads", "backbone text_config"))
    head_dim = int(_require(text, "head_dim", "backbone text_config"))
    vocab_size = int(_require(text, "vocab_size", "backbone text_config"))
    intermediate_size = int(_require(text, "intermediate_size", "backbone text_config"))
    num_layers = int(_require(text, "num_hidden_layers", "backbone text_config"))

    _expect(
        _shape(tensors, _EMBED_TOKENS),
        (vocab_size, hidden_size),
        "embed_tokens must be [vocab_size, hidden_size]",
    )
    _expect(
        _shape(tensors, _BACKBONE_LAYER.format("self_attn.q_proj.weight")),
        (num_heads * head_dim, hidden_size),
        "backbone q_proj must be [num_attention_heads * head_dim, hidden_size]",
    )
    _expect(
        _shape(tensors, _BACKBONE_LAYER.format("self_attn.k_proj.weight")),
        (num_kv_heads * head_dim, hidden_size),
        "backbone k_proj must be [num_key_value_heads * head_dim, hidden_size]",
    )
    _expect(
        _shape(tensors, _BACKBONE_LAYER.format("self_attn.q_norm.weight")),
        (head_dim,),
        "backbone q_norm must expose head_dim",
    )
    _expect(
        _shape(tensors, _BACKBONE_LAYER.format("mlp.gate_proj.weight")),
        (intermediate_size, hidden_size),
        "backbone mlp must be [intermediate_size, hidden_size]",
    )
    _expect(
        int(_require(layer_counts, "backbone", "checkpoint layer_counts")),
        num_layers,
        "checkpoint backbone layer count must match the backbone config",
    )

    vision_hidden = int(_require(vision, "hidden_size", "backbone vision_config"))
    vision_layers = int(_require(vision, "depth", "backbone vision_config"))
    vision_heads = int(_require(vision, "num_heads", "backbone vision_config"))
    patch_size = int(_require(vision, "patch_size", "backbone vision_config"))
    temporal_patch = int(_require(vision, "temporal_patch_size", "backbone vision_config"))
    merge_size = int(_require(vision, "spatial_merge_size", "backbone vision_config"))
    out_hidden = int(_require(vision, "out_hidden_size", "backbone vision_config"))
    deepstack = tuple(int(i) for i in _require(vision, "deepstack_visual_indexes", "backbone vision_config"))

    patch_shape = _shape(tensors, _VISION_PATCH)
    _expect(
        patch_shape,
        (vision_hidden, 3, temporal_patch, patch_size, patch_size),
        "vision patch_embed must be [hidden, 3, temporal_patch, patch, patch]",
    )
    merged_width = vision_hidden * merge_size * merge_size
    _expect(
        _shape(tensors, _VISION_MERGER.format("linear_fc1.weight")),
        (merged_width, merged_width),
        "vision merger fc1 must merge spatial_merge_size**2 patches",
    )
    _expect(
        _shape(tensors, _VISION_MERGER.format("linear_fc2.weight")),
        (out_hidden, merged_width),
        "vision merger fc2 must project to out_hidden_size",
    )
    _expect(
        int(_require(layer_counts, "vision", "checkpoint layer_counts")),
        vision_layers,
        "checkpoint vision layer count must match the backbone config",
    )
    _expect(
        int(_require(layer_counts, "deepstack_mergers", "checkpoint layer_counts")),
        len(deepstack),
        "checkpoint deepstack merger count must match the backbone config",
    )

    return BackboneCapabilities(
        family=str(_require(backbone_config, "model_type", "backbone config")),
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=int(_require(text, "max_position_embeddings", "backbone text_config")),
        vision_hidden_size=vision_hidden,
        vision_num_layers=vision_layers,
        vision_num_heads=vision_heads,
        vision_patch_size=patch_size,
        vision_temporal_patch_size=temporal_patch,
        vision_spatial_merge_size=merge_size,
        vision_out_hidden_size=out_hidden,
        deepstack_visual_indexes=deepstack,
    )


def _parse_action_expert(
    train: Mapping[str, Any],
    tensors: Mapping[str, Sequence[int]],
    layer_counts: Mapping[str, int],
    moe_implementation: str,
    *,
    adanorm_time: bool,
) -> ActionExpertCapabilities:
    hidden_size = int(train["expert_hidden_size"])
    intermediate_size = int(train["expert_intermediate_size"])
    num_heads = int(train["action_num_attention_heads"])
    num_kv_heads = int(train["action_num_key_value_heads"])
    head_dim = int(train["action_head_dim"])
    use_moe = bool(train["use_moe"])
    moe_layers = tuple(int(i) for i in (train["token_moe_layers"] or ()))
    num_layers = int(_require(layer_counts, "action_expert", "checkpoint layer_counts"))

    if train["router_activation"] not in SUPPORTED_ROUTER_ACTIVATIONS:
        raise CapabilityParseError(
            f"Unsupported router_activation {train['router_activation']!r}; "
            f"expected one of {SUPPORTED_ROUTER_ACTIVATIONS}."
        )
    if train["token_top_k"] < 1 or train["token_top_k"] > train["token_num_experts"]:
        raise CapabilityParseError(
            "token_top_k must be in [1, token_num_experts], got "
            f"{train['token_top_k']} with {train['token_num_experts']} experts."
        )
    if use_moe and not moe_layers:
        raise CapabilityParseError("use_moe=True requires a non-empty token_moe_layers list.")
    if use_moe and (max(moe_layers) >= num_layers or min(moe_layers) < 0):
        raise CapabilityParseError(
            f"token_moe_layers must index the {num_layers} action-expert layers, "
            f"got {moe_layers}."
        )

    _expect(
        _shape(tensors, _EXPERT_LAYER.format("self_attn.q_proj.weight")),
        (num_heads * head_dim, hidden_size),
        "action-expert q_proj must be [num_attention_heads * head_dim, hidden_size]",
    )
    _expect(
        _shape(tensors, _EXPERT_LAYER.format("self_attn.k_proj.weight")),
        (num_kv_heads * head_dim, hidden_size),
        "action-expert k_proj must be [num_key_value_heads * head_dim, hidden_size]",
    )

    num_experts = int(train["token_num_experts"])
    routed_intermediate = int(train["token_moe_intermediate_size"])
    shared_intermediate = int(train["token_shared_intermediate_size"])
    if use_moe:
        _expect(
            _shape(tensors, _EXPERT_LAYER.format("mlp.gate.weight")),
            (num_experts, hidden_size),
            "router gate must be [num_experts, hidden_size]",
        )
        _expect(
            _shape(tensors, _EXPERT_LAYER.format("mlp.experts.gate_proj")),
            (num_experts, routed_intermediate, hidden_size),
            "fused routed experts must be [num_experts, routed_intermediate, hidden_size]",
        )
        _expect(
            _shape(tensors, _EXPERT_LAYER.format("mlp.experts.down_proj")),
            (num_experts, hidden_size, routed_intermediate),
            "fused routed experts must be [num_experts, hidden_size, routed_intermediate]",
        )
        _expect(
            _shape(tensors, _EXPERT_LAYER.format("mlp.shared_expert.gate_proj.weight")),
            (shared_intermediate, hidden_size),
            "shared expert must be [shared_intermediate, hidden_size]",
        )

    return ActionExpertCapabilities(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        use_moe=use_moe,
        moe_layers=moe_layers,
        num_experts=num_experts,
        top_k=int(train["token_top_k"]),
        router_activation=str(train["router_activation"]),
        routed_scaling_factor=float(train["routed_scaling_factor"]),
        moe_implementation=moe_implementation,
        routed_intermediate_size=routed_intermediate,
        shared_intermediate_size=shared_intermediate,
        use_shared_expert_gate=bool(train["use_shared_expert_gate"]),
        bias_update_speed=float(train["bias_update_speed"]),
        adanorm_time=adanorm_time,
    )


def _parse_action_head(
    train: Mapping[str, Any],
    tensors: Mapping[str, Sequence[int]],
    proj_width: int,
) -> ActionHeadCapabilities:
    action_dim = int(train["action_dim"])
    max_action_dim = int(train["max_action_dim"])
    max_state_dim = int(train["max_state_dim"])
    if action_dim > max_action_dim:
        raise CapabilityParseError(
            f"action_dim ({action_dim}) cannot exceed max_action_dim ({max_action_dim})."
        )

    _expect(
        _shape(tensors, "model.action_in_proj.weight"),
        (proj_width, max_action_dim),
        "action_in_proj must be [proj_width, max_action_dim]",
    )
    _expect(
        _shape(tensors, "model.action_out_proj.weight"),
        (max_action_dim, proj_width),
        "action_out_proj must be [max_action_dim, proj_width]",
    )
    _expect(
        _shape(tensors, "model.action_time_mlp_in.weight"),
        (proj_width, proj_width * 2),
        "action_time_mlp_in must concatenate action and time embeddings",
    )
    _expect(
        _shape(tensors, "model.state_proj.weight"),
        (proj_width, max_state_dim),
        "state_proj must be [proj_width, max_state_dim]",
    )

    return ActionHeadCapabilities(
        action_dim=action_dim,
        max_action_dim=max_action_dim,
        max_state_dim=max_state_dim,
        chunk_size=int(train["chunk_size"]),
        num_steps=FLOW_MATCHING_NUM_STEPS,
        loss_type=str(train["loss_type"]),
        action_fp32=bool(train["action_fp32"]),
        proj_width=proj_width,
    )


def _parse_multimodal(
    data: Mapping[str, Any],
    train: Mapping[str, Any],
) -> MultiModalCapabilities:
    cameras = data.get("cameras")
    if not cameras:
        raise CapabilityParseError("official config data.cameras is required.")
    joints = data.get("joints")
    if not joints:
        raise CapabilityParseError("official config data.joints is required.")
    joint_state_dim = sum(int(dim) for entry in joints for dim in entry.values())

    return MultiModalCapabilities(
        cameras=tuple(str(camera) for camera in cameras),
        image_size=int(data["img_size"]),
        num_images=len(cameras),
        joint_state_dim=joint_state_dim,
        state_padding_dim=int(train["max_state_dim"]),
    )


def _parse_sequence(
    data: Mapping[str, Any],
    train: Mapping[str, Any],
    backbone: BackboneCapabilities,
    tokenizer_config: Mapping[str, Any],
) -> SequenceCapabilities:
    return SequenceCapabilities(
        max_seq_len=int(data["max_seq_len"]),
        tokenizer_max_length=int(train["tokenizer_max_length"]),
        backbone_max_position_embeddings=backbone.max_position_embeddings,
        tokenizer_model_max_length=int(
            _require(tokenizer_config, "model_max_length", "tokenizer config")
        ),
    )


def _section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise CapabilityParseError(f"Missing mapping section {key!r}.")
    return value


def _merged_section(
    mapping: Mapping[str, Any],
    key: str,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(defaults)
    override = mapping.get(key)
    if override is not None:
        if not isinstance(override, Mapping):
            raise CapabilityParseError(f"Section {key!r} must be a mapping.")
        merged.update(override)
    return merged


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise CapabilityParseError(f"Missing required field {key!r} in {context}.")
    return mapping[key]


def _shape(tensors: Mapping[str, Sequence[int]], name: str) -> tuple[int, ...]:
    if name not in tensors:
        raise CapabilityParseError(f"Checkpoint summary is missing tensor {name!r}.")
    return tuple(int(dim) for dim in tensors[name])


def _expect(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise CapabilityParseError(f"{message}: got {actual}, expected {expected}.")


__all__ = [
    "ActionExpertCapabilities",
    "ActionHeadCapabilities",
    "BackboneCapabilities",
    "CapabilityParseError",
    "LingbotVLAV2Capabilities",
    "MultiModalCapabilities",
    "SequenceCapabilities",
    "TrainingCapabilities",
    "WeightCapabilities",
    "load_capabilities",
    "parse_capabilities",
]
