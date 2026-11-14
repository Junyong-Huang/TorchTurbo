"""Config parsing tests for the LingBot-VLA-V2 6B adaptation.

The tests use data recorded from the public checkpoint and the official
repository only: the 31-byte HF ``config.json``, the V2 training YAML, the
Qwen3-VL backbone config, and tensor shapes read from the safetensors headers.
No model weights are downloaded or instantiated.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
import yaml

from torchturbo.models.base import TurboModel, TurboVLMModel
from torchturbo.models.lingbot_vla_v2_6b.capabilities import (
    CapabilityParseError,
    load_capabilities,
    parse_capabilities,
)
from torchturbo.models.lingbot_vla_v2_6b.contract import (
    CONFIG_FIELDS,
    CONTRACT_CONSTANTS,
    CONTRACT_HOOKS,
    MODULE_PATHS,
    MULTIMODAL_INPUT_KEYS,
    PACKED_INPUT_KEYS,
    PARAMETER_GROUP_PATHS,
    VLA_INPUT_KEYS,
    resolve_path,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lingbot_vla_v2_6b"


def _read_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _sources() -> dict:
    return {
        "hf_config": _read_json("hf_config.json"),
        "official_config": yaml.safe_load(
            (FIXTURES / "official_train_config.yaml").read_text(encoding="utf-8")
        ),
        "backbone_config": _read_json("backbone_config.json"),
        "tokenizer_config": _read_json("tokenizer_config.json"),
        "checkpoint_summary": _read_json("checkpoint_summary.json"),
    }


@pytest.fixture(scope="module")
def capabilities():
    return load_capabilities(FIXTURES)


def test_hf_config_declares_only_the_vlm_family():
    # The checkpoint config.json is not a standard architecture config.
    assert _read_json("hf_config.json") == {"vlm_family": "qwen3_vl"}


def test_model_identity(capabilities):
    assert capabilities.model_key == "lingbot_vla_v2_6b"
    assert capabilities.hf_repo_id == "robbyant/lingbot-vla-v2-6b"
    assert capabilities.vlm_family == "qwen3_vl"
    assert capabilities.config_key == "LingbotVLAV2Config"
    assert capabilities.hf_revision == (
        "11c703bf6a5c1f45b3b69168482da11fdbba53d7"
    )


def test_backbone_capabilities(capabilities):
    backbone = capabilities.backbone
    assert backbone.family == "qwen3_vl"
    assert backbone.hidden_size == 2560
    assert backbone.num_hidden_layers == 36
    assert backbone.num_attention_heads == 32
    assert backbone.num_key_value_heads == 8
    assert backbone.head_dim == 128
    assert backbone.intermediate_size == 9728
    assert backbone.vocab_size == 151936
    assert backbone.max_position_embeddings == 262144


def test_vision_capabilities(capabilities):
    backbone = capabilities.backbone
    assert backbone.vision_hidden_size == 1024
    assert backbone.vision_num_layers == 24
    assert backbone.vision_num_heads == 16
    assert backbone.vision_patch_size == 16
    assert backbone.vision_temporal_patch_size == 2
    assert backbone.vision_spatial_merge_size == 2
    assert backbone.vision_out_hidden_size == 2560
    assert backbone.deepstack_visual_indexes == (5, 11, 17)


def test_action_expert_is_the_token_moe_stack(capabilities):
    expert = capabilities.action_expert
    assert expert.hidden_size == 768
    assert expert.num_hidden_layers == 36
    assert expert.num_attention_heads == 32
    assert expert.num_key_value_heads == 8
    assert expert.head_dim == 128
    assert expert.use_moe is True
    assert expert.moe_layers == tuple(range(36))
    assert expert.num_experts == 32
    assert expert.top_k == 4
    assert expert.router_activation == "sigmoid"
    assert expert.routed_scaling_factor == 4.0
    assert expert.moe_implementation == "fused"
    assert expert.routed_intermediate_size == 512
    assert expert.shared_intermediate_size == 704
    assert expert.use_shared_expert_gate is False
    assert expert.adanorm_time is True


def test_action_head_capabilities(capabilities):
    head = capabilities.action_head
    # max_action_dim is checkpoint-fixed by action_in_proj [768, 55]; the
    # official V2 YAML also sets action_dim (loss width) to 55.
    assert head.action_dim == 55
    assert head.max_action_dim == 55
    assert head.max_state_dim == 55
    assert head.chunk_size == 50
    assert head.num_steps == 10
    assert head.loss_type == "fm"
    assert head.action_fp32 is False
    assert head.proj_width == 768


def test_action_dim_may_be_narrower_than_max_action_dim():
    # An 8-dimension embodiment pads actions to 55 but slices the loss to 8.
    sources = _sources()
    sources["official_config"]["train"]["action_dim"] = 8

    head = parse_capabilities(**sources).action_head
    assert head.action_dim == 8
    assert head.max_action_dim == 55


def test_rejects_action_dim_wider_than_max_action_dim():
    sources = _sources()
    sources["official_config"]["train"]["action_dim"] = 56

    with pytest.raises(CapabilityParseError, match="cannot exceed max_action_dim"):
        parse_capabilities(**sources)


def test_multimodal_capabilities(capabilities):
    multimodal = capabilities.multimodal
    assert multimodal.cameras == (
        "camera_top",
        "camera_wrist_left",
        "camera_wrist_right",
    )
    assert multimodal.num_images == 3
    assert multimodal.image_size == 256
    assert multimodal.joint_state_dim == 51
    assert multimodal.state_padding_dim == 55


def test_sequence_capabilities(capabilities):
    sequence = capabilities.sequence
    assert sequence.max_seq_len == 2048
    # The provided V2 configs set 72. The tokenizer file's 262144 is the
    # model-card maximum, not the training truncation length.
    assert sequence.tokenizer_max_length == 72
    assert sequence.backbone_max_position_embeddings == 262144
    assert sequence.tokenizer_model_max_length == 262144


def test_training_and_weight_capabilities(capabilities):
    assert capabilities.training.optimizer == "muon"
    assert capabilities.training.use_compile is True
    assert capabilities.training.enable_fp32 is True
    assert capabilities.training.attention_implementation == "flex_cached"
    assert capabilities.training.ulysses_parallel_size == 1
    assert capabilities.weights.dtype == "float32"
    assert capabilities.weights.total_size_bytes == 25503630044


def test_rejects_unsupported_vlm_family():
    sources = _sources()
    sources["hf_config"]["vlm_family"] = "qwen2_vl"

    with pytest.raises(CapabilityParseError, match="vlm_family"):
        parse_capabilities(**sources)


def test_rejects_action_dim_that_contradicts_checkpoint():
    sources = _sources()
    sources["checkpoint_summary"]["tensors"]["model.action_in_proj.weight"] = [
        768,
        8,
    ]

    with pytest.raises(CapabilityParseError, match="action_in_proj"):
        parse_capabilities(**sources)


def test_rejects_bad_moe_top_k():
    sources = _sources()
    sources["official_config"]["train"]["token_top_k"] = 0

    with pytest.raises(CapabilityParseError, match="token_top_k"):
        parse_capabilities(**sources)


def test_rejects_backbone_layer_count_mismatch():
    sources = _sources()
    sources["checkpoint_summary"]["layer_counts"]["backbone"] = 2

    with pytest.raises(CapabilityParseError, match="backbone layer count"):
        parse_capabilities(**sources)


def test_rejects_missing_cameras():
    sources = _sources()
    sources["official_config"]["data"]["cameras"] = []

    with pytest.raises(CapabilityParseError, match="cameras"):
        parse_capabilities(**sources)


def test_parse_does_not_mutate_sources():
    sources = _sources()
    snapshot = copy.deepcopy(sources)
    parse_capabilities(**sources)
    assert sources == snapshot


def _normalize(value, transform):
    if transform == "tuple":
        return list(value)
    if transform == "sum_joint_dims":
        return sum(dim for entry in value for dim in entry.values())
    if isinstance(value, tuple):
        return list(value)
    return value


def test_contract_config_fields_match_capabilities(capabilities):
    sources = _sources()
    mismatches = {}
    for field in CONFIG_FIELDS:
        if field.source == "default":
            raw_value = field.default_value
        else:
            raw_value = resolve_path(sources[field.source], field.path)
        source_value = _normalize(raw_value, field.transform)
        capability_value = _normalize(resolve_path(capabilities, field.capability), None)
        if source_value != capability_value:
            mismatches[f"{field.source}:{field.path}"] = (
                source_value,
                capability_value,
            )
    assert not mismatches


def test_contract_hooks_cover_base_api():
    base_hooks = set()
    for cls in (TurboModel, TurboVLMModel):
        for name, member in inspect.getmembers(cls):
            if callable(member) and name.startswith(("get_", "on_", "register_", "init_")):
                base_hooks.add(name)

    mapped = {hook.hook for hook in CONTRACT_HOOKS}
    assert mapped == base_hooks


def test_contract_module_paths_exist_in_checkpoint():
    tensors = _read_json("checkpoint_summary.json")["tensors"]
    missing = {
        name: path
        for name, path in MODULE_PATHS.items()
        if not any(key.startswith(path + ".") for key in tensors)
    }
    assert not missing


def test_parameter_groups_cover_primary_modules():
    covered = [path for paths in PARAMETER_GROUP_PATHS.values() for path in paths]
    uncovered = [
        path
        for path in MODULE_PATHS.values()
        if not any(path == group or path.startswith(group + ".") for group in covered)
    ]
    assert not uncovered


def test_contract_constants_match_capabilities(capabilities):
    assert CONTRACT_CONSTANTS["num_backbone_layers"] == capabilities.backbone.num_hidden_layers
    assert (
        CONTRACT_CONSTANTS["num_action_expert_layers"]
        == capabilities.action_expert.num_hidden_layers
    )
    assert CONTRACT_CONSTANTS["sp_size"] == capabilities.training.ulysses_parallel_size == 1
    assert CONTRACT_CONSTANTS["ep_size"] == 1
    assert (
        CONTRACT_CONSTANTS["gpu_moe_implementation"]
        == capabilities.action_expert.moe_implementation
    )
    assert CONTRACT_CONSTANTS["use_lm_head"] is False
    assert CONTRACT_CONSTANTS["is_diffusion_model"] is False


def test_contract_vla_inputs_cover_packed_and_media_contract():
    # base.on_prepare_inputs requires cu_seq_lens_q/k and rejects attention_mask.
    assert {"cu_seq_lens_q", "cu_seq_lens_k"} <= set(PACKED_INPUT_KEYS)
    assert "attention_mask" not in PACKED_INPUT_KEYS
    assert {"pixel_values", "image_grid_thw"} <= set(MULTIMODAL_INPUT_KEYS)
    assert {"state", "actions"} <= set(VLA_INPUT_KEYS)
