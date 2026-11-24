"""Engine-side contract tests for the LingBot-VLA-V2 6B adapter.

The tests use tiny random-initialized configs: the CPU acceptance path only
validates the adapter contract, module tree and attention policy, never the
28 GB checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchturbo.engine.config import EngineConfig, ParameterGroup
from torchturbo.models import get_model
from torchturbo.models.lingbot_vla_v2_6b.configuration_lingbot_vla_v2_6b import (
    LingbotVLAV2Config,
    load_vla_config,
)
from torchturbo.models.lingbot_vla_v2_6b.model import (
    TurboLingbotVLAV2ForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parents[2]
HF_CONFIG = ROOT / "hf" / "lingbot-vla-v2-6b" / "config.json"


def tiny_config(**overrides) -> LingbotVLAV2Config:
    text_config = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000.0,
        "rope_scaling": {
            "rope_type": "default",
            "mrope_section": [4, 3, 3],
            "mrope_interleaved": True,
        },
        "dtype": "float32",
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 32,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
        "dtype": "float32",
    }
    return LingbotVLAV2Config(
        text_config=text_config,
        vision_config=vision_config,
        dtype="float32",
        **overrides,
    )


@pytest.fixture(scope="module")
def model() -> TurboLingbotVLAV2ForConditionalGeneration:
    return TurboLingbotVLAV2ForConditionalGeneration(tiny_config())


def test_model_is_registered():
    assert get_model("lingbot_vla_v2_6b") is (
        TurboLingbotVLAV2ForConditionalGeneration
    )


def test_checkpoint_config_declares_only_the_vlm_family():
    assert HF_CONFIG.is_file()
    # The backbone architecture is merged from the sibling backbone config.
    config = load_vla_config(
        HF_CONFIG, num_layers=2, num_vision_layers=1
    )
    assert config.vlm_family == "qwen3_vl"
    assert config.text_config.num_hidden_layers == 2
    assert config.vision_config.depth == 1
    assert config.text_config.hidden_size == 2560
    assert config.text_config.dtype == torch.float32
    assert config.text_config._attn_implementation == "eager"


def test_config_rejects_unknown_vlm_family():
    with pytest.raises(ValueError, match="vlm_family"):
        tiny_config(vlm_family="qwen2_vl")


def test_full_checkpoint_shapes_are_preserved(model):
    text_config = model.config.text_config
    assert model.base_model_prefix == "model"
    assert text_config.num_hidden_layers == 2
    assert model.config.max_action_dim == 55
    assert model.config.action_dim == 8
    assert len(model.config.moe_layers) == 36
    assert model.config.moe_layers[0] == 0
    assert model.config.moe_layers[-1] == 35
    assert model.config.cameras == ("top", "wrist_left", "wrist_right")


def test_checkpoint_module_tree(model):
    backbone = model.model.qwenvl_with_expert.qwenvl
    assert backbone.model.language_model is not None
    assert backbone.model.visual is not None
    assert backbone.model.visual.merger is not None
    # The checkpoint has no lm_head; the adapter must not recreate one.
    assert not hasattr(backbone, "lm_head")
    assert model.model.state_proj.in_features == 55
    assert model.model.action_in_proj.in_features == 55
    assert model.model.action_out_proj.out_features == 55
    assert model.model.action_time_mlp_in.in_features == 2 * 768


def test_head_and_norm_hooks(model):
    norm = model.get_norm()
    assert norm is model.model.qwenvl_with_expert.qwenvl.model.language_model.norm
    assert model.get_lm_head() is None
    assert model.get_output_embeddings() is None
    assert model.get_input_embeddings() is (
        model.model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens
    )


def test_init_context_forces_cpu_fp32_eager(model):
    config = tiny_config()
    config.dtype = torch.bfloat16
    config.text_config.dtype = torch.bfloat16
    with TurboLingbotVLAV2ForConditionalGeneration.init_context(
        config, EngineConfig(mixed_precision="fp32")
    ):
        assert config.dtype == torch.float32
        assert config.text_config.dtype == torch.float32
        assert config.text_config._attn_implementation == "eager"
        assert config.vision_config._attn_implementation == "eager"
    assert config.text_config.dtype == torch.bfloat16


def test_parameter_groups_are_disjoint_and_complete(model):
    groups = model.get_parameter_groups()
    total = {name for name, _ in model.named_parameters()}
    language = {name for name, _ in groups[ParameterGroup.LANGUAGE_MODEL]}
    vision = {name for name, _ in groups[ParameterGroup.VISION_TOWER]}
    projector = {
        name for name, _ in groups[ParameterGroup.MULTI_MODAL_PROJECTOR]
    }

    assert language
    assert vision
    assert projector
    assert language.isdisjoint(vision)
    assert language.isdisjoint(projector)
    assert vision.isdisjoint(projector)
    assert language | vision | projector == total
    assert any(name.startswith("model.state_proj") for name in language)
    assert any(
        name.startswith("model.qwenvl_with_expert.qwen_expert")
        for name in language
    ) is False


def test_sequence_parallel_is_refused_above_one(model):
    assert model.get_sp_attention_config() is model.config.text_config
    model.register_sp_attention_interface({"sp_size": 1})
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        model.register_sp_attention_interface({"sp_size": 2})

    spec = model.get_sequence_parallel_input_slicing_spec()
    assert spec.forward_name == "forward"
    assert spec.sliced_inputs[0][0] == "inputs_embeds"


def test_on_materialized_rebuilds_rotary_buffers(model):
    rotary = model.model.qwenvl_with_expert.qwenvl.model.language_model.rotary_emb
    before = rotary.inv_freq.clone()
    model.on_materialized()
    assert rotary.inv_freq.shape == before.shape
    assert torch.equal(rotary.inv_freq, before)


def test_forward_projects_tied_embedding_logits(model):
    input_ids = torch.randint(0, 128, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (2, 8, 128)
    assert torch.isfinite(output.logits).all()
    assert output.logits.abs().max() > 0

    hidden_states = (
        model.model.qwenvl_with_expert.qwenvl.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
    )
    expected = hidden_states @ model.get_input_embeddings().weight.t()
    assert torch.allclose(output.logits, expected)


def test_encode_state_projects_padded_joint_state(model):
    state = torch.randn(2, model.config.max_state_dim)
    features = model.encode_state(state)
    assert features.shape == (2, model.config.expert_hidden_size)
    assert torch.isfinite(features).all()

    features.sum().backward()
    grad = model.model.state_proj.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0
    model.model.state_proj.weight.grad = None

    with pytest.raises(ValueError, match="max_state_dim"):
        model.encode_state(torch.randn(2, model.config.joint_state_dim))


def test_prepare_inputs_preserves_vla_fields(model):
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "cu_seq_lens_q": torch.tensor([0, 4]),
        "cu_seq_lens_k": torch.tensor([0, 4]),
        "max_length_q": 4,
        "max_length_k": 4,
    }
    state = torch.randn(1, model.config.max_state_dim)
    actions = torch.randn(1, model.config.chunk_size, model.config.max_action_dim)
    source_inputs = {
        "state": state,
        "actions": actions,
        "pixel_values": torch.randn(2, 3, 16, 16),
        "image_grid_thw": torch.tensor([[1, 1, 2]]),
    }

    prepared = model.on_prepare_inputs(inputs, source_inputs=source_inputs)

    assert prepared is inputs
    assert prepared["state"] is state
    assert prepared["actions"] is actions
    assert prepared["pixel_values"] is source_inputs["pixel_values"]
