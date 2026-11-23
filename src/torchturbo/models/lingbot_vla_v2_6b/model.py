"""TorchTurbo adapter for LingBot-VLA-V2 6B (engine-side contract).

The adapter owns the TurboModel lifecycle: registration, module-tree layout,
parameter groups and the attention backend policy. The Qwen3-VL backbone is
the public ``Qwen3VLForConditionalGeneration`` stack; the flow-matching action
head and the token-MoE action expert are added by their own tasks, so this
module reserves the checkpoint module paths and the VLA projections without
duplicating those implementations.

The checkpoint module tree is preserved:

``model.qwenvl_with_expert.qwenvl.model.language_model.*``
    Qwen3-VL text backbone (36 dense layers).
``model.qwenvl_with_expert.qwenvl.model.visual.*``
    Qwen3-VL vision tower, merger and deepstack mergers.
``model.state_proj`` / ``model.action_*``
    Flow-matching action projections.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextModel,
)

from torchturbo.engine.config import EngineConfig, ParameterGroup
from torchturbo.runtime import get_device_runtime

from ..base import TurboVLMModel
from ..modules.attention.ulysses import SequenceParallelInputSlicingSpec
from ..registry import register_model
from .configuration_lingbot_vla_v2_6b import LingbotVLAV2Config

BACKBONE_PREFIX = "model.qwenvl_with_expert.qwenvl.model"
VISION_PREFIX = f"{BACKBONE_PREFIX}.visual"
TEXT_PREFIX = f"{BACKBONE_PREFIX}.language_model"

# Only one camera is consumed by the first SFT version. The key list still
# carries the full Qwen3-VL media contract so the engine preserves every media
# tensor through packed-input transforms.
MULTIMODAL_INPUT_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
)


class QwenVLWithExpert(nn.Module):
    """Checkpoint container for the VLM backbone and the action expert.

    The action expert stack is built by ``lingbot_vla_v2_6b-custom-moe``; the
    container is created here so the checkpoint path
    ``model.qwenvl_with_expert.qwenvl`` stays stable across tasks.
    """

    def __init__(self, config: LingbotVLAV2Config) -> None:
        super().__init__()
        self.qwenvl = Qwen3VLForConditionalGeneration(config)
        # The checkpoint has no language-model head; the language embedding is
        # tied to the input embedding and the expert has its own embedding.
        del self.qwenvl.lm_head


class LingbotVLAV2Model(nn.Module):
    """VLA module tree below the adapter root."""

    def __init__(self, config: LingbotVLAV2Config) -> None:
        super().__init__()
        self.qwenvl_with_expert = QwenVLWithExpert(config)
        self.state_proj = nn.Linear(config.max_state_dim, config.expert_hidden_size)
        self.action_in_proj = nn.Linear(
            config.max_action_dim, config.expert_hidden_size
        )
        self.action_out_proj = nn.Linear(
            config.expert_hidden_size, config.max_action_dim
        )
        self.action_time_mlp_in = nn.Linear(
            2 * config.expert_hidden_size, config.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            config.expert_hidden_size, config.expert_hidden_size
        )

    def named_vla_projections(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield the VLA projection parameters with their checkpoint names."""
        for name in (
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ):
            module = self.get_submodule(name)
            yield from module.named_parameters(prefix=f"model.{name}")


@register_model(
    "lingbot_vla_v2_6b",
    config_cls=LingbotVLAV2Config,
    auto_class=AutoModel,
)
class TurboLingbotVLAV2ForConditionalGeneration(TurboVLMModel, PreTrainedModel):
    """TorchTurbo wrapper for LingBot-VLA-V2 6B."""

    config_class = LingbotVLAV2Config
    base_model_prefix = "model"
    multimodal_input_keys = MULTIMODAL_INPUT_KEYS
    vision_tower_module_names = (VISION_PREFIX,)
    multi_modal_projector_module_names = (
        f"{VISION_PREFIX}.merger",
        f"{VISION_PREFIX}.deepstack_merger_list",
    )
    language_model_module_names = (TEXT_PREFIX,)

    def __init__(self, config: LingbotVLAV2Config) -> None:
        if not isinstance(config, LingbotVLAV2Config):
            config = LingbotVLAV2Config.from_dict(config.to_dict())
        super().__init__(config)
        self.model = LingbotVLAV2Model(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run the Qwen3-VL backbone and return tied-embedding logits.

        The checkpoint has no ``lm_head``; the language embedding matrix is
        tied, so logits are the backbone hidden states projected with the
        input embedding. Media tensors are fused by the Qwen3-VL model before
        the text stack, so they are forwarded unchanged.

        Packed flash-attention metadata (``cu_seq_lens_q/k`` and
        ``max_length_q/k``) is forwarded through ``**kwargs`` for the GPU
        training path. The CPU acceptance path uses a padded attention mask,
        which eager attention consumes directly.

        Args:
            input_ids: Token IDs shaped ``[batch, seq]`` or packed ``[1, seq]``.
            attention_mask: Padded attention mask, or packed boundaries via
                the flash-attention kwargs.
            position_ids: Optional mRoPE positions; derived by Qwen3-VL when
                omitted.
            pixel_values: Patch tensor for images.
            pixel_values_videos: Patch tensor for videos.
            image_grid_thw: Image grid metadata.
            video_grid_thw: Video grid metadata.
            **kwargs: Extra backbone arguments, including packed flash
                attention metadata.

        Returns:
            Logits and optional hidden states of the backbone.
        """
        backbone = self.model.qwenvl_with_expert.qwenvl.model
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = torch.matmul(
            hidden_states, self.get_input_embeddings().weight.t()
        )
        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
        )

    @classmethod
    @contextmanager
    def init_context(
        cls, config: LingbotVLAV2Config, engine_config: EngineConfig
    ) -> Iterator[None]:
        with super().init_context(config, engine_config):
            # CPU has no fused attention kernels and the requirement fixes the
            # CPU path at fp32. GPU keeps the engine default.
            if get_device_runtime().device.type == "cpu":
                config._attn_implementation = "eager"
                config.text_config._attn_implementation = "eager"
                config.vision_config._attn_implementation = "eager"
            yield

    def get_input_embeddings(self) -> nn.Module:
        return self.model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens

    def get_norm(self) -> nn.Module:
        # Until the action expert exists, the backbone norm is the end of the
        # loss path. The custom-moe task replaces this with the expert norm.
        return self.model.qwenvl_with_expert.qwenvl.model.language_model.norm

    def get_lm_head(self) -> None:
        return None

    def get_output_embeddings(self) -> None:
        return None

    def get_sp_attention_config(self) -> Any:
        return self.config.text_config

    def register_sp_attention_interface(self, sp_info: dict[str, Any]) -> None:
        if sp_info["sp_size"] > 1:
            raise NotImplementedError(
                "LingBot-VLA-V2 uses max_seq_len=2048 and "
                "ulysses_parallel_size=1; sequence parallelism is not part "
                "of this adaptation."
            )

    def get_sequence_parallel_input_slicing_spec(
        self,
    ) -> SequenceParallelInputSlicingSpec:
        # Media fusion happens inside the outer Qwen3-VL model, so the text
        # model entry is the first boundary that can be sequence-sliced.
        return SequenceParallelInputSlicingSpec(
            self.model.qwenvl_with_expert.qwenvl.model.language_model,
            Qwen3VLTextModel.forward,
        )

    def get_parameter_groups(
        self,
    ) -> dict[ParameterGroup, Iterable[tuple[str, nn.Parameter]]]:
        projector_prefixes = (
            f"{VISION_PREFIX}.merger",
            f"{VISION_PREFIX}.deepstack_merger_list",
        )
        language_prefixes = (
            TEXT_PREFIX,
            "model.state_proj",
            "model.action_in_proj",
            "model.action_out_proj",
            "model.action_time_mlp_in",
            "model.action_time_mlp_out",
        )

        def under(prefixes: tuple[str, ...]) -> Iterator[tuple[str, nn.Parameter]]:
            for name, parameter in self.named_parameters():
                if any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in prefixes
                ):
                    yield name, parameter

        def vision_tower() -> Iterator[tuple[str, nn.Parameter]]:
            for name, parameter in under((VISION_PREFIX,)):
                if any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in projector_prefixes
                ):
                    continue
                yield name, parameter

        return {
            ParameterGroup.ALL: self.named_parameters(),
            ParameterGroup.VISION_TOWER: vision_tower(),
            ParameterGroup.MULTI_MODAL_PROJECTOR: under(projector_prefixes),
            ParameterGroup.LANGUAGE_MODEL: under(language_prefixes),
        }

    def on_materialized(self) -> None:
        """Recompute non-persistent rotary buffers after meta-device loading."""
        rotary = (
            self.model.qwenvl_with_expert.qwenvl.model.language_model.rotary_emb
        )
        device = next(self.parameters()).device
        inv_freq, attention_scaling = rotary.rope_init_fn(rotary.config, device)
        rotary.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary.original_inv_freq = inv_freq
        rotary.attention_scaling = attention_scaling

    def _init_weights(self, module: nn.Module) -> None:
        # The nested Qwen3-VL model initializes itself; only the VLA
        # projections below the wrapper root need the checkpoint convention.
        projections = (
            self.model.state_proj,
            self.model.action_in_proj,
            self.model.action_out_proj,
            self.model.action_time_mlp_in,
            self.model.action_time_mlp_out,
        )
        if module in projections:
            nn.init.normal_(
                module.weight, std=self.config.text_config.initializer_range
            )
            nn.init.zeros_(module.bias)


__all__ = [
    "LingbotVLAV2Model",
    "QwenVLWithExpert",
    "TurboLingbotVLAV2ForConditionalGeneration",
]
