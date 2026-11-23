"""Configuration for the LingBot-VLA-V2 6B TorchTurbo adapter.

The public checkpoint ``robbyant/lingbot-vla-v2-6b`` ships a ``config.json``
that only declares ``vlm_family``. The architecture lives in the Qwen3-VL
backbone config referenced by the official training stack, so a local
``backbone_config.json`` is merged in when the checkpoint config is loaded.

The wrapper keeps the Qwen3-VL text and vision configs untouched; the VLA
fields below are consumed by the action head, the multimodal input adapter and
the sequence/parallel policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

MODEL_KEY = "lingbot_vla_v2_6b"
VLM_FAMILY = "qwen3_vl"
BACKBONE_CONFIG_FILENAME = "backbone_config.json"

# The checkpoint is 36 layers of action expert. Tests and the CPU smoke tools
# truncate this stack; the default mirrors the checkpoint.
DEFAULT_ACTION_EXPERT_LAYERS = 36
DEFAULT_MAX_ACTION_DIM = 55
DEFAULT_MAX_STATE_DIM = 55
DEFAULT_ACTION_DIM = 8
DEFAULT_CHUNK_SIZE = 50
DEFAULT_NUM_STEPS = 10

DEFAULT_CAMERAS = ("top", "wrist_left", "wrist_right")
DEFAULT_JOINT_STATE_DIM = 51
DEFAULT_IMAGE_SIZE = 256


class LingbotVLAV2Config(Qwen3VLConfig):
    """Qwen3-VL backbone plus the VLA-specific fields.

    Attributes:
        vlm_family: Backbone family; only ``qwen3_vl`` is supported.
        cameras: Camera order used by the VLA collator.
        image_size: Square camera image side length.
        joint_state_dim: Valid joint dimensions before padding.
        max_state_dim: Checkpoint ``state_proj`` input width.
        max_action_dim: Checkpoint action projection width.
        action_dim: Number of loss-bearing action dimensions.
        chunk_size: Action horizon predicted per sample.
        num_steps: Flow-matching integration steps.
        loss_type: Action loss selector; ``fm`` is flow matching.
        num_action_expert_layers: Depth of the token-MoE action expert.
        expert_hidden_size: Action expert hidden width.
        num_experts: Token-MoE routed expert count.
        top_k: Token-MoE routed experts per token.
        router_activation: Router score activation.
        routed_scaling_factor: Scale applied to the routed branch.
        token_moe_intermediate_size: Routed expert FFN width.
        token_shared_intermediate_size: Shared expert FFN width.
        moe_implementation: Checkpoint MoE implementation (fused); the CPU
            adapter falls back to the eager reference implementation.
        moe_layers: Action-expert layers that carry token-MoE.
    """

    model_type = "lingbot_vla_v2"

    def __init__(
        self,
        text_config: dict[str, Any] | Any | None = None,
        vision_config: dict[str, Any] | Any | None = None,
        vlm_family: str = VLM_FAMILY,
        cameras: tuple[str, ...] | list[str] = DEFAULT_CAMERAS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        joint_state_dim: int = DEFAULT_JOINT_STATE_DIM,
        max_state_dim: int = DEFAULT_MAX_STATE_DIM,
        max_action_dim: int = DEFAULT_MAX_ACTION_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        loss_type: str = "fm",
        action_fp32: bool = False,
        num_action_expert_layers: int = DEFAULT_ACTION_EXPERT_LAYERS,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        num_experts: int = 32,
        top_k: int = 4,
        router_activation: str = "sigmoid",
        routed_scaling_factor: float = 4.0,
        token_moe_intermediate_size: int = 512,
        token_shared_intermediate_size: int = 704,
        use_shared_expert_gate: bool = False,
        moe_implementation: str = "fused",
        moe_layers: tuple[int, ...] | list[int] | None = None,
        adanorm_time: bool = True,
        **kwargs: Any,
    ) -> None:
        if vlm_family != VLM_FAMILY:
            raise ValueError(
                f"Unsupported vlm_family {vlm_family!r}; "
                f"expected {VLM_FAMILY!r}."
            )

        self.vlm_family = vlm_family
        self.cameras = tuple(cameras)
        self.image_size = image_size
        self.joint_state_dim = joint_state_dim
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.loss_type = loss_type
        self.action_fp32 = action_fp32
        self.num_action_expert_layers = num_action_expert_layers
        self.expert_hidden_size = expert_hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_activation = router_activation
        self.routed_scaling_factor = routed_scaling_factor
        self.token_moe_intermediate_size = token_moe_intermediate_size
        self.token_shared_intermediate_size = token_shared_intermediate_size
        self.use_shared_expert_gate = use_shared_expert_gate
        self.moe_implementation = moe_implementation
        if moe_layers is None:
            self.moe_layers = tuple(range(num_action_expert_layers))
        else:
            self.moe_layers = tuple(moe_layers)
        self.adanorm_time = adanorm_time

        # The checkpoint has no lm_head; the language embedding is tied.
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: Any,
    ) -> "LingbotVLAV2Config":
        """Load the checkpoint config and merge the sibling backbone config.

        Args:
            pretrained_model_name_or_path: Directory containing
                ``config.json`` and ``backbone_config.json``, or a path to
                ``config.json``.
            **kwargs: Extra arguments forwarded to
                :meth:`transformers.PretrainedConfig.from_dict`.

        Returns:
            The merged wrapper config.

        Raises:
            FileNotFoundError: If the expected config files are missing.
        """
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        config_dict = _merge_backbone_config(
            pretrained_model_name_or_path, config_dict
        )
        return cls.from_dict(config_dict, **kwargs)


def _resolve_config_dir(config_path: str | Path) -> Path:
    path = Path(config_path)
    return path if path.is_dir() else path.parent


def _merge_backbone_config(
    config_path: str | Path, config_dict: dict[str, Any]
) -> dict[str, Any]:
    if "text_config" in config_dict and "vision_config" in config_dict:
        return config_dict

    config_dir = _resolve_config_dir(config_path)
    backbone_path = config_dir / BACKBONE_CONFIG_FILENAME
    if not backbone_path.is_file():
        raise FileNotFoundError(
            f"LingBot-VLA-V2 config at {config_path} declares only "
            f"vlm_family; expected the backbone config at {backbone_path}."
        )

    backbone_dict = json.loads(backbone_path.read_text(encoding="utf-8"))
    merged = dict(backbone_dict)
    merged.update(config_dict)
    return merged


def load_vla_config(
    config_path: str | Path,
    *,
    num_layers: int | None = None,
    num_vision_layers: int | None = None,
    dtype: str = "float32",
    attn_implementation: str = "eager",
) -> LingbotVLAV2Config:
    """Load a CPU-friendly, optionally truncated VLA config.

    Args:
        config_path: Path to the checkpoint ``config.json`` or its directory.
        num_layers: Optional backbone text layer override.
        num_vision_layers: Optional vision tower depth override.
        dtype: Model dtype recorded in the nested configs.
        attn_implementation: Attention implementation for text and vision.

    Returns:
        The merged wrapper config with the requested overrides.
    """
    path = Path(config_path)
    config_file = path if path.is_file() else path / "config.json"
    config_dict = json.loads(config_file.read_text(encoding="utf-8"))
    config_dict = _merge_backbone_config(config_file, config_dict)

    text_config = dict(config_dict["text_config"])
    vision_config = dict(config_dict["vision_config"])
    if num_layers is not None:
        text_config["num_hidden_layers"] = int(num_layers)
    if num_vision_layers is not None:
        vision_config["depth"] = int(num_vision_layers)
    text_config["dtype"] = dtype
    vision_config["dtype"] = dtype
    config_dict["text_config"] = text_config
    config_dict["vision_config"] = vision_config
    config_dict["dtype"] = dtype
    config_dict["attn_implementation"] = attn_implementation

    config = LingbotVLAV2Config.from_dict(config_dict)
    config.text_config._attn_implementation = attn_implementation
    config.vision_config._attn_implementation = attn_implementation
    return config


__all__ = [
    "BACKBONE_CONFIG_FILENAME",
    "DEFAULT_ACTION_DIM",
    "DEFAULT_ACTION_EXPERT_LAYERS",
    "DEFAULT_CAMERAS",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_JOINT_STATE_DIM",
    "DEFAULT_MAX_ACTION_DIM",
    "DEFAULT_MAX_STATE_DIM",
    "DEFAULT_NUM_STEPS",
    "LingbotVLAV2Config",
    "MODEL_KEY",
    "VLM_FAMILY",
    "load_vla_config",
]
