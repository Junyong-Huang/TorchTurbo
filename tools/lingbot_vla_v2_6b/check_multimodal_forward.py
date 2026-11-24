#!/usr/bin/env python3
"""Check the LingBot-VLA-V2 multimodal forward on a truncated CPU model.

The check builds the adapter with randomly initialized weights (the 28 GB
checkpoint is never downloaded), feeds one 256x256 ``camera_wrist_right``
image through the Qwen3-VL vision tower, and verifies that:

* the image tokens are fused and only affect positions inside and after the
  image span (causal attention),
* the adapter forward returns finite logits,
* the joint state projects through ``state_proj`` with a nonzero gradient.

The text layers are truncated with ``--layers``; the vision tower is truncated
to one layer because the CPU check validates the input path, not the full
24-layer vision stack.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from torchturbo.models.lingbot_vla_v2_6b.configuration_lingbot_vla_v2_6b import (  # noqa: E402
    load_vla_config,
)
from torchturbo.models.lingbot_vla_v2_6b.model import (  # noqa: E402
    TurboLingbotVLAV2ForConditionalGeneration,
)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
VISION_LAYERS = 1
TEXT_BEFORE = 8
TEXT_AFTER = 4


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the truncated multimodal VLA forward on CPU."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the checkpoint config.json or its directory.",
    )
    parser.add_argument(
        "--layers", type=int, default=2, help="Truncated text layer count."
    )
    parser.add_argument(
        "--img-size", type=int, default=256, help="Square camera image side."
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device, e.g. cpu or cuda:0."
    )
    parser.add_argument(
        "--dtype", choices=sorted(DTYPES), default="fp32", help="Model dtype."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args(argv)


def _grid_and_shapes(
    image_size: int, patch_size: int, merge_size: int, temporal: int, channels: int
) -> tuple[tuple[int, int, int], int, int, int]:
    """Return grid, patch count, image token count and patch width."""
    if image_size % patch_size:
        raise SystemExit(
            f"FAIL: image size {image_size} is not divisible by patch size "
            f"{patch_size}"
        )
    grid_h = grid_w = image_size // patch_size
    if grid_h % merge_size or grid_w % merge_size:
        raise SystemExit(
            f"FAIL: grid {(grid_h, grid_w)} is not divisible by merge size "
            f"{merge_size}"
        )
    num_patches = grid_h * grid_w
    num_image_tokens = (grid_h // merge_size) * (grid_w // merge_size)
    patch_width = channels * temporal * patch_size * patch_size
    return (1, grid_h, grid_w), num_patches, num_image_tokens, patch_width


def main(argv: list[str] | None = None) -> int:
    """Run the multimodal forward check and return a process exit code."""
    args = _parse_args(argv)
    dtype = DTYPES[args.dtype]
    device = torch.device(args.device)

    config = load_vla_config(
        args.config,
        num_layers=args.layers,
        num_vision_layers=VISION_LAYERS,
        dtype=str(dtype).removeprefix("torch."),
        attn_implementation="eager",
    )
    if args.img_size != config.image_size:
        raise SystemExit(
            f"FAIL: --img-size {args.img_size} does not match config "
            f"image_size {config.image_size}"
        )

    vision = config.vision_config
    grid_thw, num_patches, num_image_tokens, patch_width = _grid_and_shapes(
        args.img_size,
        vision.patch_size,
        vision.spatial_merge_size,
        vision.temporal_patch_size,
        vision.in_channels,
    )

    torch.manual_seed(args.seed)
    model = (
        TurboLingbotVLAV2ForConditionalGeneration(copy.deepcopy(config))
        .to(device=device, dtype=dtype)
        .eval()
    )
    inner = model.model.qwenvl_with_expert.qwenvl.model

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    text_before = torch.randint(0, 1000, (1, TEXT_BEFORE), generator=generator)
    text_after = torch.randint(0, 1000, (1, TEXT_AFTER), generator=generator)
    image_tokens = torch.full((1, num_image_tokens), config.image_token_id)
    input_ids = torch.cat((text_before, image_tokens, text_after), dim=1).to(device)
    attention_mask = torch.ones_like(input_ids)
    grid = torch.tensor([grid_thw], dtype=torch.long, device=device)

    image_a = torch.randn(
        num_patches, patch_width, generator=generator, dtype=dtype
    ).to(device)
    image_b = torch.randn(
        num_patches, patch_width, generator=generator, dtype=dtype
    ).to(device)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=image_a,
            image_grid_thw=grid,
        ).logits
        hidden_a = inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=image_a,
            image_grid_thw=grid,
        ).last_hidden_state
        hidden_b = inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=image_b,
            image_grid_thw=grid,
        ).last_hidden_state
        vision_embeds = inner.visual(image_a, grid_thw=grid)[0]

    expected_shape = (1, TEXT_BEFORE + num_image_tokens + TEXT_AFTER, config.text_config.vocab_size)
    if tuple(logits.shape) != expected_shape:
        raise SystemExit(
            f"FAIL: logits shape {tuple(logits.shape)} != {expected_shape}"
        )
    if not torch.isfinite(logits).all():
        raise SystemExit("FAIL: logits contain non-finite values")
    if vision_embeds.shape[0] != num_image_tokens:
        raise SystemExit(
            f"FAIL: vision produced {vision_embeds.shape[0]} tokens, expected "
            f"{num_image_tokens}"
        )

    image_start = TEXT_BEFORE
    image_end = TEXT_BEFORE + num_image_tokens
    before_diff = (hidden_a[:, :image_start] - hidden_b[:, :image_start]).abs().max()
    image_diff = (
        hidden_a[:, image_start:image_end] - hidden_b[:, image_start:image_end]
    ).abs().max()
    if before_diff.item() > 1e-6:
        raise SystemExit(
            "FAIL: tokens before the image changed with a different image "
            f"(max diff {before_diff.item():.3e})"
        )
    if image_diff.item() <= 1e-6:
        raise SystemExit(
            "FAIL: image tokens did not change with a different image"
        )

    state = torch.randn(
        2, config.max_state_dim, generator=generator, dtype=dtype
    ).to(device)
    state_features = model.encode_state(state)
    if tuple(state_features.shape) != (2, config.expert_hidden_size):
        raise SystemExit(
            f"FAIL: state features shape {tuple(state_features.shape)} != "
            f"{(2, config.expert_hidden_size)}"
        )
    if not torch.isfinite(state_features).all():
        raise SystemExit("FAIL: state features contain non-finite values")
    state_features.sum().backward()
    state_grad = model.model.state_proj.weight.grad
    if state_grad is None or state_grad.abs().sum().item() == 0:
        raise SystemExit("FAIL: state_proj received no gradient")

    print(
        f"text_layers={args.layers} vision_layers={VISION_LAYERS} "
        f"image_size={args.img_size} grid={grid_thw} "
        f"patches={num_patches} image_tokens={num_image_tokens}"
    )
    print(f"device={device} dtype={args.dtype} seed={args.seed}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"tokens_before_image_max_diff={before_diff.item():.3e}")
    print(f"image_token_max_diff={image_diff.item():.3e}")
    print(
        f"state_features_shape={tuple(state_features.shape)} "
        f"state_grad_sum={state_grad.abs().sum().item():.3e}"
    )
    print("PASS: image fusion and joint-state projection are wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
