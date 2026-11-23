#!/usr/bin/env python3
"""Align the adapter text logits with the upstream Qwen3-VL reference.

The official lingbot-vla-v2 training stack builds its VLM backbone from the
upstream Qwen3-VL modeling code, so this tool uses
``Qwen3VLForConditionalGeneration`` from transformers as the golden reference
and compares it with the TorchTurbo adapter forward. Both models are randomly
initialized with the same seed and the adapter backbone is loaded from the
reference state dict, which isolates the adapter path from weight differences.

The tool only reads the config fixture: it never downloads the 28 GB
checkpoint. The vision tower is truncated to one layer because the reference
input is text-only; text layers are truncated with ``--layers``.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

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
BATCH_SIZE = 2
SEQ_LEN = 16


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare adapter text logits against Qwen3-VL reference."
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
        "--device", default="cpu", help="Torch device, e.g. cpu or cuda:0."
    )
    parser.add_argument(
        "--dtype", choices=sorted(DTYPES), default="fp32", help="Model dtype."
    )
    parser.add_argument(
        "--atol", type=float, default=2e-3, help="Max absolute logit difference."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the alignment check and return a process exit code."""
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

    torch.manual_seed(args.seed)
    reference = (
        Qwen3VLForConditionalGeneration(copy.deepcopy(config))
        .to(device=device, dtype=dtype)
        .eval()
    )
    torch.manual_seed(args.seed)
    adapter = (
        TurboLingbotVLAV2ForConditionalGeneration(copy.deepcopy(config))
        .to(device=device, dtype=dtype)
        .eval()
    )

    load_result = adapter.model.qwenvl_with_expert.qwenvl.load_state_dict(
        reference.state_dict(), strict=False
    )
    if load_result.missing_keys:
        raise SystemExit(
            f"FAIL: adapter is missing reference weights: {load_result.missing_keys}"
        )
    if load_result.unexpected_keys != ["lm_head.weight"]:
        raise SystemExit(
            "FAIL: adapter has unexpected weights: "
            f"{load_result.unexpected_keys}"
        )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        config.text_config.vocab_size,
        (BATCH_SIZE, SEQ_LEN),
        generator=generator,
    ).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        reference_logits = reference(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        adapter_logits = adapter(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

    if adapter_logits.shape != reference_logits.shape:
        raise SystemExit(
            f"FAIL: shape mismatch {adapter_logits.shape} vs "
            f"{reference_logits.shape}"
        )
    if not torch.isfinite(adapter_logits).all():
        raise SystemExit("FAIL: adapter logits contain non-finite values")

    max_abs_diff = (adapter_logits - reference_logits).abs().max().item()
    scale = reference_logits.abs().max().item()
    if scale == 0:
        raise SystemExit("FAIL: reference logits are all zero")

    print(
        f"text_layers={args.layers} vision_layers={VISION_LAYERS} "
        f"device={device} dtype={args.dtype} seed={args.seed}"
    )
    print(f"logits_shape={tuple(adapter_logits.shape)}")
    print(f"reference_abs_max={scale:.6e}")
    print(f"max_abs_diff={max_abs_diff:.6e} atol={args.atol:.1e}")
    if max_abs_diff > args.atol:
        print("FAIL: adapter logits differ from the Qwen3-VL reference")
        return 1

    print("PASS: adapter logits match the Qwen3-VL reference modeling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
