# LingBot-VLA-V2 6B capability report (T1)

Task: `lingbot_vla_v2_6b-t1` model parsing for the TorchTurbo adaptation.
Scope: read-only inspection of the checkpoint config and the official
`lingbot-vla-v2` modeling source. No full checkpoint weights were downloaded.

## Sources and provenance

| Source | Pinned revision | What was read |
| --- | --- | --- |
| `robbyant/lingbot-vla-v2-6b` | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` | `config.json`, `tokenizer_config.json`, `model.safetensors.index.json`, safetensors headers |
| `Robbyant/lingbot-vla-v2` | `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` | `configs/vla/real_robot/real_robot.yaml`, `lingbotvla/models/vla/lingbot_vla/*` |
| `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | backbone `config.json` (referenced by `model.tokenizer_path`) |

The checkpoint `config.json` is not a normal architecture config. Its whole
content is:

```json
{"vlm_family": "qwen3_vl"}
```

Architecture values therefore come from the official training YAML, the
Qwen3-VL backbone config, and the tensor shapes in the checkpoint. Tensor
shapes were read from the first 256 KiB of each safetensors shard with HTTP
Range requests; only metadata was transferred, never the 25.5 GB of weights.

### Provenance checks

The fixtures were re-verified against the pinned revisions on 2026-11-14:

- `config.json` is 31 bytes and equals `{"vlm_family": "qwen3_vl"}`.
- `tokenizer_config.json` and the Qwen3-VL backbone config match the live
  files field-for-field.
- `model.safetensors.index.json` lists 1708 tensors (all F32) with
  `total_size = 25503630044`; every shape in `checkpoint_summary.json`
  matches the shard headers exactly.
- The official repository `HEAD` is `be969b8...`; the training YAML at that
  commit matches `official_train_config.yaml`.

## Capability summary

### Backbone: Qwen3-VL 4B (dense)

| Field | Value | Evidence |
| --- | --- | --- |
| family | `qwen3_vl` | HF `config.json` |
| hidden size | 2560 | `embed_tokens [151936, 2560]` |
| layers | 36 | `language_model.layers.{0..35}`, backbone config |
| attention heads / KV heads | 32 / 8 | `q_proj [4096, 2560]`, `k_proj [1024, 2560]`, `q_norm [128]` |
| head dim | 128 | `q_norm` |
| FFN intermediate size | 9728 | `mlp.gate_proj [9728, 2560]` |
| vocab size | 151936 | `embed_tokens` |
| max positions | 262144 | backbone config |
| vision depth / hidden | 24 / 1024 | `visual.blocks.{0..23}`, `patch_embed [1024, 3, 2, 16, 16]` |
| vision heads | 16 | backbone `vision_config.num_heads` |
| patch / temporal patch | 16 / 2 | `patch_embed` shape |
| spatial merge | 2 | `merger.linear_fc1 [4096, 4096]` with 1024 hidden |
| deepstack indexes | 5, 11, 17 | backbone config; 3 merger stacks present |
| vision output width | 2560 | `merger.linear_fc2 [2560, 4096]` |
| LM head | absent | no `lm_head` tensor; `use_lm_head: false` |

The backbone is a dense Qwen3-VL text model. The "36 layer full token-MoE"
described in the requirement is the action expert, not the VLM backbone; both
stacks happen to have 36 layers.

### Action expert: 36-layer full token-MoE

| Field | Value | Evidence |
| --- | --- | --- |
| layers | 36 | `qwen_expert.model.layers.{0..35}` |
| hidden size | 768 | `q_proj [4096, 768]`, expert config default |
| attention heads / KV heads | 32 / 8 | `q_proj` / `k_proj` widths |
| head dim | 128 | config default |
| MoE layers | 0..35 (all layers) | `train.token_moe_layers` |
| experts / top-k | 32 / 4 | `experts.gate_proj [32, 512, 768]`, config |
| router activation | `sigmoid` | `train.router_activation` |
| routed scaling factor | 4.0 | `train.routed_scaling_factor` |
| routed intermediate size | 512 | `experts.gate_proj [32, 512, 768]` |
| shared intermediate size | 704 | `shared_expert.gate_proj [704, 768]` |
| shared expert gate | disabled | `train.use_shared_expert_gate: false` |
| MoE implementation | `fused` | `model.moe_implementation` |
| correction bias | present | `mlp.e_score_correction_bias` per layer |
| AdaNorm on time | enabled | `input_layernorm.gamma [768, 768]`, `beta` |
| router z-loss / sequence-wise loss | 1e-4 / 1e-3 | YAML coefficients |

`expert_intermediate_size: 2752` remains in the config, but every layer's MLP
is replaced by the token-MoE block, so the routed path uses the 512-wide
experts and the 704-wide shared expert instead.

### Flow-matching action head

| Field | Value | Evidence |
| --- | --- | --- |
| max_action_dim (projection width) | 55 | `action_in_proj [768, 55]`, `action_out_proj [55, 768]` |
| action_dim (loss width) | 55 in the V2 YAML; run-configurable | `losses[:, :, :self.config.action_dim]` |
| max_state_dim | 55 | `state_proj [768, 55]` |
| chunk size / n_action_steps | 50 / 50 | config `chunk_size` |
| flow-matching steps | 10 | hard-coded `LingbotVLAConfig.num_steps` |
| loss | flow matching MSE (`fm`) | `FlowMatchingV2.forward` |
| projection width | 768 | action head tensor shapes |
| action fp32 | disabled by default | `MyTrainingArguments.action_fp32` |

Training loss is a custom flow-matching MSE between the predicted velocity and
`noise - actions`, plus depth, future-video, sequence-wise balance and router
z-loss terms. It is not the standard causal-LM loss and has no LM head.

Two action-dimension fields have different meanings and must not be conflated:

- `max_action_dim` is fixed by the checkpoint projection layers (`55`). It
  determines the width of `action_in_proj`, `action_out_proj` and
  `state_proj`, so every embodiment's actions and states are zero-padded to
  55 before they reach the action expert.
- `action_dim` is a per-run knob used only to slice the action loss
  (`losses[:, :, :action_dim]`). The official V2 configs set it to 55, but an
  embodiment with 8 controlled dimensions trains with `action_dim: 8` while
  keeping `max_action_dim: 55`. The T3 verification command uses that
  `8 / 55` pair, and the parser accepts it because the projection-width
  cross-check only depends on `max_action_dim`.

### Multimodal inputs

| Field | Value | Evidence |
| --- | --- | --- |
| cameras | `camera_top`, `camera_wrist_left`, `camera_wrist_right` | `data.cameras` |
| image size | 256 | `MyDataArguments.img_size` default |
| image count | 3 | camera list |
| joint state dimension | 51 | `data.joints` (`14+14+2+4+2+3+12`) |
| padded state dimension | 55 | `max_state_dim` |
| image tokens | derived by the Qwen3-VL processor from `image_grid_thw` | `precompute_grid_thw: true` |

The model consumes precomputed `image_grid_thw`; the number of image tokens
per camera is a function of the image processor, not a static config field.

### Sequence and tokenizer

| Field | Value | Evidence |
| --- | --- | --- |
| training max sequence length | 2048 | `DataArguments.max_seq_len` default |
| tokenizer truncation length | 72 | `real_robot.yaml` (`train.tokenizer_max_length`) |
| tokenizer default in code | 48 | `MyTrainingArguments` |
| Qwen3-VL max positions | 262144 | backbone config |
| tokenizer `model_max_length` | 262144 | checkpoint `tokenizer_config.json` |
| Ulysses / context parallel | disabled (size 1) | `train.ulysses_parallel_size: 1` |

This is a short-sequence workload. Ulysses/CP long-sequence templates do not
apply.

### Training-side configuration

| Field | Value |
| --- | --- |
| optimizer | `muon` |
| compile | enabled |
| gradient checkpointing | disabled in `real_robot.yaml`, enabled in the guide |
| fp32 master/mixed precision | `enable_fp32: true` |
| attention implementation | `flex_cached` (vision `flash_attention_2`) |
| data parallel mode | `fsdp2` |
| expert parallel | not configured, defaults to 1 |
| dense weight dtype | float32 (all 1708 tensors in the index are F32) |
| checkpoint size | 25,503,630,044 bytes across 6 shards |

## Requirement-vs-checkpoint discrepancies

| Requirement note | Checkpoint / official source | Resolution |
| --- | --- | --- |
| `action_dim=8` | `max_action_dim=55` is checkpoint-fixed; `action_dim` is a loss-width run knob (V2 YAML: 55) | Keep `max_action_dim=55`; allow `action_dim=8` for an 8-dim embodiment (loss slice only, projections stay 55-wide). |
| `tokenizer_max_length=75` | 72 in the provided V2 configs; code default 48 | Parse the configured 72; 75 is a larger truncation bound that also fits in 2048 and can be set without architecture changes. |
| "36 layer full token-MoE" | Backbone dense 36 layers; action expert MoE 36 layers | The MoE stack is the action expert. |
| `seqLen=2048` | `DataArguments.max_seq_len` default 2048 | Confirmed. |
| `moe_implementation` field | `model.moe_implementation: fused` | Config attribute is `_moe_implementation`; CPU must downgrade to `eager`. |
| EP split interface | `LingbotVlaV2Policy.get_parallel_plan` shards fused expert tensors on dim 0 | Reserve the interface, keep EP=1 for L0-L2. |

## L0-L2 adaptation constraints (macOS CPU)

1. **No fused kernels.** The fused path depends on Triton group-GEMM kernels
   and is unavailable on CPU. Build the action expert with
   `moe_implementation="eager"` and validate routing/structure only.
2. **No flash-attention or flex attention.** Fall back to `eager`/SDPA
   attention; `flex_cached` and `flash_attention_2` are CUDA-oriented.
3. **fp32 only.** All checkpoint tensors are float32, and CPU training does
   not use bf16 autocast.
4. **Disable compile.** `use_compile`, fused MoE and compile-based grid
   caching are GPU-cluster settings.
5. **Two-layer subset.** Truncate the backbone and expert stacks for smoke
   tests; keep the 36/36 layer counts in the parsed capability.
6. **No Ulysses/CP.** Sequence length is 2048 and `ulysses_parallel_size` is 1.
7. **EP=1.** Multi-card FSDP2 shard=16, bf16 and muon are reference settings
   for later GPU stages only; the CPU run only smoke-tests optimizer
   construction.
8. **No full weights.** Numeric alignment uses random initialization; only
   config metadata is fetched.

## Contract mapping

`LingbotVLAV2Config` is selected by `model.config_key` in the V2 YAML and is
parsed as `capabilities.config_key`. The mapping from every parsed field and
checkpoint module path to the `TurboModel` / `TurboVLMModel` contract is in
`turbo_model_contract_mapping.md`; the machine-readable version is
`src/torchturbo/models/lingbot_vla_v2_6b/contract.py`.

## Fixtures

`tests/fixtures/lingbot_vla_v2_6b/` contains the evidence consumed by the
parser tests:

- `hf_config.json` - the 31-byte checkpoint config.
- `official_train_config.yaml` - `configs/vla/real_robot/real_robot.yaml`.
- `backbone_config.json` - Qwen3-VL-4B-Instruct config.
- `tokenizer_config.json` - checkpoint tokenizer config.
- `checkpoint_summary.json` - layer counts and 63 representative tensor
  shapes read from the safetensors headers, plus weight dtype and total size.

The capability values above are parsed by
`src/torchturbo/models/lingbot_vla_v2_6b/capabilities.py`.
Unknown or externally-sourced values are tracked by the unknowns report added
in the dedicated PR slice.
