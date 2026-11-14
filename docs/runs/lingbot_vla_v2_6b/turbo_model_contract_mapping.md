# LingBot-VLA-V2 6B: HF fields to TurboModel contract mapping

Task: `lingbot_vla_v2_6b-t1` (T1) slice 2.

This document maps the checkpoint and official-repository fields to the
TorchTurbo `TurboModel` / `TurboVLMModel` contract. The machine-readable
version of the same mapping lives in
`src/torchturbo/models/lingbot_vla_v2_6b/contract.py`; the config parser and
fixtures are described in `capability_report.md`.

Pinned sources:

| Source | Revision |
| --- | --- |
| `robbyant/lingbot-vla-v2-6b` | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` |
| `Robbyant/lingbot-vla-v2` | `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` |
| `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |

The wrapper config is `LingbotVLAV2Config`. The HF `config.json` only carries
`vlm_family`; every architecture value is read from the official V2 YAML, the
Qwen3-VL backbone config, or the safetensors headers.

## 1. Config field mapping

`Default` means the official code supplies the value as a dataclass default
rather than in the V2 YAML. The parser records the same value, so the adapter
never re-implements it.

| Source | Field | Value | TurboModel consumer |
| --- | --- | --- | --- |
| HF `config.json` | `vlm_family` | `qwen3_vl` | config-class selection; rejects non-Qwen3-VL checkpoints |
| V2 YAML | `model.config_key` | `LingbotVLAV2Config` | config/AutoConfig registration |
| V2 YAML | `model.moe_implementation` | `fused` | token-MoE factory: `fused` on GPU, `eager` on CPU |
| V2 YAML | `model.adanorm_time` | `true` | action-expert AdaNorm time embedding |
| Default | `DataArguments.max_seq_len` | `2048` | packed sequence limit; no CP/SP split |
| Default | `DataArguments.img_size` | `256` | image processor and collator |
| V2 YAML | `data.cameras` | top, wrist-left, wrist-right | camera order in the VLA batch |
| V2 YAML | `data.joints` | 14+14+2+4+2+3+12 = 51 | state layout, padded to `max_state_dim=55` |
| V2 YAML | `train.use_moe` | `true` | replace every action-expert MLP with a token-MoE block |
| V2 YAML | `train.token_moe_layers` | `0..35` | MoE layer indices |
| V2 YAML | `train.token_num_experts` | `32` | router gate width |
| V2 YAML | `train.token_top_k` | `4` | router top-k |
| V2 YAML | `train.token_moe_intermediate_size` | `512` | routed expert FFN width |
| V2 YAML | `train.token_shared_intermediate_size` | `704` | shared expert FFN width |
| V2 YAML | `train.router_activation` | `sigmoid` | router score activation |
| V2 YAML | `train.routed_scaling_factor` | `4.0` | routed output scale |
| V2 YAML | `train.use_shared_expert_gate` | `false` | shared expert gating |
| V2 YAML | `train.bias_update_speed` | `0` | router correction-bias update rate |
| V2 YAML | `train.action_dim` | `55` (run-configurable) | flow-matching loss slice |
| V2 YAML | `train.max_action_dim` | `55` | `action_in_proj` / `action_out_proj` widths |
| V2 YAML | `train.max_state_dim` | `55` | `state_proj` width |
| Default | `MyTrainingArguments.chunk_size` | `50` | action horizon `n_action_steps` |
| Default | `MyTrainingArguments.loss_type` | `fm` | loss selection |
| V2 YAML | `train.tokenizer_max_length` | `72` | text truncation length (75 also fits 2048) |
| V2 YAML | `train.optimizer` | `muon` | engine optimizer (GPU reference; CPU smoke only) |
| V2 YAML | `train.use_compile` | `true` | compile policy (off on CPU) |
| V2 YAML | `train.enable_gradient_checkpointing` | `false` | activation checkpointing policy |
| V2 YAML | `train.enable_fp32` | `true` | mixed-precision policy (fp32 on CPU) |
| V2 YAML | `train.attention_implementation` | `flex_cached` | text attention backend (`eager`/SDPA on CPU) |
| Default | `MyTrainingArguments.vit_attn_implementation` | `flash_attention_2` | vision attention backend |
| V2 YAML | `train.ulysses_parallel_size` | `1` | SP/CP policy; must stay 1 |
| V2 YAML | `train.data_parallel_mode` | `fsdp2` | parallel-plan selection |
| Checkpoint | `weight_dtype` | `float32` | dtype check |
| Checkpoint | `total_size_bytes` | `25,503,630,044` | checkpoint size check |
| Backbone config | `text_config.hidden_size` | `2560` | backbone projections and parameter groups |
| Backbone config | `text_config.num_hidden_layers` | `36` | layer plan and FSDP wrapping |
| Backbone config | `text_config.num_attention_heads` | `32` | attention reshape |
| Backbone config | `text_config.num_key_value_heads` | `8` | GQA key/value width |
| Backbone config | `vision_config.depth` | `24` | vision tower plan |
| Backbone config | `vision_config.spatial_merge_size` | `2` | image token count and merger width |

`action_dim` and `max_action_dim` are different contracts:

- `max_action_dim=55` is fixed by the checkpoint projections. Every
  embodiment's actions/states are padded to 55 before the expert.
- `action_dim` is the loss width. The official YAML sets 55; an 8-dimension
  embodiment trains with `action_dim=8` while the projections stay 55-wide.

## 2. Module tree and parameter groups

`base_model_prefix` is `model`, but `model` is the `FlowMatchingV2` container,
not a transformer with a final `norm`. The adapter must override the
norm/head hooks instead of relying on the `TurboModel` defaults.

| Capability | Module path | ParameterGroup |
| --- | --- | --- |
| backbone text | `model.qwenvl_with_expert.qwenvl.model.language_model` | `language_model` |
| backbone final norm | `model.qwenvl_with_expert.qwenvl.model.language_model.norm` | `language_model` |
| vision tower | `model.qwenvl_with_expert.qwenvl.model.visual` | `vision_tower` |
| vision projector | `model.qwenvl_with_expert.qwenvl.model.visual.merger` | `multi_modal_projector` |
| deepstack mergers | `model.qwenvl_with_expert.qwenvl.model.visual.deepstack_merger_list` | `multi_modal_projector` |
| action expert | `model.qwenvl_with_expert.qwen_expert.model` | `language_model` |
| action-expert final norm | `model.qwenvl_with_expert.qwen_expert.model.norm` | `language_model` |
| state projection | `model.state_proj` | `language_model` |
| action input projection | `model.action_in_proj` | `language_model` |
| action output projection | `model.action_out_proj` | `language_model` |
| action/time MLP in | `model.action_time_mlp_in` | `language_model` |
| action/time MLP out | `model.action_time_mlp_out` | `language_model` |

The projector groups overlap the vision tower because `merger` and
`deepstack_merger_list` are nested under `visual`. `get_parameter_groups()` is
only consumed by `FreezePolicy`, and freezing is idempotent, so the overlap is
harmless.

## 3. Hook contract

Every public hook on `TurboModel` / `TurboVLMModel` is listed. The mapping is
checked by the test suite, so a new base hook forces an update here.

| Hook | Required LingBot behavior | Owner |
| --- | --- | --- |
| `get_norm` | return `...qwen_expert.model.norm` (the loss path ends at the expert norm), not the backbone norm | T2 |
| `get_lm_head` | return `None`; both lm_heads are deleted because `use_lm_head` is false | T2 |
| `get_output_embeddings` | return `None`, matching `get_lm_head` | T2 |
| `init_context` | force fp32 on wrapper and nested configs; eager/SDPA on CPU; keep the vision tower on its own attention implementation | T2 |
| `get_default_parallel_plan` | FSDP2 plan with EP=1 and SP=1; shard=16, bf16 and compile are GPU-only reference settings | T3 |
| `get_sp_attention_config` | no Ulysses/CP; expose the backbone text config only for a future SP path | T2 |
| `register_sp_attention_interface` | no-op at SP=1 | T3 |
| `on_parallelized` | no-op at EP=1; reserve the fused-expert `Shard(0)` interface (`experts.gate_proj/up_proj/down_proj`) | T3 |
| `on_materialized` | initialize rotary inverse frequencies and grid caches on the real device | T2 |
| `get_moe_runtime_inputs` | build the padding mask from `cu_seq_lens_q`; token-MoE routes every unpadded token, so no EP metadata | custom-moe |
| `get_sequence_parallel_input_slicing_spec` | declare the media-fusion boundary, but keep SP=1 so it is inactive in L0-L2 | T2/T3 |
| `on_prepare_inputs` | normalize the packed VLA batch and preserve media fields; reject `attention_mask` batches | T6 |
| `get_parameter_groups` | vision/vision-projector/language-model groups from section 2 | T2 |
| `get_optimizer_recipe` | muon matrix split for GPU reference; CPU only smoke-tests construction | T3/T9 |
| `get_weight_converter` | custom converter for fused experts and action head; skip optional align-head keys | T5 |
| `get_flops_class` | not required (no throughput target) | none |

A wrapper config note for T2: `LingbotVLAV2Config` has no `text_config` or
`vision_config` attributes. The nested Qwen3-VL configs are built inside
`FlowMatchingV2` (`QwenvlWithExpertV2Config` and `qwen_expert_config`). The
`TurboVLMModel` defaults for `nested_config_names`, `init_context` and
`get_sp_attention_config` therefore do not apply as-is; the adapter must
either expose properties for the nested configs or override those hooks.

## 4. Input contract

| Input | Shape / meaning | Contract |
| --- | --- | --- |
| `input_ids` / `lang_tokens` | packed Qwen tokenizer ids, `tokenizer_max_length=72` per sample | packed contract |
| `images` / `pixel_values` | 3 camera views at `img_size=256` | preserved through SP padding |
| `image_grid_thw` | Qwen3-VL processor grid, 3 entries per sample | media fusion |
| `state` | joint state, 51 valid dims padded to 55 | `state_proj` |
| `actions` / `actions_mask` | action chunk, `chunk_size=50`, padded to 55 | flow-matching loss |
| `cu_seq_lens_q/k` | packed boundaries | `on_prepare_inputs` requires them; `attention_mask` batches are rejected |

The first SFT adapter consumes current-camera images. `data.use_future_image`
and `train.align_params` are auxiliary distillation inputs; they are out of
scope for the first SFT version (see section 6).

## 5. Weight-key prefixes (T5 handoff)

The index has 1708 tensors, all F32, 25,503,630,044 bytes in 6 shards.

| Prefix | Content | Notes |
| --- | --- | --- |
| `model.qwenvl_with_expert.qwenvl.model.language_model.*` | 36 dense Qwen3-VL layers + embedding + final norm | plain HF Qwen3-VL keys |
| `model.qwenvl_with_expert.qwenvl.model.visual.*` | 24 vision blocks + `merger` + 3 `deepstack_merger_list` stacks | patch embed `[1024, 3, 2, 16, 16]` |
| `model.qwenvl_with_expert.qwen_expert.model.*` | 36 action-expert layers + final norm | fused MoE tensors below |
| `model.action_in_proj.*`, `model.action_out_proj.*` | `[768, 55]`, `[55, 768]` | flow-matching action head |
| `model.action_time_mlp_in.*`, `model.action_time_mlp_out.*` | `[768, 1536]`, `[768, 768]` | time conditioning |
| `model.state_proj.*` | `[768, 55]` | state conditioning |
| `model.depth_align_head.*`, `model.*_video_align_head.*`, `model.*task_proj.*` | optional distillation heads | skipped when `align_params` is empty |

Fused expert tensors per layer keep the checkpoint names:

| Key | Shape |
| --- | --- |
| `...layers.N.mlp.gate.weight` | `[32, 768]` |
| `...layers.N.mlp.experts.gate_proj` | `[32, 512, 768]` |
| `...layers.N.mlp.experts.up_proj` | `[32, 512, 768]` |
| `...layers.N.mlp.experts.down_proj` | `[32, 768, 512]` |
| `...layers.N.mlp.shared_expert.gate_proj.weight` | `[704, 768]` |
| `...layers.N.mlp.shared_expert.up_proj.weight` | `[704, 768]` |
| `...layers.N.mlp.shared_expert.down_proj.weight` | `[768, 704]` |
| `...layers.N.mlp.e_score_correction_bias` | `[32]` |

Because the fused expert convention matches the checkpoint names, the T5
converter can keep them verbatim and only needs to handle the expert
`Shard(0)` split for EP>1 (not exercised at EP=1).

## 6. Optional auxiliary heads and out-of-scope fields

`train.align_params` in `real_robot.yaml` enables depth and future-video
distillation. The checkpoint carries the corresponding heads and task-token
buffers, but they pull in external MoGe/DINO-Video assets and losses. The
first SFT version keeps `align_params` empty, drops these modules, and the
weight converter skips their keys. They remain documented so a later stage can
enable them without re-reading the checkpoint.

Also out of scope for L0-L2:

- Compile / flex-attention `flex_cached` kernels (CPU falls back to
  eager/SDPA).
- Fused MoE Triton kernels (CPU falls back to the reference eager implementation
  and only routing/structure is validated).
- SP/CP (size 1) and EP split (size 1).
