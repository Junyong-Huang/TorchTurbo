# Local LingBot-VLA-V2 6B config fixtures

`config.json` is the 31-byte checkpoint config from
`robbyant/lingbot-vla-v2-6b` (revision
`11c703bf6a5c1f45b3b69168482da11fdbba53d7`); it only declares `vlm_family`.

`backbone_config.json` is the Qwen3-VL-4B-Instruct backbone config pinned by
the official training stack. It is stored next to the checkpoint config
because the CPU verification tools must work without downloading the 28 GB
checkpoint.
