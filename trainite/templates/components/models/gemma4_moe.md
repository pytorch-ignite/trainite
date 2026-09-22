# Gemma 4 MoE model

A compact decoder-only Gemma 4 mixture-of-experts model for training from scratch.
It alternates sliding-window and global attention. Each block combines a dense
gated MLP with routed experts, while only the top experts run for each token.

## Helpful reference

This implementation was developed with
[rwightman/gemma4_pytorch_codex](https://github.com/rwightman/gemma4_pytorch_codex)
as an architecture reference.

## What goes in / out

- `input_ids`: `(batch, sequence)` token IDs
- `attention_mask`: optional `(batch, sequence)` padding mask
- output: `(batch, sequence, vocab_size)` logits

## Minimal config example

```yaml
model:
  _target_: models.gemma4_moe.Gemma4TextModel
  hidden_size: 64
  num_layers: 12
  num_attention_heads: 4
  num_key_value_heads: 2
  dense_intermediate_size: 128
  expert_dim: 64
  num_experts: 4
  top_k: 2
  head_dim: 16
  layer_pattern: sssssg
  pattern_repeats: 2
  sliding_window: 64
  global_num_key_value_heads: 2
  global_head_dim: 16
  global_key_equals_value: true
```

`layer_types` (one entry per layer) and `layer_pattern` + `pattern_repeats`
are mutually exclusive. `layer_pattern` is a compact string with
`s` = sliding and `g` = global (e.g. `sssssg` = 5x sliding + 1x global).
It is repeated to fill `num_layers`: `pattern_repeats` defaults to
`num_layers / len(pattern)` (must divide evenly), or set it explicitly
(e.g. `sssssg` repeated 5x for 30 layers).
Keep `top_k <= num_experts`.
