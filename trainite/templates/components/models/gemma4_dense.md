# Gemma 4 Dense model

It is a decoder-only causal Transformer model based on Google DeepMind's Gemma 4 architecture: given a sequence of token IDs, it predicts the next token at every position.

## What goes in / out

- **Input**:
  - `input_ids` with shape `(batch, seq_len)`
  - `attention_mask` (optional) with shape `(batch, seq_len)`
- **Output**: logits with shape `(batch, seq_len, vocab_size)`

The trainer uses these logits with cross-entropy loss.

## What the model is made of

1. **Embedding**
   Converts token IDs into vectors, scaled by $\sqrt{\text{hidden\_size}}$.
2. **Alternating Attention Blocks**
   Alternates between local sliding-window attention (100% RoPE with $\theta=10{,}000$) and global full attention (25% RoPE with $\theta=1{,}000{,}000$).
3. **Q/K Normalization**
   Applies RMSNorm per head before RoPE to stabilize training dynamics and prevent softmax saturation.
4. **Sandwich Normalization**
   Applies RMSNorm before and after both Attention and MLP sub-layers.
5. **Gated GeLU MLP**
   GeLU feed-forward network with `tanh` approximation and no biases (`DenseMLP`).
6. **Learnable Layer Scale**
   Scales each block's output with a learnable scalar parameter.
7. **Final RMSNorm & Projection**
   Normalizes the final representations and projects to vocabulary logits (with optional logit soft-capping and weight tying).

## Config knobs

### `hidden_size`
Total embedding and hidden dimension size.

### `num_layers`
Total number of transformer layers to stack.

### `num_attention_heads`
Number of query attention heads. Must be divisible by `num_key_value_heads`.

### `num_key_value_heads`
Number of key/value heads for Grouped-Query Attention (GQA).

### `dense_intermediate_size`
Hidden dimension of the gated GeLU feed-forward sub-layer.

### `head_dim`
Dimension of each attention head.

### `sliding_window`
Context window length for local sliding-window attention layers (default: 512).

### `sliding_ratio`
Number of sliding-window layers before each global full-attention layer (default: 5, meaning 5 sliding then 1 global).

### `rope_theta`
Base frequency $\theta$ for local sliding-window layers (default: 10000.0).

### `global_rope_theta`
Base frequency $\theta$ for global full-attention layers (default: 1000000.0 for long-range scaling).

### `rotary_fraction`
Fraction of head dimensions rotated in local sliding layers (default: 1.0 for 100% rotation).

### `global_rotary_fraction`
Fraction of head dimensions rotated in global full layers (default: 0.25 for 25% partial rotation).

### `global_num_key_value_heads`
Number of key/value heads for global layers (default: 2).

### `global_head_dim`
Dimension of attention heads in global layers (default: 16).

### `global_key_equals_value`
Whether key projection is reused as value projection in global attention layers (default: true).

### `tie_word_embeddings`
Whether to tie the output projection weights to the input embedding table (default: true).

### `final_logit_softcap`
Optional $\tanh$-based soft-capping bound for logits to prevent overconfidence (default: null).

### `max_seq_len`
Maximum sequence length constraint for the model and trainer (default: 512).

## Minimal config example

```yaml
model:
  _target_: trainite.models.gemma4_moe.Gemma4DenseModel
  hidden_size: 64
  num_layers: 2
  num_attention_heads: 4
  num_key_value_heads: 2
  dense_intermediate_size: 128
  head_dim: 16
  sliding_window: 512
  sliding_ratio: 5
  rope_theta: 10000.0
  global_rope_theta: 1000000.0
  rotary_fraction: 1.0
  global_rotary_fraction: 0.25
  global_key_equals_value: true
  tie_word_embeddings: true
  final_logit_softcap: null
  max_seq_len: 512
```
