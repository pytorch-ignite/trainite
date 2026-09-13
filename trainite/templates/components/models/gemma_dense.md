# Gemma Dense model

It is a decoder-only causal Transformer model based on Google DeepMind's Gemma 4 architecture: given a sequence of token IDs, it predicts the next token at every position.

## What goes in / out

- **Input**:
  - `input_ids` with shape `(batch, seq_len)`
  - `attention_mask` (optional) with shape `(batch, seq_len)`
- **Output**: logits with shape `(batch, seq_len, vocab_size)`

The trainer uses these logits with cross-entropy loss.

## What the model is made of

1. **Embedding**
   Converts token IDs into vectors, scaled by $\sqrt{\text{dim}}$.
2. **Alternating Attention Blocks**
   Alternates between local sliding-window attention (100% RoPE with $\theta=10{,}000$) and global full attention (25% RoPE with $\theta=1{,}000{,}000$).
3. **Q/K Normalization**
   Applies RMSNorm per head before RoPE to stabilize training dynamics and prevent softmax saturation.
4. **Sandwich Normalization**
   Applies RMSNorm before and after both Attention and MLP sub-layers.
5. **Gated GeLU MLP**
   GeLU feed-forward network with `tanh` approximation and no biases.
6. **Learnable Layer Scalar**
   Scales each block's output with a learnable scalar parameter.
7. **Final RMSNorm & Projection**
   Normalizes the final representations and projects to vocabulary logits (with optional logit soft-capping and weight tying).

## Config knobs

### `dim`
Total embedding and hidden dimension size.

### `num_layers`
Total number of `GemmaBlock` transformer layers to stack.

### `num_heads`
Number of query attention heads. Must be divisible by `num_kv_heads`.

### `num_kv_heads`
Number of key/value heads for Grouped-Query Attention (GQA).

### `head_dim`
Dimension of each attention head (e.g., 16 or 256).

### `intermediate_size`
Hidden dimension of the gated GeLU feed-forward sub-layer (typically $4\times$ to $8\times$ `dim`).

### `sliding_window`
Context window length for local sliding-window attention layers (default: 512).

### `local_rope_theta`
Base frequency $\theta$ for local sliding-window layers (default: 10000.0).

### `global_rope_theta`
Base frequency $\theta$ for global full-attention layers (default: 1000000.0 for long-range scaling).

### `local_rope_proportion`
Fraction of head dimensions rotated in local sliding layers (default: 1.0 for 100% rotation).

### `global_rope_proportion`
Fraction of head dimensions rotated in global full layers (default: 0.25 for 25% partial rotation).

### `sliding_ratio`
Number of sliding-window layers before each global full-attention layer (default: 5, meaning 5 sliding then 1 global).

### `rms_norm_eps`
Epsilon added to the variance in `RMSNorm` layers for numerical stability (default: 1e-6).

### `dropout`
Dropout rate applied in attention and feed-forward sub-layers (default: 0.0).

### `final_logit_softcap`
Optional $\tanh$-based soft-capping bound for logits to prevent overconfidence (e.g., 30.0, or `null` to disable).

### `tie_word_embeddings`
Whether to tie the output projection weights to the input embedding table to save memory (default: true).

### `max_seq_len`
Maximum sequence length constraint for the model and trainer (default: 512).

## Minimal config example

```yaml
model:
  _target_: trainite.models.gemma.GemmaDenseModel
  dim: 64
  num_layers: 2
  num_heads: 4
  num_kv_heads: 2
  head_dim: 16
  intermediate_size: 128
  sliding_window: 512
  local_rope_theta: 10000.0
  global_rope_theta: 1000000.0
  local_rope_proportion: 1.0
  global_rope_proportion: 0.25
  sliding_ratio: 5
  rms_norm_eps: 1e-6
  dropout: 0.0
  final_logit_softcap: null
  tie_word_embeddings: true
  max_seq_len: 512
```
