# Transformer model

It is a decoder-only Transformer: given a sequence of token IDs, it predicts the next token at every position.

## What goes in / out

- **Input**:
  - `input_ids` with shape `(batch, seq_len)`
  - `attention_mask` (optional) with shape `(batch, seq_len)`
- **Output**: logits with shape `(batch, seq_len, vocab_size)`

The trainer uses these logits with cross-entropy loss.

## What the model is made of

The model is small and standard:

1. **Embedding**
   Converts token IDs into vectors.
2. **Positional encoding**
   Adds token order information.
3. **Transformer blocks**
   Repeated attention + feedforward layers.
4. **Final projection**
   Maps hidden states back to vocabulary logits.

## Practical notes

### Padding
Padding ID is `0`. The model ignores padded positions in attention.

### Sequence length
`max_seq_len` represents the precomputed cache size for RoPE. Inputs longer than this will fall back to slower on-the-fly calculations.

### Hidden size and heads
`hidden_size` must be divisible by `num_heads`.

### Rotary positional encoding
`head_dim` (which is `hidden_size // num_heads`) must be even because RoPE rotates pairs of dimensions.

## Config knobs

### `hidden_size`
Size of the token embeddings and hidden states.

Bigger values usually give more capacity, but cost more memory and time.

### `num_layers`
How many Transformer blocks to stack.

More layers = more capacity, slower training.

### `num_heads`
How many attention heads to use.

Choose a value that divides `hidden_size` cleanly.

### `feedforward_dim`
Size of the feedforward layer inside each block.

A common choice is `2x` to `4x` of `hidden_size`.

### `dropout`
Dropout rate used in attention and feedforward layers.

### `max_seq_len`
Precomputed cache size for the rotary position embeddings.

## Minimal config example

```yaml
model:
  _target_: models.transformer.TransformerModel
  hidden_size: 128
  num_layers: 4
  num_heads: 4
  feedforward_dim: 256
  dropout: 0.1
  max_seq_len: 64
```

## When to change this file

Edit `models/transformer.py` if you want to:

- make the model wider or deeper
- swap attention or feedforward behavior
- change how padding is handled
- add caching, rotary embeddings, or other sequence features

## Good starting rule

If the dataset is tiny, start with a small model:

- `hidden_size=64`
- `num_layers=2`
- `num_heads=2`
- `feedforward_dim=128`

If training is stable and the model underfits, scale up from there.
