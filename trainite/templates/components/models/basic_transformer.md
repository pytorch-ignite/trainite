# Basic Transformer model

It is a decoder-only Transformer using absolute positional encoding: given a sequence of token IDs, it predicts the next token at every position.

## What goes in / out

- **Input**:
  - `input_ids` with shape `(batch, seq_len)`
  - `attention_mask` (optional) with shape `(batch, seq_len)`
- **Output**: logits with shape `(batch, seq_len, vocab_size)`

The trainer uses these logits with cross-entropy loss.

## What the model is made of

The model is small and standard:

1. **Token Embedding**
   Converts token IDs into vectors.
2. **Absolute Positional Encoding**
   Adds sinusoidal position vectors (`SinusoidalPositionalEncoding(hidden_size, max_seq_len)`). Position IDs are computed dynamically from `attention_mask` to support left-padded batches.
3. **Transformer blocks**
   Repeated attention + feedforward layers.
4. **Final projection**
   Maps hidden states back to vocabulary logits.

## Practical notes

### Padding
Padding ID is `0`. The model ignores padded positions in attention.

### Sequence length
`max_seq_len` represents the maximum supported sequence length. Inputs cannot exceed this limit.

### Hidden size and heads
`hidden_size` must be divisible by `num_heads`.

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
Maximum sequence length supported by the positional embeddings.

## Minimal config example

```yaml
model:
  _target_: models.basic_transformer.BasicTransformerModel
  hidden_size: 128
  num_layers: 4
  num_heads: 4
  feedforward_dim: 256
  dropout: 0.1
  max_seq_len: 64
```

## When to change this file

Edit `models/basic_transformer.py` if you want to:

- make the model wider or deeper
- swap attention or feedforward behavior
- change positional encoding strategy (e.g. sinusoidal vs learned)

## Good starting rule

If the dataset is tiny, start with a small model:

- `hidden_size=64`
- `num_layers=2`
- `num_heads=2`
- `feedforward_dim=128`

If training is stable and the model underfits, scale up from there.
