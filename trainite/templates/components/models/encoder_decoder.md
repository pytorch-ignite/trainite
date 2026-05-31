# Encoder-Decoder Transformer model

This is the model that learns the string-reverse task using sequence-to-sequence modeling.

It consists of an explicit Encoder and Decoder:
- **Encoder**: Processes the source sequence (the original string).
- **Decoder**: Autoregressively generates the target sequence (the reversed string) by performing causal self-attention over generated tokens and cross-attention over the encoder's hidden representations.

## What goes in / out

- **Input**:
  - `encoder_input_ids` with shape `(batch, src_seq_len)`
  - `decoder_input_ids` with shape `(batch, tgt_seq_len)`
- **Output**: logits with shape `(batch, tgt_seq_len, vocab_size)`

The trainer uses these logits with cross-entropy loss against target labels.

## What the model is made of

The model is small and modular:

1. **Embedding Layer**
   Shared token embeddings between encoder and decoder to map token IDs to hidden states.
2. **Positional Encoding**
   Sinusoidal positional encoding to add token order information.
3. **Encoder Stack**
   Stack of Encoder blocks executing self-attention and feedforward layers to encode the source sequence.
4. **Decoder Stack**
   Stack of Decoder blocks executing causal self-attention, cross-attention over the encoder's memory, and feedforward layers.
5. **Final Projection**
   Linear layer mapping decoder outputs back to vocabulary logits.

## Practical notes

### Padding
Padding ID is `0`. The model ignores padded positions in attention.

### Sequence length
`encoder_max_seq_len` and `decoder_max_seq_len` should be at least as long as your longest source and target input sequences, respectively.

### Hidden size and heads
`hidden_size` must be divisible by `num_heads`.

### Positional encoding
`hidden_size` must be even because the positional encoding uses sine and cosine pairs.

## Config knobs

### `hidden_size`
Size of the token embeddings and hidden states.

### `num_encoder_layers`
How many Transformer blocks to stack in the encoder.

### `num_decoder_layers`
How many Transformer blocks to stack in the decoder.

### `num_heads`
How many attention heads to use. Choose a value that divides `hidden_size` cleanly.

### `feedforward_dim`
Size of the feedforward layer inside each block. Usually `2x` to `4x` of `hidden_size`.

### `dropout`
Dropout rate used in attention, feedforward layers, and positional encoding.

### `encoder_max_seq_len`
Maximum sequence length supported by encoder positional encoding.

### `decoder_max_seq_len`
Maximum sequence length supported by decoder positional encoding.

## Minimal config example

```yaml
model:
  _target_: trainite.models.encoder_decoder.build_encoder_decoder_model
  hidden_size: 128
  num_encoder_layers: 4
  num_decoder_layers: 4
  num_heads: 4
  feedforward_dim: 256
  dropout: 0.1
  encoder_max_seq_len: 64
  decoder_max_seq_len: 64
```

## When to change this file

Edit `models/encoder_decoder.py` if you want to:

- modify cross-attention behavior
- change how positional information is encoded (e.g. swap sinusoidal with relative/rotary embeddings)
- customize layer-normalization placement (e.g. Post-LN vs Pre-LN)

## Good starting rule

If the dataset is tiny, start with a small model:

- `hidden_size=64`
- `num_encoder_layers=2`
- `num_decoder_layers=2`
- `num_heads=2`
- `feedforward_dim=128`

If training is stable and the model underfits, scale up from there.
