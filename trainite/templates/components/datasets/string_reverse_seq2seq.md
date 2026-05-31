# String Reverse dataset (Seq2Seq)

This dataset gives you a very small task: feed in a random character sequence, predict the reversed sequence.

It is useful because it is easy to understand, fast to generate, and good for checking whether the model and training loop are wired correctly.

## What each sample looks like

Each example is split into source and target sequences:

*   **Source prompt** (sent to the Encoder):
    ```text
    <bos> abc <eos>
    ```
*   **Target output** (shifted by one position for Decoder teacher forcing):
    *   Decoder Input: `<bos> cba`
    *   Expected Labels: `cba <eos>`

The model is asked to read the input sequence using the encoder, then produce the reversed text autoregressively using the decoder.

## What the dataset returns

Each item is a dictionary:

```python
{
    "encoder_input_ids": Tensor,
    "decoder_input_ids": Tensor,
    "decoder_labels": Tensor,
}
```

Use `collate_fn` to pad a batch:

- `encoder_input_ids` and `decoder_input_ids` are padded with `0`
- `decoder_labels` are padded with `-100`

## Tokenizer

The dataset uses a simple character tokenizer with these special tokens:

- `0` = `<pad>`
- `1` = `<bos>`
- `2` = `<eos>`
- `3` = `<unk>`

It supports printable ASCII characters by default.

## Config knobs

### `size`
How many examples to generate.

### `charset`
Which characters the random strings can use.

Built-in presets:

- `@universal` — all printable ASCII + space
- `@alpha` — letters only
- `@digits` — numbers only
- `@alphanumeric` — letters and numbers

You can also pass a custom string, for example `"abc123"`.

### `seq_len`
Use one fixed sequence length for every sample.

### `min_seq_len` / `max_seq_len`
Use a random length in this range.

Do not set `seq_len` together with `min_seq_len` or `max_seq_len`.

### `seed`
Controls dataset generation so runs are repeatable.

## Minimal config example

```yaml
data:
  train:
    dataset:
      _target_: trainite.datasets.string_reverse_seq2seq.build_string_reverse_dataset
      size: 512
      charset: "@alpha"
      min_seq_len: 2
      max_seq_len: 12
```

## When to change this file

Edit `trainite/datasets/string_reverse_seq2seq.py` if you want to:

- change the tokenization rules
- add or remove allowed characters
- make the task longer or harder
- change how padding or labels are built

## Good starting rule

If you only want to test the pipeline, keep the defaults and just change:

- `size`
- `charset`
- `min_seq_len` / `max_seq_len`
