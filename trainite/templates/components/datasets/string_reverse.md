# String Reverse dataset

This dataset gives you a very small task: feed in a random character sequence, predict the reversed sequence.

It is useful because it is easy to understand, fast to generate, and good for checking whether the model and training loop are wired correctly.

## What each sample looks like

Each dataset item is a dictionary of raw strings:

```python
{
    "source": "abc",
    "target": "cba",
}
```

## Tokenizer

This dataset yields raw text strings (`source` and `target`). These strings are processed by the project's preprocessor before being passed to the model.

## Config knobs

### `per_seq_size`
How many unique examples to generate per sequence length bucket.

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
      _target_: dataset_impl.string_reverse.StringReverseDataset
      per_seq_size: 512
      charset: "@alpha"
      min_seq_len: 2
      max_seq_len: 12
```

## When to change this file

Edit `dataset_impl/string_reverse.py` if you want to:

- change the tokenization rules
- add or remove allowed characters
- make the task longer or harder
- change how padding or labels are built

## Good starting rule

If you only want to test the pipeline, keep the defaults and just change:

- `per_seq_size`
- `charset`
- `min_seq_len` / `max_seq_len`
