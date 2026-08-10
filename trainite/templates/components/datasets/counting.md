# Counting dataset

This dataset generates random sequences from the alternating language `L_k` (composed of alternating blocks of 'a' and 'b' characters), along with prefix-classification targets indicating if each character belongs to the final alternating block.

## What each sample looks like

Each dataset item is a dictionary containing:

* `source`: the sequence of alternating characters (e.g. `"aaabbb"`)
* `target`: the target prefix classification sequence of `"0"`s and `"1"`s (e.g. `"000111"`)

## Tokenizer

This dataset yields raw text strings (`source` and `target`). These strings are processed by the project's preprocessor and tokenizer before being passed to the model.

## Config knobs

### `total_size`
The maximum number of unique sequences to generate.

### `k`
The target sequence block alternation count (k).

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
      _target_: datasets.counting.CountingDataset
      total_size: 1000
      k: 3
      min_seq_len: 201
      max_seq_len: 250
```

## When to change this file

Edit `datasets/counting.py` if you want to:

* change the alternating blocks generation rules
* change how targets/prefix boundaries are calculated
* modify how padding or labels are formatted
