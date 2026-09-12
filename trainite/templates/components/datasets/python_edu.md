# Python-Edu dataset

This built-in dataset loads the `python-edu` config of
[`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
and applies a causal language-modeling transform to each sample.

## Configure the dataset

```yaml
data:
  dataset:
    split: train
    min_int_score: 4
    max_samples: null
```

* `min_int_score` filters to files scored at or above this value by
  HuggingFaceTB's educational-code classifier (the upstream default is `4`).
* `max_samples` caps how many rows are loaded — useful for a quick local
  smoke test before running on the full ~7.7M-row dataset.

## File contents are downloaded separately

Unlike a typical Hugging Face text dataset, `python-edu` ships only file
metadata (`blob_id`, `repo_name`, `path`, `length_bytes`, `score`,
`int_score`) — no `text` field. The actual source file for each row is
fetched, gzip-decoded, from Software Heritage's public S3 bucket; no AWS
credentials are required.

* **With `max_samples` set**, the dataset streams rows from the Hub lazily
  and stops as soon as enough rows pass the score filter and download
  successfully — it never downloads the full ~600MB metadata file. This is
  the fast path, intended for local iteration.
* **With `max_samples` unset** (a full training run), the full split's
  metadata is downloaded and cached by Hugging Face `datasets` up front (a
  one-time cost reused by every subsequent run), then every row's file
  content is fetched.

A small number of files can fail to download (removed from Software
Heritage, etc.); those rows are filtered out automatically.

Refer to [the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids)
for the data's license terms before redistributing or training on this content.

## Configure the transform

```yaml
data:
  transform:
    max_length: 128
```

`max_length` controls the maximum sequence length. The tokenizer handles
special-token addition. The transform creates:

* `train_input_ids`
* `train_label_ids`
* `attention_mask`
* `eval_input_ids`

## Dataset splits

`python-edu` only has a `train` split upstream, so this dataset uses
Trainite's auto-split strategy (`test_ratio` / `val_ratio` in `config.yaml`)
rather than explicit `train`/`val`/`test` splits.

## Hugging Face authentication

The dataset metadata is downloaded and cached by Hugging Face Datasets. Do
not put Hugging Face access tokens in `config.yaml`. Authenticate through
the Hugging Face CLI or the environment when required.
