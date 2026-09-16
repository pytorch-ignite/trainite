# Python-Edu dataset

This built-in dataset loads
[`tyoc213/split-avelina-python-edu-distilled`](https://huggingface.co/datasets/tyoc213/split-avelina-python-edu-distilled)
— a cleaned, pre-split dataset derived from
[`Avelina/python-edu-cleaned`](https://huggingface.co/datasets/Avelina/python-edu-cleaned),
which is based on the `python-edu` configuration of
[`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

Unlike the original `smollm-corpus` Python-Edu dataset, this dataset already
includes the source code in a `text` column, so no additional file downloads
are required.

## Configure the dataset

```yaml
data:
  train:
    dataset:
      path: tyoc213/split-avelina-python-edu-distilled
      split: "train[:90%]"
```

Like the WikiText dataset, `path` and `split` are plain config values passed
directly to Hugging Face `datasets.load_dataset`.

## No separate file download

The original
[`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
Python-Edu configuration provides file metadata such as `blob_id`,
`repo_name`, `path`, `length_bytes`, `score`, and `int_score`, with the actual
source code stored separately in Software Heritage.

In contrast,
[`tyoc213/split-avelina-python-edu-distilled`](https://huggingface.co/datasets/tyoc213/split-avelina-python-edu-distilled)
already contains the decoded source code in its `text` column. The dataset
can therefore be loaded directly with `datasets.load_dataset`, without
additional per-file downloads, AWS access, or Software Heritage requests.

The dataset is downloaded and cached by Hugging Face `datasets` on first use.
Subsequent runs reuse the local cache.

## Configure the transform

```yaml
data:
  transform:
    max_length: 128
```

`max_length` controls the maximum sequence length passed to the tokenizer.
The transform creates:

* `train_input_ids`
* `train_label_ids`
* `attention_mask`
* `eval_input_ids`

## Dataset splits

The upstream dataset provides `train` and `test` splits but does not provide
a separate `val` split.

The default Trainite configuration creates the validation split using
Hugging Face split-slicing syntax:

* **`train`**: `"train[:90%]"` — the first 90% of the upstream `train` split.
* **`val`**: `"train[90%:]"` — the remaining 10% of the upstream `train` split.
* **`test`**: `"test"` — the upstream test split, used as-is.

The train and validation splits are therefore non-overlapping. You can adjust
the train/validation ratio by changing the split expressions in the dataset
configuration.

## Hugging Face authentication

The dataset is loaded and cached by Hugging Face Datasets. Do not put Hugging
Face access tokens directly in `config.yaml`.

If authentication is required for the dataset, authenticate through the
Hugging Face CLI or the appropriate environment configuration.
