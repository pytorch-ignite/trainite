# Python-Edu dataset

This built-in dataset loads the [`Avelina/python-edu-cleaned`](https://huggingface.co/datasets/Avelina/python-edu-cleaned) dataset from Hugging Face Datasets.

Python-Edu samples contain a `text` field with Python source code. The built-in transform uses this field as the causal language-modeling sequence.

## Configure the dataset

The dataset contains a `train` split. Trainite uses `DataWithAutoSplit` to automatically create the `train`, `val`, and `test` splits.

```yaml
data:
  dataset:
    path: Avelina/python-edu-cleaned
```

The default split ratios are:

* **train:** 80%
* **val:** 10%
* **test:** 10%

The ratios can be configured through `test_ratio` and `val_ratio`.

`path` is passed to Hugging Face `datasets.load_dataset`.

## Configure the transform

The transform can be configured with:

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

## Hugging Face authentication

The dataset is loaded and cached by Hugging Face Datasets. Do not put Hugging Face access tokens directly in `config.yaml`.

If authentication is required, authenticate through the Hugging Face CLI or the appropriate environment configuration.
