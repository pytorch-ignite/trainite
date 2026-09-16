# Python-Edu dataset

This built-in dataset loads the [`tyoc213/split-avelina-python-edu-distilled`](https://huggingface.co/datasets/tyoc213/split-avelina-python-edu-distilled) dataset from Hugging Face Datasets. It is a subset of [`Avelina/python-edu-cleaned`](https://huggingface.co/datasets/Avelina/python-edu-cleaned), which is based on the Python-Edu dataset from [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

Python-Edu samples contain a `text` field with Python source code. The built-in transform uses this field as the causal language-modeling sequence.

## Configure the dataset

The default configuration uses the `train` and `test` splits from the dataset, with validation created from the training data:

```yaml
data:
  train:
    dataset:
      path: tyoc213/split-avelina-python-edu-distilled
      split: "train[:90%]"

  val:
    dataset:
      path: tyoc213/split-avelina-python-edu-distilled
      split: "train[90%:]"

  test:
    dataset:
      path: tyoc213/split-avelina-python-edu-distilled
      split: test
```

`path` and `split` are passed directly to Hugging Face `datasets.load_dataset`.

## Configure the transform

Each split uses a transform configured with:

```yaml
data:
  train:
    transform:
      max_length: 128
```

`max_length` controls the maximum sequence length.

The tokenizer handles special-token addition. The transform creates:

* `train_input_ids`
* `train_label_ids`
* `attention_mask`
* `eval_input_ids`

## Hugging Face authentication

The dataset is downloaded and cached by Hugging Face Datasets. Do not put Hugging Face access tokens in `config.yaml`. Authenticate through the Hugging Face CLI or the environment when required.
