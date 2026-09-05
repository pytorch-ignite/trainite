# WikiText dataset

This built-in dataset loads a WikiText dataset configuration from Hugging Face
Datasets and applies a causal language-modeling transform to each sample.

## Configure the dataset

The default configuration uses the WikiText-2 raw dataset with explicit train, validation, and
test splits:

```yaml
data:
  train:
    dataset:
      path: Salesforce/wikitext
      name: wikitext-2-raw-v1
      split: train
  val:
    dataset:
      path: Salesforce/wikitext
      name: wikitext-2-raw-v1
      split: validation
  test:
    dataset:
      path: Salesforce/wikitext
      name: wikitext-2-raw-v1
      split: test
```

The available WikiText configurations are:

* `wikitext-103-raw-v1`
* `wikitext-103-v1`
* `wikitext-2-raw-v1`
* `wikitext-2-v1`

WikiText samples contain a single `text` field. The built-in transform uses this field as the
causal language-modeling sequence.

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

## Dataset splits

WikiText provides native `train`, `validation`, and `test` splits on Hugging Face. The default
configuration maps each split directly to the corresponding Hugging Face split, so no automatic
splitting is performed.

## Hugging Face authentication

The dataset is downloaded and cached by Hugging Face Datasets. Do not put Hugging Face access
tokens in `config.yaml`. Authenticate through the Hugging Face CLI or the environment when required.
