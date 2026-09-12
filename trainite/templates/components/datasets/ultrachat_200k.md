# UltraChat 200k dataset

This built-in dataset loads the UltraChat 200k dataset from Hugging Face
Datasets and applies a multi-turn conversation transform for causal language-modeling.

## Configure the dataset

The default configuration uses the UltraChat 200k dataset with explicit train and validation splits:

```yaml
data:
  train:
    dataset:
      path: HuggingFaceH4/ultrachat_200k
      split: train_sft
  val:
    dataset:
      path: HuggingFaceH4/ultrachat_200k
      split: test_sft
```

UltraChat samples contain a `messages` field with `role` and `content` for each conversation turn.

## Configure the transform

Each split uses a transform configured with:

```yaml
data:
  train:
    transform:
      max_length: 128
```

`max_length` controls the maximum sequence length.

The transform renders each turn as `User: <content>\n` and `Assistant: <content>\n`.
The tokenizer handles special-token addition. The transform creates:

* `train_input_ids`
* `train_label_ids`
* `attention_mask`
* `eval_input_ids`

During evaluation and inference logging, the prompt preserves the multi-turn context and ends with
`Assistant: ` so the model generates the final assistant response.

## Dataset splits

The `train_sft` split is used for training and `test_sft` is mapped to Trainite's validation split.
The generation-ranking splits (`train_gen` and `test_gen`) are outside the scope of this built-in wrapper.

## Hugging Face authentication

The dataset is downloaded and cached by Hugging Face Datasets. Do not put Hugging Face access
tokens in `config.yaml`. Authenticate through the Hugging Face CLI or the environment when required.
