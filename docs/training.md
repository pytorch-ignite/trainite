# Training Guide

This page explains the YAML configuration that Trainite expects and how to kick off a training run.
For CLI commands that generate and extend projects, see the [CLI Guide](cli.md).

For a working example you can run right away, see the [string reversal example](https://github.com/pytorch-ignite/trainite/tree/main/examples/string_reversal).

## Configuration overview

Every Trainite experiment is described by a single YAML file. At startup the file is loaded, validated against `ProjectConfig`, and used to build all the components (model, optimizer, data pipeline, etc.) automatically.

Here is a stripped-down config to show the overall shape:

```yaml
project_name: my_experiment

preprocessor:
  _target_: preprocessors.char_tokenizer.CharTokenizer

model:
  _target_: models.rope_transformer.RoPETransformerModel
  collate_fn_target: models.rope_transformer.CausalLMCollateFn
  hidden_size: 64
  num_layers: 2
  num_heads: 2
  feedforward_dim: 128
  dropout: 0.1
  max_seq_len: 128

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.0003

loss:
  _target_: torch.nn.CrossEntropyLoss
  ignore_index: -100

data:
  dataset:
    _target_: dataset_impl.string_reverse.StringReverseDataset
    per_seq_size: 1000
    charset: "@alphanumeric"
    min_seq_len: 1
    max_seq_len: 16
  transform:
    _target_: dataset_impl.string_reverse.PromptCompletionTransform
    ignore_index: -100
  dataloader:
    batch_size: 128
    shuffle: true
  val_ratio: 0.1
  test_ratio: 0.1

trainer:
  epochs: 3
  log_every_steps: 50

output:
  root: outputs
  run_name: first_run

logger: tensorboard
seed: 42
device: null
```

### How `_target_` works

Any block that needs to create a Python object uses the `_target_` key. The value is a dotted import path — Trainite will import that path and pass the remaining keys as arguments. For example:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.0003
```

The `preprocessor`, `model`, `optimizer`, `loss`, and dataset/transform blocks all use this pattern (for example `data.dataset` in auto-split configs, or `data.train.dataset` / `data.val.dataset` in explicit-split configs).

### Configuration blocks

**`project_name`** — Name of the project.

**`preprocessor`** — The tokenizer or preprocessing component. Must provide a `_target_`.
Common starter options are a character tokenizer (`preprocessors.char_tokenizer.CharTokenizer`)
or GPT-2 tokenizer loader (`preprocessors.gpt2_tokenizer.load_gpt2_tokenizer`), depending on dataset choice.

**`model`** — Model architecture and hyperparameters. The `_target_` points to the model class (such as `models.rope_transformer.RoPETransformerModel`); constructor arguments like `hidden_size`, `num_layers`, and `num_heads` are passed through. You can also specify `collate_fn_target` (e.g. `models.rope_transformer.CausalLMCollateFn`) to configure the custom batch collation function used by the data loaders.

**`optimizer`** — Defaults to `torch.optim.AdamW` with `lr=0.001` if not specified.

**`loss`** — Defaults to `torch.nn.CrossEntropyLoss` with `ignore_index=-100` if not specified. Point `_target_` at a custom loss (defined next to its model or in an installed package) to override it; remaining keys are passed to its constructor.

**`data`** — Datasets, transforms, and dataloaders. Trainite supports two ways to set up your data splits:

  1. **Auto-split** — provide a single `dataset` block (and optional `transform`) along with `val_ratio` and `test_ratio`. Trainite calls `torch.utils.data.random_split` to divide the data. This is what most starter projects use.
  2. **Explicit splits** — provide separate `train`, `val`, and optionally `test` blocks, each with its own `dataset`, `transform`, and `dataloader` config.

Note that `dataloader` blocks accept standard PyTorch `DataLoader` options (such as `batch_size`, `shuffle`, and `num_workers`) rather than using a `_target_`.

**`trainer`** — Training loop parameters: `epochs`, `log_every_steps`, `early_stopping_patience` (set to `null` to disable), `inference_every_epochs`, `inference_num_samples`, `max_inference_new_tokens`, and `grad_clip_norm`.

**`output`** — Where artifacts are saved. `root` specifies the parent directory and `run_name` identifies the experiment run. Artifacts are saved to `<root>/<run_name>/<timestamp>/`.

**`logger`** — Either `tensorboard` (default) or `clearml`.

**`seed`** — Random seed for reproducibility. Defaults to `42`.

**`device`** — `cpu`, `cuda`, or `null` to let Trainite pick automatically via PyTorch-Ignite's distributed utilities.

### Starter component choices

`trainite init` currently provides:

- Models: `rope-transformer`, `basic-transformer`
- Datasets: `string-reverse`, `counting`, `hugging-face`, `wikitext`, `ultrachat-200k`, `python-edu`
- Trainer: `decoder-trainer`

Tokenizer defaults by dataset family:

- `string-reverse`, `counting` → character tokenizer (`preprocessors.char_tokenizer.CharTokenizer`)
- `hugging-face`, `wikitext`, `ultrachat-200k`, `python-edu` → GPT-2 tokenizer (`preprocessors.gpt2_tokenizer.load_gpt2_tokenizer`)

Split behavior by dataset family:

- `string-reverse`, `counting`, `hugging-face`, `python-edu` use auto-split (`val_ratio` / `test_ratio`).
- `wikitext` and `ultrachat-200k` use explicit dataset splits in config (`train` / `val` / optional `test`).

## Running training

Each example has a `main.py` entry point. From the example directory:

```bash
uv run python main.py config.yaml
```

Or without `uv`:

```bash
python main.py config.yaml
```

What happens under the hood:

1. The YAML is loaded and validated into a `ProjectConfig` instance.
2. The preprocessor, model, and dataset/transform components are instantiated from their `_target_` entries. DataLoaders are constructed from the `dataloader` options and `collate_fn_target`. The optimizer is created with the model parameters.
3. Logging, metrics, and the output directory are set up.
4. `trainer.run()` starts the PyTorch-Ignite training loop — running epochs, evaluating on the validation set, saving checkpoints, and optionally running test evaluation at the end.

## Outputs

Each run creates a timestamped directory under `<output.root>/<run_name>/`:

```
outputs/
└── first_run/
    └── 20260901_143022/
        ├── config.yaml         # Copy of the run configuration
        ├── output.log          # Training logs
        ├── best.pt             # Best model checkpoint (by validation loss)
        ├── last.pt             # Most recent checkpoint
        └── tensorboard/        # TensorBoard event files (when using tensorboard logger)
```

If you're using TensorBoard, you can visualize metrics with:

```bash
uv run tensorboard --logdir outputs
```

## Examples

Working examples are the best way to understand how everything fits together:

* [**String Reversal**](https://github.com/pytorch-ignite/trainite/tree/main/examples/string_reversal) — a sequence-to-sequence toy task that trains a decoder-only Transformer with RoPE to reverse character strings.
* [**Counting**](https://github.com/pytorch-ignite/trainite/tree/main/examples/counting) — trains a decoder-only Transformer to count through an alternating block language, reproducing experiments from [*Knee-Deep in C-RASP: A Transformer Depth Hierarchy*](https://arxiv.org/abs/2506.16055).
