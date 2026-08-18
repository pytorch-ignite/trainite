# Hugging Face dataset

This template downloads one dataset split through Hugging Face Datasets and applies your transform lazily as samples
enter the Trainite pipeline. Hugging Face manages the local cache; dataset files are not stored in this project.

## Configure the source

Edit `data.dataset` in `config.yaml`:

```yaml
data:
  dataset:
    _target_: datasets.load_dataset
    path: namespace/dataset-name
    name: null
    split: train
    revision: null
```

- `path` is the Hub dataset ID.
- `name` selects an optional dataset subset/configuration.
- `split` must select one map-style split.
- `revision` may be a branch, tag, or commit hash. Pin a commit for reproducible runs.

Other `load_dataset` parameters remain available when explicitly added to the config. For example, `data_files`
selects physical files for local or custom file-based datasets; it does not control Trainite's random split ratios.

## Implement the transform

Edit `HuggingFaceTransform.__call__` in `dataset_impl/hugging_face.py`. It receives one raw dictionary from the selected
dataset and must return the included `DatapointModel`. The comments on that model explain every tensor and logging
field expected by the causal-LM collate function and trainer.

`data.transform.max_length` defaults to `128` and is passed to the transform constructor. Add another transform config
field only when you also add the matching constructor parameter.

The generated placeholder deliberately raises `NotImplementedError`: dataset columns and training objectives differ,
so Trainite cannot infer a correct generic transform.

## Limits

Streaming is unsupported because the current Trainite pipeline requires dataset length and random indexing for
splitting, shuffling, and inference samples. Never put a Hugging Face access token in `config.yaml`; authenticate through the
Hugging Face CLI or environment because Trainite copies configs into run outputs.
