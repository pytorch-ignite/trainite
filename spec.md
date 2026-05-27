# Tranite — Project Spec

**Version:** 0.1 (Draft)
**Project:**  PyTorch-Ignite LM Training Toolbox
**Stack:** PyTorch-Ignite, Pydantic, YAML, UV

---

## 1. What is Tranite?

Tranite is a cookiecutter-style toolbox for training language models with PyTorch-Ignite. It is not a framework. It does not own your training loop. Instead, it generates a clean, working project template — model, dataset, config, trainer, entrypoint — and gets out of your way.

The core idea: a researcher should go from zero to a running training loop in under five minutes, without writing boilerplate for checkpointing, logging, engine setup, or config parsing.

**Tranite is explicitly NOT:**
- A replacement for HuggingFace Trainer or Accelerate
- A tool for training large models on large corpora
- A framework that forces you into its abstractions
- A 200-line YAML configuration system

---

## 2. Design Philosophy

**Cookiecutter, not framework.** When you run `tranite init`, you get real, readable Python files. You own them. Tranite has no magic imports you must depend on at runtime — it gives you the skeleton and steps aside.

**Minimal config by default.** The starter `config.yaml` should be short enough to read in 30 seconds. Every key in it should be obviously necessary. Advanced options exist but are never generated unless asked for.

**Do whatever you want.** A user can override just the model, just the train step, just the dataset, or all of the above. They can also skip the YAML entirely and wire things manually. Nothing is sealed.

**PyTorch-Ignite stays visible.** Tranite does not hide Ignite's `Engine`, `Events`, or `State`. Users who want to attach custom handlers or tap into the event system should be able to do so without fighting the abstraction.

---

## 3. Core Abstractions

### 3.1 `Trainer`

The shared trainer structure used across the project. Each trainer is independent and owns the parts users never want to write themselves:

- Ignite `Engine` creation and wiring
- Optimizer and scheduler instantiation from config
- Evaluator setup (train evaluator + validation evaluator)
- Checkpoint saving: latest, best (by metric), periodic
- TensorBoard / logging setup
- A `metrics` dict that subclasses and users populate
- Handler attachment: a single `attach_handlers(trainer)` hook that gives callers full access to `trainer.engine`, `trainer.model`, `trainer.optimizer`, `trainer.scheduler`, `trainer.train_loader`, `trainer.val_loader`, `trainer.train_evaluator`, `trainer.val_evaluator`

`Trainer` does **not** define `train_step` or `eval_step`. Those are the responsibility of the specific trainer implementation or the user.

**Key interface (subject to refinement):**

```python
class Trainer:
    engine: Engine
    model: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler | None
    train_loader: DataLoader
    val_loader: DataLoader | None
    train_evaluator: Engine | None
    val_evaluator: Engine | None
    metrics: dict[str, Metric]

    def run(self) -> None: ...
```

### 3.2 `PreTrainer`

Standalone trainer for training a model from scratch (next-token prediction / sequence-to-sequence). Provides a default `train_step` and `eval_step` suitable for autoregressive language modeling. User can override either or both.

**Scope:** standard supervised training on a dataset with a loss function.
### 3.3 `RLTrainer`

Standalone trainer for post-training / reinforcement learning workflows. Provides a default `train_step` that supports a reward function. The reward function is user-supplied — Tranite defines the interface, not the reward logic.

**Scope:** RLHF-adjacent training, reward-signal-based fine-tuning. The exact RL algorithm (PPO, GRPO, etc.) is out of scope for v1.

### 3.4 `Config`

Configuration is loaded from a YAML file and parsed into a Pydantic model. This gives users type checking, validation errors with clear messages, and IDE autocomplete.

The config is split into sections. Users extend the config by adding fields both to the Pydantic model **and** the YAML — there is no dynamic field injection. This is intentional: it keeps configs explicit and auditable.

**Minimal config sections:**
- `model` — architecture hyperparameters
- `dataset` — data source, split ratios, batch size, sequence length
- `optimizer` — class path + kwargs (e.g. `torch.optim.AdamW`)
- `scheduler` — optional, class path + kwargs
- `loss_fn` — function path
- `trainer` — train/eval step paths, device, max epochs, run args
- `metrics` — dict of name → metric factory path
- `handlers` — optional, path to handler attachment function + kwargs
- `output` — output directory, run name

**What is not in the starter config:** individual checkpoint policies, TensorBoard flush intervals, handler-level knobs.Users can add them if they need them.

---

## 4. CLI and Init Flow

### `tranite init`

The primary entrypoint. Two modes:

**Flag mode** — pass everything as CLI arguments:
```
tranite init --model=transformer --dataset=string-reverse --trainer=pre
```

**Interactive mode** — prompted step by step, like `npm init`:
```
$ tranite init

? Project name: my-lm-experiment
? Trainer type: (pre / rl) › pre
? Built-in model or bring your own? › built-in
? Choose a model: › transformer
? Built-in dataset or bring your own? › built-in
? Choose a dataset: › string-reverse
? Output directory: › ./output

✔ Generated config.yaml
✔ Generated model.py
✔ Generated dataset.py
✔ Generated trainer.py
✔ Generated main.py

Run: python main.py config.yaml
```

The goal is under 7 steps to a runnable experiment. Interactive mode should not ask about things the user can customize later in the YAML.

### `tranite run` (tentative)

```
tranite run config.yaml --output=./output
```

A thin wrapper around `python main.py config.yaml`. May be dropped if it adds no value over running main.py directly.

---

## 5. Generated Template Structure

After `tranite init`, the user has:

```
<project>/
├── config.yaml        # Minimal YAML config — start here
├── model.py           # Model definition (built-in copy or blank template)
├── dataset.py         # Dataset + dataloader utilities
├── trainer.py         # Trainer subclass (optional overrides)
├── main.py            # Entrypoint: load config → build objects → run
```

**Everything is a real file the user owns.** Tranite is not imported at runtime by default. The generated `trainer.py` may import shared utilities from `tranite.trainers`, but each trainer remains independent and self-contained.

---

## 6. Built-in Components

These ship with Tranite and are available to select during `tranite init` or reference from config.

### Models

| Name | Architecture | Notes |
|---|---|---|
| `transformer` | Decoder-only Transformer | Baseline; matches prototype |
| `encoder-decoder` | Encoder-Decoder Transformer | For seq2seq tasks |
| `rwkv` | RWKV v7 | <link to model to add>
| `mamba` | Mamba SSM | Stretch goal; depends on contributor bandwidth |


Models are provided as complete, editable copies — not as black-box imports. The user is expected to read and modify them.

### Datasets

| Name | Task | Notes |
|---|---|---|
| `string-reverse` | Reverse a random string | Primary prototype dataset |
| `arithmetic` | Integer arithmetic (add, subtract) | Synthetic; tests generalization |

Both are synthetic and CPU-generatable. No downloads required.

### Metrics

Metrics are task-coupled: the built-in datasets ship with appropriate default metrics.

- `string-reverse` → token accuracy, exact-match accuracy
- `arithmetic` → exact-match accuracy, expression validity rate

Users can add additional Ignite metrics by registering them in the `metrics` dict on the trainer.

---

## 7. Extensibility

Tranite is designed so that users can override at any layer without touching others.

| What to override | How |
|---|---|
| Model architecture | Edit `model.py` freely |
| Dataset / dataloader | Edit `dataset.py`; point config to new loader factory |
| Train step | Set `trainer.train_step` in config to a function path, or override `train_step()` in `trainer.py` |
| Eval step | Same as train step |
| Handlers | Set `handlers.path` in config to a function that receives the `trainer` object |
| Metrics | Add to `metrics` dict in config or in trainer subclass |
| Entire trainer | Replace the trainer implementation directly and override whatever you need |

The handler interface deserves special note. Because the handler function receives the full `trainer` object, it can attach any Ignite event handler to `trainer.engine` — giving full access to the Ignite event system without Tranite needing to model it in config.

---

## 8. Output Structure

```
output/
└── <run_name>/
    └── <YYYYMMDD_HHMM>/
        ├── output.log
        ├── config.yaml          # Copy of config used for this run
        ├── last.pt              # Latest checkpoint
        ├── best.pt              # Best checkpoint by tracked metric
        ├── code/                # Snapshot of user's source files
        │   ├── main.py
        │   ├── model.py
        │   └── ...
        └── tensorboard/         # TensorBoard event files
```

Run name defaults to `<model>__<dataset>` if not set. The code snapshot ensures reproducibility — the exact files used for a run are always recoverable.

---

## 9. Testing Strategy

Testing for Tranite has two distinct concerns that must be addressed separately.

### 9.1 Unit Testing

The CLI, config parsing, and file generation logic should have unit tests covering:

- `tranite init` generates the correct files for each combination of flags
- Generated files are valid Python (parseable without errors)
- Config validation raises clear errors on malformed or missing fields
- Output directory structure matches the spec

### 9.2 Integration Testing

There should be integration test per built-in dataset and model that:

1. Runs `tranite init` with a given configuration
2. Runs `python main.py config.yaml` for a small number of epochs
3. Asserts that training loss decreases
4. Asserts that `last.pt` and `best.pt` checkpoints are saved
5. Asserts that TensorBoard logs are written to the output directory

### 9.3 CI

1. Installs dependencies (PyTorch, Ignite, trainite itself, other external dependencies if any)
2. Runs unit tests and integration tests on every PR.



---

## 10. Open Questions

1. **Distributed training** — How to approach it?
    => We can scope DDP and use `idist.auto_*` methods
    => `trainite run` could handle more verbose `torchrun ...`
3. **RL algorithm** — `RLTrainer` needs a concrete algorithm (PPO, GRPO, or a simpler REINFORCE baseline) to be implementable. Decision needed early in the project.
    => let's say GRPO algorithm
5. **Handler discoverability** — users writing custom handlers need to know what attributes are on `trainer`. Should there be a `trainer.summary()` CLI command, or is documentation enough?
6. **Mamba dependency** — `mamba-ssm` has non-trivial CUDA build requirements. Should it be an optional install extra, or deferred entirely?
    => Let's put this out of the scope of v1 for now
7. **Model Scope** - We need to decide the scope of the project when it comes to which models are we covering. Currently, the spec also states supportin RLTrainer in addition to the other models such as Mamba, transformer, encoder-decoder. A discussion on the models we aim to in v1 will help us plan better.
---

## 11. Out of Scope (v1)

- Training models larger than fit on a single GPU
- Integration with HuggingFace Hub, datasets, or tokenizers
- Inference / serving tooling
- Experiment comparison UI (beyond TensorBoard user can add theirs)
- `trainite` cmd can optionally setup a code agent via a harness to help coding
