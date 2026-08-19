import importlib
import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch
import yaml
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    create_lr_scheduler_with_warmup,
)
from ignite.handlers.clearml_logger import ClearMLSaver
from ignite.handlers.fbresearch_logger import FBResearchLogger
from ignite.handlers.param_scheduler import ParamScheduler
from ignite.utils import setup_logger
from omegaconf import OmegaConf
from pydantic import BaseModel
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset, random_split

from trainite.config.base import (
    DataConfigBase,
    DataWithAutoSplit,
)
from trainite.datasets.transformed import TransformedDataset

T = TypeVar("T", bound=BaseModel)


def get_target(target_path: str) -> Any:
    """Resolve a dotted import path to a Python class or function.

    Trainite uses a ``_target_`` key in config files (e.g.
    ``_target_: trainite.models.transformer.TransformerModel``) to specify which
    class or function should be instantiated at runtime without hard-coding the
    import.  This function performs that resolution by splitting the path on the
    last dot, importing the module, and returning the named attribute.
    """
    if not target_path:
        raise ValueError("The '_target_' key must not be empty.")

    try:
        # Split 'module.submodule.ClassName' into 'module.submodule' and 'ClassName'
        module_path, symbol_name = target_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        target_symbol = getattr(module, symbol_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not locate target '{target_path}': {e}") from e

    return target_symbol


def _inject_if_accepted(target_symbol: Any, *, allow_var_kwargs: bool = True, **candidates: Any) -> dict[str, Any]:
    """Filter keyword arguments to only those accepted by the target's signature.

    When building datasets or transforms, the caller may want to pass ``tokenizer``
    or ``preprocessor`` to the component — but not every component declares those
    parameters.  This function inspects the target's signature and silently drops
    any candidates it does not accept, so components remain decoupled from the
    calling convention.

    If the target accepts ``**kwargs`` and ``allow_var_kwargs`` is true, all
    candidates are passed through as-is.
    """
    try:
        sig = inspect.signature(target_symbol)
        if allow_var_kwargs and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return candidates
        return {k: v for k, v in candidates.items() if k in sig.parameters}
    except Exception:
        return candidates


def instantiate(config: BaseModel, **kwargs) -> Any:
    """Instantiate a class or call a function described by a Pydantic config.

    The config must contain a ``_target_`` field with a dotted import path (e.g.
    ``trainite.models.transformer.TransformerModel``).  All other fields in the
    config are forwarded as keyword arguments to the target.

    Extra ``**kwargs`` (e.g. ``vocab_size``, ``pad_token_id``) override or
    supplement the config fields, allowing the caller to inject runtime-resolved
    values that are not known at config-parse time.

    ``_inject_if_accepted`` ensures that only parameters the target actually
    declares are passed, so components are not required to accept every possible
    caller keyword.
    """
    if isinstance(config, BaseModel):
        config_dict = config.model_dump(by_alias=True, polymorphic_serialization=True)
    else:
        raise ValueError("Config must be an instance of BaseModel")

    if not isinstance(config_dict, dict) or "_target_" not in config_dict:
        raise ValueError("Config must be a dict containing a '_target_' key")

    # Work on a copy to avoid mutating the original config data
    params = config_dict.copy()
    target_path = params.pop("_target_")

    target_symbol = get_target(target_path)

    final_kwargs = {**params, **kwargs}
    final_kwargs = _inject_if_accepted(target_symbol, **final_kwargs)

    return target_symbol(**final_kwargs)


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping")
    return data


def dump_yaml(data: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def dump_config(config: BaseModel, path: str | Path) -> None:
    data = config.model_dump(by_alias=True, polymorphic_serialization=True)
    dump_yaml(data, path)


def load_config(path: str | Path, config_cls: type[T]) -> T:
    raw_conf = OmegaConf.load(path)
    return config_cls.model_validate(raw_conf)


# ==========================================
# Builders (Dataset, Dataloader, Model Setup)
# ==========================================


# Builds the model based on the provided configuration, tokenizer, and device.
def build_model(model_config: Any, device: str | torch.device, **kwargs) -> nn.Module:
    target_symbol = get_target(model_config.target)
    kwargs = _inject_if_accepted(target_symbol, **kwargs)
    model = instantiate(model_config, **kwargs)
    model.to(device)
    return model


def build_dataset(dataset_config: Any, transform_config: Any, tokenizer: Any) -> Dataset:
    ds = get_target(dataset_config.target)
    dataset = instantiate(
        dataset_config,
        **_inject_if_accepted(
            ds,
            allow_var_kwargs=False,
            preprocessor=tokenizer,
            tokenizer=tokenizer,
        ),
    )
    transform = None
    if transform_config is not None:
        tf = get_target(transform_config.target)
        transform = instantiate(
            transform_config, **_inject_if_accepted(tf, preprocessor=tokenizer, tokenizer=tokenizer)
        )
    # Wraps the dataset with the transform if provided, otherwise returns the dataset as is.
    return TransformedDataset(dataset, transform)


def create_dataloader(
    dataset: Dataset,
    dl_config: Any,
    tokenizer: Any,
    shuffle: bool | None = None,
    collate_fn_target: str | None = None,
) -> DataLoader:
    dl_kwargs = dl_config.model_dump(exclude={"shuffle"})
    if shuffle is None:
        shuffle = getattr(dl_config, "shuffle", False)
    collate_fn = None
    if collate_fn_target:
        target_symbol = get_target(collate_fn_target)
        if isinstance(target_symbol, type):
            collate_fn = target_symbol(tokenizer=tokenizer)
        else:
            collate_fn = target_symbol
    return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **dl_kwargs)


def _loaders_from_splits(
    data_config: DataConfigBase,
    tokenizer: Any,
    collate_fn_target: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build DataLoaders from explicitly defined train/val/test split configs.

    Each split (``data_config.train``, ``data_config.val``, ``data_config.test``)
    specifies its own dataset, transform, and dataloader config independently.
    The test loader is optional — if ``data_config.test`` is absent, ``None`` is returned.
    """

    def _make(split_config: Any) -> DataLoader:
        ds = build_dataset(split_config.dataset, split_config.transform, tokenizer)
        return create_dataloader(ds, split_config.dataloader, tokenizer, collate_fn_target=collate_fn_target)

    return _make(data_config.train), _make(data_config.val), _make(data_config.test) if data_config.test else None


def _loaders_from_ratios(
    data_config: DataWithAutoSplit,
    tokenizer: Any,
    seed: int,
    collate_fn_target: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build DataLoaders by splitting a single dataset by ratio.

    A single dataset is built and then split into train/val/test subsets using
    ``torch.utils.data.random_split`` with a fixed ``seed`` for reproducibility.
    Split sizes are derived from ``val_ratio`` and ``test_ratio``; the remainder
    goes to training.  The test loader is omitted (``None``) when ``test_ratio``
    leaves zero samples for testing.
    """
    dataset = build_dataset(data_config.dataset, data_config.transform, tokenizer)
    total_len = len(dataset)  # type: ignore
    if total_len == 0:
        raise ValueError("Training dataset is empty. Cannot perform train/val/test split.")
    val_ratio = data_config.val_ratio
    test_ratio = data_config.test_ratio
    train_len = int(total_len * (1.0 - test_ratio - val_ratio))
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )
    dl = data_config.dataloader
    return (
        create_dataloader(train_ds, dl, tokenizer, shuffle=True, collate_fn_target=collate_fn_target),
        create_dataloader(val_ds, dl, tokenizer, shuffle=False, collate_fn_target=collate_fn_target),
        create_dataloader(test_ds, dl, tokenizer, shuffle=False, collate_fn_target=collate_fn_target)
        if test_len > 0
        else None,
    )


def build_dataloaders(
    data_config: Any,
    tokenizer: Any,
    seed: int,
    collate_fn_target: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Entry point for building train/val/test DataLoaders from a data config.

    Supports two config strategies (set in ``config.yaml`` under ``data:``):

    * **Explicit splits** (``DataConfigBase``): ``train``, ``val``, and optionally
      ``test`` are each configured independently with their own dataset and
      dataloader settings.
    * **Auto-split** (``DataWithAutoSplit``): a single dataset is split into
      train/val/test by ratio, sharing the same dataloader settings.
    """
    if isinstance(data_config, DataWithAutoSplit):
        return _loaders_from_ratios(data_config, tokenizer, seed, collate_fn_target)
    return _loaders_from_splits(data_config, tokenizer, collate_fn_target)


# ==========================================
# Handlers (Ignite Engine Wiring & Setup)
# ==========================================


def make_run_dir(output_config: Any) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = output_config.run_name
    run_dir = Path(output_config.root) / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def attach_lr_scheduler(
    engine: Engine,
    optimizer: Any,
    total_iters: int,
    peak_lr: float,
) -> None:
    """Attach a linear warm-up + linear decay learning-rate schedule to the engine.

    The schedule has two phases:
    1. **Warm-up** (first 10 % of iterations): LR ramps linearly from 0 to `peak_lr`.
    2. **Decay** (remaining iterations): LR decays linearly from `peak_lr` to 0.

    This is a common schedule for Transformer training.  The ``create_lr_scheduler_with_warmup``
    helper from PyTorch-Ignite wraps a standard ``torch.optim.lr_scheduler`` and
    fires after every iteration via ``Events.ITERATION_COMPLETED``.

    See: https://docs.pytorch.org/ignite/generated/ignite.handlers.param_scheduler.create_lr_scheduler_with_warmup.html
    """
    # Reserve at least 2 iterations for warm-up even for very short training runs
    warmup_iters = max(2, int(0.1 * total_iters))
    linear_decay = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_iters - warmup_iters,
    )
    scheduler: ParamScheduler = create_lr_scheduler_with_warmup(
        linear_decay,
        warmup_start_value=0.0,
        warmup_end_value=peak_lr,
        warmup_duration=warmup_iters,
    )
    engine.add_event_handler(Events.ITERATION_COMPLETED, scheduler)


def attach_early_stopping(
    val_evaluator: Engine,
    trainer_engine: Engine,
    patience: int,
) -> None:
    """Stop training early when validation loss stops improving.

    ``EarlyStopping`` from PyTorch-Ignite monitors a score function after each
    validation run.  If the score does not improve for `patience` consecutive
    validation epochs, it sends a termination signal to the trainer engine so
    training stops gracefully without wasting compute.

    The score function here returns the raw validation loss (positive value).
    ``mode='min'`` tells ``EarlyStopping`` to treat *lower* scores as better,
    so training stops when the loss stops decreasing.

    See: https://docs.pytorch.org/ignite/generated/ignite.handlers.EarlyStopping.html#ignite.handlers.EarlyStopping
    """
    early_stopping = EarlyStopping(
        patience=patience,
        score_function=lambda engine: engine.state.metrics["loss"],
        trainer=trainer_engine,
        min_delta=0.0,
        mode="min",
    )
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopping)


def setup_training_checkpointing(
    engine: Engine,
    to_save: dict[str, Any],
    save_handler: DiskSaver | ClearMLSaver,
) -> Checkpoint:
    """Save the latest model and optimizer state at the end of every epoch.

    Keeps only the single most recent checkpoint (``n_saved=1``) so disk usage
    stays bounded.  The checkpoint can be used to resume training after an
    interruption.

    The filename pattern is ``last.pt`` (prefix=``last``, no score suffix).

    See: https://docs.pytorch.org/ignite/generated/ignite.handlers.Checkpoint.html#checkpoint
    """
    last_checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=save_handler,
        filename_prefix="last",
        n_saved=1,
        global_step_transform=lambda *_: engine.state.iteration,
        filename_pattern="{filename_prefix}.{ext}",
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, last_checkpoint)
    return last_checkpoint


def setup_best_model_checkpoint(
    engine: Engine,
    val_evaluator: Engine,
    to_save: dict[str, Any],
    save_handler: DiskSaver | ClearMLSaver,
    score_function: Callable,
    score_name: str,
) -> Checkpoint:
    """Save the model that achieves the best validation score during training.

    ``Checkpoint`` from PyTorch-Ignite compares the `score_function` result after
    each validation run and overwrites the file only when the score improves, so
    the saved file always contains the best weights seen so far.

    Setting ``n_saved=1`` means at most one "best" checkpoint is kept on disk.

    See: https://docs.pytorch.org/ignite/generated/ignite.handlers.Checkpoint.html#checkpoint
    """
    checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=save_handler,
        filename_prefix="best",
        score_function=score_function,
        score_name=score_name,
        n_saved=1,
        global_step_transform=lambda *_: engine.state.iteration,
        filename_pattern="{filename_prefix}.{ext}",
    )
    val_evaluator.add_event_handler(Events.COMPLETED, checkpoint)
    return checkpoint


def setup_console_logger(
    run_dir: Path,
    engine: Engine,
    log_every_steps: int,
    optimizer: Any,
) -> logging.Logger:
    """Attach a console (and file) logger to the training engine.

    Uses ``FBResearchLogger`` from PyTorch-Ignite which prints training progress
    in a compact, human-readable format every `log_every_steps` iterations, similar
    to the style used in Facebook Research's training scripts.

    A ``logging.Logger`` writing to both stdout and ``<run_dir>/output.log`` is
    returned so other parts of the code can reuse the same logger instance.

    See: https://docs.pytorch.org/ignite/generated/ignite.handlers.fbresearch_logger.html#module-ignite.handlers.fbresearch_logger
    """
    logger = setup_logger(
        "trainer",
        level=logging.INFO,
        filepath=str(run_dir / "output.log") if run_dir else None,
        reset=True,
    )
    train_fb_logger = FBResearchLogger(logger=logger, show_output=True)
    train_fb_logger.attach(
        engine,
        name="Train",
        every=log_every_steps,
        optimizer=optimizer,
        output_transform=lambda output: {"loss": output["loss"].item()},
    )
    return logger
