import importlib
import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset, random_split

from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    create_lr_scheduler_with_warmup,
)
from ignite.handlers.fbresearch_logger import FBResearchLogger
from ignite.handlers.param_scheduler import ParamScheduler
from ignite.handlers.tensorboard_logger import OptimizerParamsHandler, TensorboardLogger
from ignite.utils import setup_logger
from torch.optim.lr_scheduler import LinearLR

from trainite.config.base import DataConfigBase, DataWithAutoSplit
from trainite.datasets.transformed import TransformedDataset

T = TypeVar("T", bound=BaseModel)


def get_target(target_path: str) -> Any:
    """
    Gets a class or a function defined by a `_target_` key
    using an OmegaConf DictConfig.
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


def instantiate(config: BaseModel, **kwargs) -> Any:
    """
    Instantiates a class or calls a function defined by a `_target_` key
    using an OmegaConf DictConfig.
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


def _inject_if_accepted(target_symbol: Any, **candidates: Any) -> dict[str, Any]:
    """Inspects the target symbol's signature and filters the candidates to

    only include those that are accepted by the target symbol.
    """
    try:
        sig = inspect.signature(target_symbol)
        return {k: v for k, v in candidates.items() if k in sig.parameters}
    except Exception:
        return {}


def build_dataset(dataset_config: Any, transform_config: Any, tokenizer: Any) -> Dataset:
    """Builds a Dataset from dataset_config and optional transform_config."""
    ds = get_target(dataset_config.target)
    dataset = instantiate(dataset_config, **_inject_if_accepted(ds, preprocessor=tokenizer, tokenizer=tokenizer))
    if transform_config is not None:
        tf = get_target(transform_config.target)
        transform = instantiate(
            transform_config, **_inject_if_accepted(tf, preprocessor=tokenizer, tokenizer=tokenizer)
        )
        return TransformedDataset(dataset, transform)
    return dataset


def create_dataloader(
    dataset: Dataset,
    dl_config: Any,
    tokenizer: Any,
    shuffle: bool | None = None,
) -> DataLoader:
    """Creates a PyTorch DataLoader from config and target options."""
    dl_kwargs = dl_config.model_dump(exclude={"collate_fn", "shuffle"})
    if shuffle is None:
        shuffle = getattr(dl_config, "shuffle", False)
    collate_fn = None
    collate_config = dl_config.collate_fn
    if collate_config:
        target_symbol = get_target(collate_config.target)
        if isinstance(target_symbol, type):
            collate_fn = instantiate(collate_config, tokenizer=tokenizer)
        else:
            collate_fn = target_symbol
    return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **dl_kwargs)


def _loaders_from_splits(
    data_config: DataConfigBase, tokenizer: Any
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    def _make(split_config: Any) -> DataLoader:
        ds = build_dataset(split_config.dataset, split_config.transform, tokenizer)
        return create_dataloader(ds, split_config.dataloader, tokenizer)

    return _make(data_config.train), _make(data_config.val), _make(data_config.test) if data_config.test else None


def _loaders_from_ratios(
    data_config: DataWithAutoSplit, tokenizer: Any, seed: int
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
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
        create_dataloader(train_ds, dl, tokenizer, shuffle=True),
        create_dataloader(val_ds, dl, tokenizer, shuffle=False),
        create_dataloader(test_ds, dl, tokenizer, shuffle=False) if test_len > 0 else None,
    )


def build_dataloaders(data_config: Any, tokenizer: Any, seed: int) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Builds train, validation, and test dataloaders from the data configuration."""
    if isinstance(data_config, DataWithAutoSplit):
        return _loaders_from_ratios(data_config, tokenizer, seed)
    return _loaders_from_splits(data_config, tokenizer)


def setup_run_dir(root_dir: str, run_name: str) -> Path:
    """Sets up a run directory with a timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root_dir) / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_console_logger(
    run_dir: Path,
    engine: Engine,
    optimizer: Any,
    log_every_steps: int,
) -> logging.Logger:
    """Configures a console and file logger with FBResearchLogger attached to training."""
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


def setup_lr_scheduler(
    optimizer: Any,
    lr: float,
    total_iters: int,
    engine: Engine,
) -> ParamScheduler:
    """Configures a warmup + linear decay learning rate scheduler and attaches it to iteration completed."""
    warmup_iters = max(2, int(0.1 * total_iters))
    linear_decay = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_iters - warmup_iters,
    )
    scheduler = create_lr_scheduler_with_warmup(
        linear_decay,
        warmup_start_value=0.0,
        warmup_end_value=lr,
        warmup_duration=warmup_iters,
    )
    engine.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    return scheduler


def setup_early_stopping(
    engine: Engine,
    val_evaluator: Engine,
    patience: int | None,
) -> None:
    """Configures early stopping handler attached to validation completion."""
    if patience is not None:
        early_stopping = EarlyStopping(
            patience=patience,
            score_function=lambda eng: eng.state.metrics["loss"],
            trainer=engine,
            min_delta=0.0,
            mode="min",
        )
        val_evaluator.add_event_handler(Events.COMPLETED, early_stopping)


def setup_checkpointing(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: Any,
    engine: Engine,
    val_evaluator: Engine,
) -> dict[str, Checkpoint]:
    """Configures model/optimizer checkpointing (best validation and last epoch)."""
    to_save = {"model": model, "optimizer": optimizer}
    checkpointers = {}

    def score_function(eng):
        loss = eng.state.metrics["loss"]
        return -loss

    checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=DiskSaver(dirname=str(run_dir), require_empty=False),
        filename_prefix="best",
        score_function=score_function,
        score_name="val_loss",
        n_saved=1,
        global_step_transform=lambda *_: engine.state.epoch,
    )
    val_evaluator.add_event_handler(Events.COMPLETED, checkpoint)
    checkpointers["checkpoint_best"] = checkpoint

    last_checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=DiskSaver(dirname=str(run_dir), require_empty=False),
        filename_prefix="last",
        n_saved=1,
        global_step_transform=lambda *_: engine.state.epoch,
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, last_checkpoint)
    checkpointers["checkpoint_last"] = last_checkpoint

    return checkpointers


def setup_metric_logger(
    run_dir: Path,
    engine: Engine,
    train_evaluator: Engine,
    val_evaluator: Engine,
    test_evaluator: Engine | None,
    optimizer: Any,
    metric_names: list[str],
) -> TensorboardLogger:
    """Configures a TensorBoard logger and attaches outputs handlers."""
    log_dir = run_dir / "tensorboard" if run_dir else None
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Log training iteration loss
    tb_logger.attach_output_handler(
        engine,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda output: {"batch_loss": output["loss"]},
        global_step_transform=lambda *_: engine.state.iteration,
    )

    # Log training epoch metrics
    tb_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=metric_names,
        global_step_transform=lambda *_: engine.state.iteration,
    )

    # Log validation epoch metrics
    tb_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=metric_names,
        global_step_transform=lambda *_: engine.state.iteration,
    )

    # Log test metrics if applicable
    if test_evaluator:
        tb_logger.attach_output_handler(
            test_evaluator,
            event_name=Events.COMPLETED,
            tag="testing",
            metric_names=metric_names,
            global_step_transform=lambda *_: engine.state.iteration,
        )

    # Log optimizer learning rates
    tb_logger.attach(
        engine,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_STARTED,
    )
    return tb_logger
