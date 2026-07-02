import logging
from datetime import datetime
from pathlib import Path
from typing import Any

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
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR

from trainite.config.base import OutputConfig


def make_run_dir(output_config: OutputConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_config.root) / output_config.run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def attach_console_logger(engine: Engine, optimizer: Optimizer, run_dir: Path, every: int) -> logging.Logger:
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
        every=every,
        optimizer=optimizer,
        output_transform=lambda output: {"loss": output["loss"].item()},
    )
    return logger


def attach_lr_scheduler(engine: Engine, optimizer: Optimizer, total_iters: int, peak_lr: float) -> ParamScheduler:
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
        warmup_end_value=peak_lr,
        warmup_duration=warmup_iters,
    )
    engine.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    return scheduler


def attach_early_stopping(val_evaluator: Engine, trainer_engine: Engine, patience: int | None) -> None:
    if patience is None:
        return
    early_stopping = EarlyStopping(
        patience=patience,
        score_function=lambda engine: engine.state.metrics["loss"],
        trainer=trainer_engine,
        min_delta=0.0,
        mode="min",
    )
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopping)


def setup_checkpointing(
    engine: Engine, val_evaluator: Engine, to_save: dict[str, Any], run_dir: Path
) -> dict[str, Checkpoint]:
    checkpointers = {}

    checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=DiskSaver(dirname=str(run_dir), require_empty=False),
        filename_prefix="best",
        score_function=lambda engine: -engine.state.metrics["loss"],
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


def setup_tensorboard(
    engine: Engine,
    train_evaluator: Engine,
    val_evaluator: Engine,
    test_evaluator: Engine,
    optimizer: Optimizer,
    run_dir: Path,
    metric_names: list[str],
    has_test: bool,
) -> TensorboardLogger:
    tb_logger = TensorboardLogger(log_dir=run_dir / "tensorboard" if run_dir else None)

    # Log training iteration loss
    tb_logger.attach_output_handler(
        engine,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda output: {"batch_loss": output["loss"]},
    )

    # Log training epoch metrics
    tb_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=metric_names,
        global_step_transform=lambda *_: engine.state.epoch,
    )

    # Log validation epoch metrics
    tb_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=metric_names,
        global_step_transform=lambda *_: engine.state.epoch,
    )

    # Log test metrics if applicable
    if has_test:
        tb_logger.attach_output_handler(
            test_evaluator,
            event_name=Events.COMPLETED,
            tag="testing",
            metric_names=metric_names,
            global_step_transform=lambda *_: engine.state.epoch,
        )

    # Log optimizer learning rates
    tb_logger.attach(
        engine,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_STARTED,
    )
    return tb_logger
