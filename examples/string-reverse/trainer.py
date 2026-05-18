import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.handlers.tensorboard_logger import OptimizerParamsHandler, TensorboardLogger
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch import nn

from config import ProjectConfig, dump_config
from utils import instantiate

logger = logging.getLogger(__name__)


class PreTrainer:
    def __init__(
        self,
        config: ProjectConfig,
        device: str | torch.device | None = None,
        learning_rate: float | None = None,
        epochs: int | None = None,
        log_every_steps: int | None = None,
        grad_clip_norm: float | None = None,
        model: nn.Module | None = None,
        train_loader=None,
        val_loader=None,
        **kwargs,
    ) -> None:
        self.config = config
        self.device = device or config.trainer.device
        self.epochs = epochs or config.trainer.epochs
        self.log_every_steps = log_every_steps or config.trainer.log_every_steps
        self.grad_clip_norm = grad_clip_norm or config.trainer.grad_clip_norm

        torch.manual_seed(config.seed)

        self.model = model or instantiate(config.model)
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate or config.trainer.learning_rate
        )

        if train_loader is None or val_loader is None:
            train_loader, val_loader = instantiate(config.dataset)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.run_dir: Optional[Path] = None
        self.handlers: dict = {}

        self.engine = Engine(self._train_step)
        self.train_evaluator = Engine(self._eval_step)
        self.val_evaluator = Engine(self._eval_step)
        self.metrics = {}

        self._attach_metrics()

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = self.config.output.run_name
        run_dir = Path(self.config.output.root) / run_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _flatten_accuracy(
        self, output: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = output["logits"].reshape(-1, output["logits"].size(-1))
        targets = output["targets"].reshape(-1)
        mask = targets != self.loss_fn.ignore_index
        return logits[mask], targets[mask]

    def _flatten_loss(
        self, output: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = output["logits"].reshape(-1, output["logits"].size(-1))
        targets = output["targets"].reshape(-1)
        mask = targets != self.loss_fn.ignore_index
        return logits[mask], targets[mask]

    def _train_step(
        self, engine: Engine, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        self.model.train()
        inputs = batch["input_ids"].to(self.device)
        targets = batch["labels"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        return {
            "loss": loss.detach(),
            "logits": logits.detach(),
            "targets": targets.detach(),
        }

    @torch.no_grad()
    def _eval_step(
        self, engine: Engine, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        self.model.eval()
        inputs = batch["input_ids"].to(self.device)
        targets = batch["labels"].to(self.device)
        logits = self.model(inputs)
        return {"logits": logits, "targets": targets}

    def _attach_metrics(self) -> None:
        RunningAverage(output_transform=lambda output: output["loss"]).attach(
            self.engine, "loss"
        )

        train_loss = Loss(self.loss_fn, output_transform=self._flatten_loss)
        train_accuracy = Accuracy(output_transform=self._flatten_accuracy)
        val_loss = Loss(self.loss_fn, output_transform=self._flatten_loss)
        val_accuracy = Accuracy(output_transform=self._flatten_accuracy)

        train_loss.attach(self.train_evaluator, "loss")
        train_accuracy.attach(self.train_evaluator, "token_accuracy")
        val_loss.attach(self.val_evaluator, "loss")
        val_accuracy.attach(self.val_evaluator, "token_accuracy")

        self.metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

    def _attach_handlers(self) -> None:
        # 1. Log training loss every N steps
        def _log_loss(engine):
            logger.info(
                f"epoch={engine.state.epoch} iteration={engine.state.iteration} "
                f"train_loss={engine.state.output['loss']:.4f}"
            )

        self.engine.add_event_handler(
            Events.ITERATION_COMPLETED(every=self.log_every_steps),
            _log_loss,
        )

        # 2. Run evaluations
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, self._run_evaluations)

        # 3. ModelCheckpoint
        to_save = {"model": self.model, "optimizer": self.optimizer}

        def score_function(engine):
            val_acc = engine.state.metrics["token_accuracy"]
            return val_acc

        checkpoint = ModelCheckpoint(
            dirname=str(self.run_dir),
            n_saved=1,
            filename_prefix="best",
            score_function=score_function,
            score_name="val_acc",
            require_empty=False,
            global_step_transform=lambda *_: self.engine.state.epoch,
        )
        self.val_evaluator.add_event_handler(Events.COMPLETED, checkpoint, to_save)

        last_checkpoint = ModelCheckpoint(
            dirname=str(self.run_dir),
            n_saved=1,
            filename_prefix="last",
            require_empty=False,
            global_step_transform=lambda *_: self.engine.state.epoch,
        )
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, last_checkpoint, to_save)

        self.handlers["checkpoint_best"] = checkpoint
        self.handlers["checkpoint_last"] = last_checkpoint

        # 4. EarlyStopping
        early_stopping = EarlyStopping(
            patience=3,
            score_function=lambda engine: -engine.state.metrics["loss"],
            trainer=self.engine,
            min_delta=0.0,
        )
        self.val_evaluator.add_event_handler(Events.COMPLETED, early_stopping)
        self.handlers["early_stopping"] = early_stopping

        # 5. TensorboardLogger
        log_dir = self.run_dir / "tensorboard" if self.run_dir else None
        tb_logger = TensorboardLogger(log_dir=log_dir)
        tb_logger.attach_output_handler(
            self.engine,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda output: {"batch_loss": output["loss"]},
        )

        metric_names = ["loss", "token_accuracy"]
        tb_logger.attach_output_handler(
            self.train_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="training",
            metric_names=metric_names,
            global_step_transform=lambda *_: self.engine.state.epoch,
        )
        tb_logger.attach_output_handler(
            self.val_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=metric_names,
            global_step_transform=lambda *_: self.engine.state.epoch,
        )

        tb_logger.attach(
            self.engine,
            log_handler=OptimizerParamsHandler(self.optimizer),
            event_name=Events.ITERATION_STARTED,
        )
        self.handlers["tensorboard"] = tb_logger

    def _run_evaluations(self, engine: Engine) -> None:
        logger.info("Evaluating on training set...")
        self.train_evaluator.run(self.train_loader)

        logger.info("Evaluating on validation set...")
        self.val_evaluator.run(self.val_loader)

        train_metrics = self.train_evaluator.state.metrics
        val_metrics = self.val_evaluator.state.metrics
        epoch = engine.state.epoch

        logger.info(
            "epoch=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["token_accuracy"],
            val_metrics["loss"],
            val_metrics["token_accuracy"],
        )

    def run(self) -> None:
        # create run directory and handlers when the run actually starts
        if self.run_dir is None:
            self.run_dir = self._make_run_dir()
            dump_config(self.config, self.run_dir / "config.yaml")
            self._attach_handlers()

        logger.info("starting run in %s", self.run_dir)
        config_data = self.config.model_dump(
            by_alias=True, polymorphic_serialization=True
        )
        if "tensorboard" in self.handlers:
            self.handlers["tensorboard"].writer.add_text("config", str(config_data))

        self.engine.run(self.train_loader, max_epochs=self.epochs)

        if "tensorboard" in self.handlers:
            self.handlers["tensorboard"].close()
