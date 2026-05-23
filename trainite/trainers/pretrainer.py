import logging
from datetime import datetime
from pathlib import Path

import torch
from ignite.engine import Engine, Events
from ignite.handlers import (
    EarlyStopping,
    ModelCheckpoint,
    create_lr_scheduler_with_warmup,
)
from ignite.handlers.fbresearch_logger import FBResearchLogger
from ignite.handlers.tensorboard_logger import OptimizerParamsHandler, TensorboardLogger
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.utils import setup_logger
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from trainite.config import ProjectConfig, SplitConfig, dump_config
from trainite.utils import get_target, instantiate


class PreTrainer:
    def __init__(
        self,
        config: ProjectConfig,
        device: str | torch.device | None = None,
        lr: float | None = None,
        epochs: int | None = None,
        log_every_steps: int | None = None,
        grad_clip_norm: float | None = None,
        model: nn.Module | None = None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        **kwargs,
    ) -> None:
        self.logger = setup_logger("trainer", level=logging.INFO)
        torch.manual_seed(config.seed)

        self.config = config
        resolved_device = config.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device or resolved_device
        self.epochs = epochs or config.trainer.epochs
        self.log_every_steps = log_every_steps or config.trainer.log_every_steps
        self.grad_clip_norm = grad_clip_norm or config.trainer.grad_clip_norm

        if train_loader is None:
            train_loader = self._build_dataloader(config.data.train)

        if val_loader is None:
            if config.data.val:
                val_loader = self._build_dataloader(config.data.val)
            else:
                self.logger.warning(
                    "Validation config not provided. Early stopping and best model checkpointing will be disabled. "
                    "Only the last model will be saved."
                )

        if test_loader is None and config.data.test:
            test_loader = self._build_dataloader(config.data.test)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.vocab_size = getattr(train_loader.dataset, "vocab_size", None)
        model_params = config.model.model_dump(by_alias=True)
        configured_vocab_size = model_params.get("vocab_size")

        if configured_vocab_size is not None:
            if self.vocab_size is not None and configured_vocab_size < self.vocab_size:
                raise ValueError(
                    f"Configured model vocab_size ({configured_vocab_size}) is smaller than "
                    f"the dataset vocabulary size ({self.vocab_size}). "
                    f"Please increase model vocab_size or remove it from config.yaml "
                    f"to let it resolve automatically."
                )
            self.vocab_size = configured_vocab_size

        self.model = model or instantiate(config.model, vocab_size=self.vocab_size)
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = instantiate(config.optimizer, params=self.model.parameters())
        self.lr = lr or config.optimizer.lr
        self.total_iters = len(self.train_loader) * self.epochs
        self.run_dir: Path | None = None
        self.handlers: dict = {}
        self.engine = Engine(self._train_step)
        self.train_evaluator = Engine(self._eval_step)
        self.val_evaluator = Engine(self._eval_step)
        self.test_evaluator = Engine(self._eval_step)
        self.metrics = {}
        self._attach_metrics()

    def _build_dataloader(self, split_config: SplitConfig) -> DataLoader:
        dataset = instantiate(split_config.dataset)
        dl_kwargs = split_config.dataloader.model_dump(exclude={"collate_fn"})
        collate_fn = None
        if split_config.dataloader.collate_fn:
            collate_fn = get_target(split_config.dataloader.collate_fn.target)

        return DataLoader(dataset, collate_fn=collate_fn, **dl_kwargs)

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = self.config.output.run_name
        run_dir = Path(self.config.output.root) / run_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _exact_accuracy_transform(
        self, output: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = output["logits"]
        targets = output["targets"]

        preds = torch.argmax(logits, dim=-1)
        mask = targets != self.loss_fn.ignore_index

        correct = (preds == targets) | ~mask
        seq_correct = correct.all(dim=-1)

        y_pred = seq_correct.long()
        y = torch.ones_like(seq_correct)

        return y_pred, y

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
        train_accuracy = Accuracy(output_transform=self._exact_accuracy_transform)
        val_loss = Loss(self.loss_fn, output_transform=self._flatten_loss)
        val_accuracy = Accuracy(output_transform=self._exact_accuracy_transform)
        test_loss = Loss(self.loss_fn, output_transform=self._flatten_loss)
        test_accuracy = Accuracy(output_transform=self._exact_accuracy_transform)

        train_loss.attach(self.train_evaluator, "loss")
        train_accuracy.attach(self.train_evaluator, "exact_accuracy")
        val_loss.attach(self.val_evaluator, "loss")
        val_accuracy.attach(self.val_evaluator, "exact_accuracy")
        test_loss.attach(self.test_evaluator, "loss")
        test_accuracy.attach(self.test_evaluator, "exact_accuracy")

        self.metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

    def _attach_handlers(self) -> None:
        self.logger = setup_logger(
            "trainer",
            level=logging.INFO,
            filepath=str(self.run_dir / "output.log") if self.run_dir else None,
            reset=True,
        )
        self.train_fb_logger = FBResearchLogger(logger=self.logger, show_output=True)
        self.train_fb_logger.attach(
            self.engine,
            name="Train",
            every=self.log_every_steps,
            optimizer=self.optimizer,
            output_transform=lambda output: {"loss": output["loss"].item()},
        )

        # 1. Step LR scheduler every iteration
        warmup_iters = max(2, int(0.1 * self.total_iters))
        linear_decay = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.total_iters - warmup_iters,
        )
        self.scheduler = create_lr_scheduler_with_warmup(
            linear_decay,
            warmup_start_value=0.0,
            warmup_end_value=self.lr,
            warmup_duration=warmup_iters,
        )

        self.engine.add_event_handler(Events.ITERATION_COMPLETED, self.scheduler)

        # 2. Run evaluations
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, self._run_evaluations)

        # 3. ModelCheckpoint
        to_save = {"model": self.model, "optimizer": self.optimizer}

        if self.val_loader:

            def score_function(engine):
                val_acc = engine.state.metrics["exact_accuracy"]
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
            self.handlers["checkpoint_best"] = checkpoint

        last_checkpoint = ModelCheckpoint(
            dirname=str(self.run_dir),
            n_saved=1,
            filename_prefix="last",
            require_empty=False,
            global_step_transform=lambda *_: self.engine.state.epoch,
        )
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, last_checkpoint, to_save)

        self.handlers["checkpoint_last"] = last_checkpoint

        # 4. EarlyStopping
        if self.val_loader:
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

        metric_names = ["loss", "exact_accuracy"]
        tb_logger.attach_output_handler(
            self.train_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="training",
            metric_names=metric_names,
            global_step_transform=lambda *_: self.engine.state.epoch,
        )
        if self.val_loader:
            tb_logger.attach_output_handler(
                self.val_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=metric_names,
                global_step_transform=lambda *_: self.engine.state.epoch,
            )
        if self.test_loader:
            tb_logger.attach_output_handler(
                self.test_evaluator,
                event_name=Events.COMPLETED,
                tag="testing",
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
        self.logger.info("Evaluating on training set...")
        self.train_evaluator.run(self.train_loader)
        train_metrics = self.train_evaluator.state.metrics
        epoch = engine.state.epoch

        if self.val_loader:
            self.logger.info("Evaluating on validation set...")
            self.val_evaluator.run(self.val_loader)
            val_metrics = self.val_evaluator.state.metrics
            self.logger.info(
                "epoch=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["exact_accuracy"],
                val_metrics["loss"],
                val_metrics["exact_accuracy"],
            )
        else:
            self.logger.info(
                "epoch=%s train_loss=%.4f train_acc=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["exact_accuracy"],
            )

    def test(self, test_loader: DataLoader | None = None) -> None:
        loader = test_loader or self.test_loader
        if loader is None:
            self.logger.warning("No test loader provided. Skipping testing.")
            return

        # Load best model if available
        checkpoint_handler = self.handlers.get("checkpoint_best") or self.handlers.get(
            "checkpoint_last"
        )
        if checkpoint_handler and checkpoint_handler.last_checkpoint:
            checkpoint_path = checkpoint_handler.last_checkpoint

            self.logger.info("Loading best model for testing from %s", checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint["model"])

        self.logger.info("Running testing...")
        self.test_evaluator.run(loader)
        metrics = self.test_evaluator.state.metrics
        self.logger.info(
            "Test results: loss=%.4f acc=%.4f",
            metrics["loss"],
            metrics["exact_accuracy"],
        )

    def run(self) -> None:
        # create run directory and handlers when the run actually starts
        if self.run_dir is None:
            self.run_dir = self._make_run_dir()
            dump_config(self.config, self.run_dir / "config.yaml")
            self._attach_handlers()

        self.logger.info("starting run in %s", self.run_dir)
        config_data = self.config.model_dump(
            by_alias=True, polymorphic_serialization=True
        )
        if "tensorboard" in self.handlers:
            self.handlers["tensorboard"].writer.add_text("config", str(config_data))

        self.engine.run(self.train_loader, max_epochs=self.epochs)

        if self.test_loader:
            self.test()

        if "tensorboard" in self.handlers:
            self.handlers["tensorboard"].close()
