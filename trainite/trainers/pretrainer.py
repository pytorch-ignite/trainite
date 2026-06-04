import logging
from datetime import datetime
from pathlib import Path

import torch
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
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.utils import setup_logger
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from trainite.config import (
    DataConfig,
    DataLoaderConfig,
    ProjectConfig,
    SplitConfig,
    dump_config,
)
from trainite.utils import get_target, instantiate


class PreTrainer:
    def __init__(self, config: ProjectConfig) -> None:
        self.logger: logging.Logger = setup_logger("trainer", level=logging.INFO)
        torch.manual_seed(config.seed)
        self.config: ProjectConfig = config

        self.train_loader: DataLoader
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.device: str | torch.device = self._resolve_device()
        self.epochs: int = config.trainer.epochs
        self.log_every_steps: int = config.trainer.log_every_steps
        self.grad_clip_norm: float | None = getattr(
            config.trainer, "grad_clip_norm", None
        )
        self.set_loaders()
        self.vocab_size: int = self._resolve_vocab_size()
        self.model: nn.Module = self._build_model()
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Optimizer = self._build_optimizer()
        self.lr: float = config.optimizer.lr
        self.total_iters: int = len(self.train_loader) * self.epochs
        self.run_dir: Path | None = None
        self.handlers: dict = {}
        self.engine = Engine(self._train_step)
        self.train_evaluator = Engine(self._eval_step)
        self.val_evaluator = Engine(self._eval_step)
        self.test_evaluator = Engine(self._eval_step)
        self.metrics = {}
        self._attach_metrics()

    def _resolve_device(self) -> str | torch.device:
        resolved = self.config.device
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        return resolved

    def _resolve_vocab_size(self) -> int:
        train_dataset = (
            self.train_loader.dataset.dataset
            if isinstance(self.train_loader.dataset, Subset)
            else self.train_loader.dataset
        )
        vocab_size = getattr(train_dataset, "vocab_size", 0)
        model_params = self.config.model.model_dump(by_alias=True)
        configured_vocab_size: int | None = model_params.get("vocab_size")
        if configured_vocab_size is not None:
            if configured_vocab_size < vocab_size:
                raise ValueError(
                    f"Configured model vocab_size ({configured_vocab_size}) is smaller than "
                    f"the dataset vocabulary size ({vocab_size}). "
                    f"Please increase model vocab_size or remove it from config.yaml "
                    f"to let it resolve automatically."
                )
            vocab_size = configured_vocab_size
        return vocab_size

    def _build_model(self) -> nn.Module:
        model = instantiate(self.config.model, vocab_size=self.vocab_size)
        model.to(self.device)
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return instantiate(self.config.optimizer, params=self.model.parameters())

    def _build_dataloader(self, split_config: SplitConfig) -> DataLoader:
        dataset = instantiate(split_config.dataset)
        return self._create_dataloader(dataset, split_config.dataloader)

    def _create_dataloader(
        self,
        dataset: Dataset,
        dl_config: DataLoaderConfig,
        shuffle: bool | None = None,
    ) -> DataLoader:
        dl_kwargs = dl_config.model_dump(exclude={"collate_fn", "shuffle"})
        if shuffle is None:
            shuffle = getattr(dl_config, "shuffle", False)

        collate_fn = None
        if dl_config.collate_fn:
            collate_fn = get_target(dl_config.collate_fn.target)

        return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **dl_kwargs)

    def _build_loaders_from_ratios(
        self, data_config: DataConfig
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        dataset = instantiate(data_config.dataset)
        total_len = len(dataset)

        if total_len == 0:
            raise ValueError(
                "Training dataset is empty. Cannot perform train/val/test split."
            )

        train_ratio = (
            data_config.train_ratio if data_config.train_ratio is not None else 1.0
        )
        val_ratio = data_config.val_ratio if data_config.val_ratio is not None else 0.0

        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        test_len = total_len - train_len - val_len

        train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        dl_config = data_config.dataloader or DataLoaderConfig()

        train_loader = self._create_dataloader(train_ds, dl_config, shuffle=True)
        val_loader = (
            self._create_dataloader(val_ds, dl_config, shuffle=False)
            if val_len > 0
            else None
        )
        test_loader = (
            self._create_dataloader(test_ds, dl_config, shuffle=False)
            if test_len > 0
            else None
        )

        return train_loader, val_loader, test_loader

    def set_loaders(self) -> None:
        if self.config.data.train:
            self.train_loader = self._build_dataloader(self.config.data.train)
            if self.config.data.val:
                self.val_loader = self._build_dataloader(self.config.data.val)
            if self.config.data.test:
                self.test_loader = self._build_dataloader(self.config.data.test)
        elif self.config.data.dataset:
            self.logger.info(
                "Automatic splitting requested with ratios: train=%s, val=%s",
                self.config.data.train_ratio,
                self.config.data.val_ratio,
            )
            (
                self.train_loader,
                self.val_loader,
                self.test_loader,
            ) = self._build_loaders_from_ratios(self.config.data)

        if self.val_loader is None:
            self.logger.warning(
                "Validation config not provided. Early stopping and best model checkpointing will be disabled. "
                "Only the last model will be saved."
            )

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
        self.train_fb_logger: FBResearchLogger = FBResearchLogger(
            logger=self.logger, show_output=True
        )
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
        self.scheduler: ParamScheduler = create_lr_scheduler_with_warmup(
            linear_decay,
            warmup_start_value=0.0,
            warmup_end_value=self.lr,
            warmup_duration=warmup_iters,
        )

        self.engine.add_event_handler(Events.ITERATION_COMPLETED, self.scheduler)

        # 2. Run evaluations
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, self._run_evaluations)

        # 3. Checkpoint
        to_save = {"model": self.model, "optimizer": self.optimizer}

        if self.val_loader:

            def score_function(engine):
                val_acc = engine.state.metrics["exact_accuracy"]
                return val_acc

            checkpoint = Checkpoint(
                to_save=to_save,
                save_handler=DiskSaver(dirname=str(self.run_dir), require_empty=False),
                filename_prefix="best",
                score_function=score_function,
                score_name="val_acc",
                n_saved=1,
                global_step_transform=lambda *_: self.engine.state.epoch,
            )
            self.val_evaluator.add_event_handler(Events.COMPLETED, checkpoint)
            self.handlers["checkpoint_best"] = checkpoint

        last_checkpoint = Checkpoint(
            to_save=to_save,
            save_handler=DiskSaver(dirname=str(self.run_dir), require_empty=False),
            filename_prefix="last",
            n_saved=1,
            global_step_transform=lambda *_: self.engine.state.epoch,
        )
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, last_checkpoint)

        self.handlers["checkpoint_last"] = last_checkpoint

        # 4. EarlyStopping
        patience = self.config.trainer.early_stopping_patience
        if self.val_loader and patience is not None:
            early_stopping = EarlyStopping(
                patience=patience,
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
