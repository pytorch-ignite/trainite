from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import torch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers.fbresearch_logger import FBResearchLogger
from ignite.utils import setup_logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from trainite.config import ProjectConfig, dump_config
from trainite.datasets import build_string_reverse_dataloaders
from trainite.models import build_transformer_model

class PreTrainer:
    def __init__(
        self,
        config: ProjectConfig,
        model: nn.Module | None = None,
        train_loader=None,
        val_loader=None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.trainer.device)
        torch.manual_seed(config.seed)

        self.model = model or build_transformer_model(config.model)
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.trainer.learning_rate
        )

        if train_loader is None or val_loader is None:
            train_loader, val_loader = build_string_reverse_dataloaders(config.dataset)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.run_dir = self._make_run_dir()
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        dump_config(config, self.run_dir / "config.yaml")

        self.best_score = float("-inf")
        self.best_path = self.run_dir / "best.pt"
        self.last_path = self.run_dir / "last.pt"

        self.engine = Engine(self._train_step)
        self.train_evaluator = Engine(self._eval_step)
        self.val_evaluator = Engine(self._eval_step)
        self.metrics = {}

        self._attach_metrics()
        self._attach_handlers()

        self.logger = setup_logger("trainer", level=logging.INFO)
        self._setup_file_logging()
        self.train_fb_logger = FBResearchLogger(logger=self.logger, show_output=True)
        self.train_fb_logger.attach(self.engine, name="Train", every=self.config.trainer.log_every_steps, optimizer=self.optimizer, output_transform=lambda output: {"loss": output["loss"].item()})


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
        mask = targets != 0
        return logits[mask], targets[mask]

    def _flatten_loss(
        self, output: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = output["logits"].reshape(-1, output["logits"].size(-1))
        targets = output["targets"].reshape(-1)
        mask = targets != 0
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

        if self.config.trainer.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.trainer.grad_clip_norm
            )

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
        self.engine.add_event_handler(
            Events.ITERATION_COMPLETED(every=self.config.trainer.log_every_steps),
            self._log_training,
        )
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, self._run_evaluations)
        self.engine.add_event_handler(
            Events.EPOCH_COMPLETED, self._save_last_checkpoint
        )
        self.val_evaluator.add_event_handler(
            Events.COMPLETED, self._update_best_checkpoint
        )

    def _log_training(self, engine: Engine) -> None:
        metrics = engine.state.metrics
        epoch = engine.state.epoch
        iteration = engine.state.iteration
        loss = float(metrics["loss"])
        self.writer.add_scalar("train/loss", loss, iteration)

    def _run_evaluations(self, engine: Engine) -> None:
        self.logger.info("Evaluating on training set...")
        self.train_evaluator.run(self.train_loader)
        
        self.logger.info("Evaluating on validation set...")
        self.val_evaluator.run(self.val_loader)

        train_metrics = self.train_evaluator.state.metrics
        val_metrics = self.val_evaluator.state.metrics
        epoch = engine.state.epoch

        self.writer.add_scalar("eval/train_loss", train_metrics["loss"], epoch)
        self.writer.add_scalar(
            "eval/train_token_accuracy", train_metrics["token_accuracy"], epoch
        )
        self.writer.add_scalar("eval/val_loss", val_metrics["loss"], epoch)
        self.writer.add_scalar(
            "eval/val_token_accuracy", val_metrics["token_accuracy"], epoch
        )

        self.logger.info(
            "epoch=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["token_accuracy"],
            val_metrics["loss"],
            val_metrics["token_accuracy"],
        )

    def _save_state(self, path: Path, epoch: int, score: float | None = None) -> None:
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config.model_dump(),
            "best_score": self.best_score,
        }
        if score is not None:
            payload["score"] = score
        torch.save(payload, path)

    def _save_last_checkpoint(self, engine: Engine) -> None:
        self._save_state(self.last_path, engine.state.epoch)

    def _update_best_checkpoint(self, engine: Engine) -> None:
        score = float(engine.state.metrics["token_accuracy"])
        if score > self.best_score:
            self.best_score = score
            self._save_state(self.best_path, engine.state.epoch, score)
            self.logger.info("saved best checkpoint score=%.4f", score)

    def _setup_file_logging(self) -> None:
        handler = logging.FileHandler(self.run_dir / "output.log")

        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        self.logger.addHandler(handler)


    def run(self) -> None:
        self.logger.info("starting run in %s", self.run_dir)
        self.writer.add_text("config", str(self.config.model_dump()))
        self.engine.run(self.train_loader, max_epochs=self.config.trainer.epochs)
        self.writer.flush()
        self.writer.close()
