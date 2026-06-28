import logging
from datetime import datetime
from pathlib import Path
from typing import Any

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
from ignite.metrics import Accuracy, Loss, Metric, RunningAverage
from ignite.utils import setup_logger
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Subset

from config import (
    OptimizerConfig,
    OutputConfig,
)

from models.transformer import TransformerModelConfig
from datasets.string_reverse import StringReverseDataConfig
from preprocessors.char_tokenizer import CharTokenizerConfig
from utils import dump_config


class DecoderTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    epochs: int = Field(default=3, gt=0)
    log_every_steps: int = Field(default=10, gt=0)
    early_stopping_patience: int | None = Field(default=3, gt=0)
    # Inference logging
    inference_every_epochs: int | None = Field(default=None, gt=0)
    inference_num_samples: int = Field(default=5, gt=0)
    max_inference_new_tokens: int = Field(default=16, gt=0)
    grad_clip_norm: float | None = Field(default=None, gt=0.0)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    preprocessor: CharTokenizerConfig | None = None
    model: TransformerModelConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: StringReverseDataConfig
    trainer: DecoderTrainerConfig = Field(default_factory=DecoderTrainerConfig)
    output: OutputConfig
    seed: int = 42
    device: str = "auto"


class DecoderTrainer:
    def __init__(
        self,
        config: ProjectConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        preprocessor: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        prompt_transform: Any = None,
    ) -> None:
        self.logger: logging.Logger = setup_logger("trainer", level=logging.INFO)
        torch.manual_seed(config.seed)
        self.config: ProjectConfig = config
        self.tokenizer = preprocessor
        self.prompt_transform = prompt_transform
        self.device: str | torch.device = self._resolve_device()
        self.epochs: int = config.trainer.epochs
        self.grad_clip_norm: float | None = getattr(config.trainer, "grad_clip_norm", None)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.model.to(self.device)

        self.inference_every_epochs = config.trainer.inference_every_epochs
        self.inference_num_samples = config.trainer.inference_num_samples
        self.max_inference_new_tokens = config.trainer.max_inference_new_tokens
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.total_iters: int = len(self.train_loader) * self.epochs
        self.run_dir: Path | None = None
        self.handlers: dict = {}
        self.engine = Engine(self._train_step)
        self.train_evaluator = Engine(self._eval_step)
        self.val_evaluator = Engine(self._eval_step)
        self.test_evaluator = Engine(self._eval_step)
        self.metrics: dict = self._attach_metrics()

        if self.inference_every_epochs is not None:
            self._setup_inference()

    def _resolve_device(self) -> str | torch.device:
        resolved = self.config.device
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        return resolved

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = self.config.output.run_name
        run_dir = Path(self.config.output.root) / run_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _flatten(self, output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits = output["logits"].reshape(-1, output["logits"].size(-1))
        targets = output["targets"].reshape(-1)
        mask = targets != self.loss_fn.ignore_index
        return logits[mask], targets[mask]

    def _train_step(self, engine: Engine, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.model.train()
        inputs = batch["input_ids"].to(self.device)
        targets = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(inputs, attention_mask=attention_mask)
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
    def _eval_step(self, engine: Engine, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.model.eval()
        inputs = batch["input_ids"].to(self.device)
        targets = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        logits = self.model(inputs, attention_mask=attention_mask)
        return {"logits": logits, "targets": targets}

    def _attach_metrics(self) -> dict[str, Metric]:
        RunningAverage(output_transform=lambda output: output["loss"]).attach(self.engine, "loss")

        metrics = {}
        for prefix, evaluator in [
            ("train", self.train_evaluator),
            ("val", self.val_evaluator),
            ("test", self.test_evaluator),
        ]:
            loss = Loss(self.loss_fn, output_transform=self._flatten)
            token_acc = Accuracy(output_transform=self._flatten)

            loss.attach(evaluator, "loss")
            token_acc.attach(evaluator, "token_accuracy")

            metrics[f"{prefix}_loss"] = loss
            metrics[f"{prefix}_token_accuracy"] = token_acc

        return metrics

    def _attach_handlers(self) -> None:
        self.logger = setup_logger(
            "trainer",
            level=logging.INFO,
            filepath=str(self.run_dir / "output.log") if self.run_dir else None,
            reset=True,
        )
        self.train_fb_logger: FBResearchLogger = FBResearchLogger(logger=self.logger, show_output=True)
        self.train_fb_logger.attach(
            self.engine,
            name="Train",
            every=self.config.trainer.log_every_steps,
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
            warmup_end_value=self.config.optimizer.lr,
            warmup_duration=warmup_iters,
        )

        self.engine.add_event_handler(Events.ITERATION_COMPLETED, self.scheduler)

        # 2. Run evaluations
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, self._run_evaluations)
        if self.inference_every_epochs is not None:
            self.engine.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.inference_every_epochs),
                self._log_inference,
                self.train_loader,
                "Train",
            )
            if self.val_loader is not None:
                self.engine.add_event_handler(
                    Events.EPOCH_COMPLETED(every=self.inference_every_epochs),
                    self._log_inference,
                    self.val_loader,
                    "Val",
                )
            if self.test_loader is not None:
                self.engine.add_event_handler(
                    Events.EPOCH_COMPLETED(every=self.inference_every_epochs),
                    self._log_inference,
                    self.test_loader,
                    "Test",
                )

        # 3. Checkpoint
        to_save = {"model": self.model, "optimizer": self.optimizer}

        if self.val_loader:

            def score_function(engine):
                loss = engine.state.metrics["loss"]
                return -loss

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

        metric_names = ["loss", "token_accuracy"]
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
                "epoch=%s train_loss=%.4f train_token_acc=%.4f val_loss=%.4f val_token_acc=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["token_accuracy"],
                val_metrics["loss"],
                val_metrics["token_accuracy"],
            )
        else:
            self.logger.info(
                "epoch=%s train_loss=%.4f train_token_acc=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["token_accuracy"],
            )

    def _validate_dataloader_for_inference(self, loader: DataLoader, name: str) -> None:
        dataset = loader.dataset
        if len(dataset) == 0:
            raise ValueError(f"{name} dataset is empty. Cannot perform inference logging.")
        dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        item = dataset[0]
        if not isinstance(item, dict):
            raise ValueError(f"{name} dataset items must be dicts with 'source' and 'target' keys.")
        if "source" not in item or "target" not in item:
            raise ValueError(f"{name} dataset items must contain 'source' and 'target' keys. Got: {list(item.keys())}")

    def _setup_inference(self) -> None:
        tokenizer: Any = self.tokenizer
        if not hasattr(tokenizer, "decode") or not callable(tokenizer):
            raise ValueError("Tokenizer must be callable and implement 'decode' method for inference logging.")

        # Validate active loaders
        self._validate_dataloader_for_inference(self.train_loader, "Train")
        if self.val_loader is not None:
            self._validate_dataloader_for_inference(self.val_loader, "Val")
        if self.test_loader is not None:
            self._validate_dataloader_for_inference(self.test_loader, "Test")

        # The transform supplies the prompt format (source -> token ids)
        if self.prompt_transform is None or not hasattr(self.prompt_transform, "build_prompt"):
            raise ValueError(
                "Inference logging requires the dataset's transform to define build_prompt(sample) -> list[int]."
            )

    def _log_inference(self, engine: Engine, loader: DataLoader, name: str) -> None:
        self.logger.info(f"Epoch {engine.state.epoch}: Running inference on {name} samples...")

        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

        dataset = loader.dataset
        total_samples = len(dataset)  # type: ignore
        num_samples = min(self.inference_num_samples, total_samples)

        # Build generation prompts via the dataset's transform (validated in _setup_inference).
        prompt_ids_list: list[torch.Tensor] = []
        prompt_attn_list: list[torch.Tensor] = []
        targets: list[str] = []
        sources: list[str] = []
        for i in range(num_samples):
            item = dataset[i]
            source = item["source"]
            target = item["target"]

            input_ids = torch.tensor(self.prompt_transform.build_prompt(item), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            prompt_ids_list.append(input_ids)
            prompt_attn_list.append(attention_mask)
            targets.append(target)
            sources.append(source)
        self.model.eval()

        # Left-pad for generation: reverse → pad_sequence → reverse
        batch_input_ids = (
            torch.nn.utils.rnn.pad_sequence(
                [t.flip(0) for t in prompt_ids_list], batch_first=True, padding_value=pad_token_id
            )
            .flip(1)
            .to(self.device)
        )
        batch_attention_mask = (
            torch.nn.utils.rnn.pad_sequence([t.flip(0) for t in prompt_attn_list], batch_first=True, padding_value=0)
            .flip(1)
            .to(self.device)
        )
        sequences = self.generate(
            input_ids=batch_input_ids,
            max_new_tokens=self.max_inference_new_tokens,
            attention_mask=batch_attention_mask,
        )
        new_tokens = sequences[:, batch_input_ids.shape[1] :]
        decoded_strs = [
            self.tokenizer.decode(
                tokens.tolist(),
                skip_special_tokens=True,
            )
            for tokens in new_tokens
        ]

        for idx in range(num_samples):
            self.logger.info(
                "    Sample %d | Prompt: %r | Target: %r | Prediction: %r",
                idx + 1,
                sources[idx],
                targets[idx],
                decoded_strs[idx],
            )

        if "tensorboard" in self.handlers:
            tb_writer = self.handlers["tensorboard"].writer
            lines = []
            for idx in range(num_samples):
                prompt = sources[idx].strip() or "(empty)"
                target = targets[idx].strip() or "(empty)"
                pred = decoded_strs[idx].strip() or "(empty)"
                prompt = prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                target = target.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                pred = pred.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                lines.append(f"Sample {idx + 1}")
                lines.append(f"  Prompt:     {prompt}")
                lines.append(f"  Target:     {target}")
                lines.append(f"  Prediction: {pred}")
                lines.append("")
            name_map = {"Train": "training", "Val": "validation", "Test": "testing"}
            tb_tag = f"inference/{name_map.get(name, name.lower())}"
            tb_writer.add_text(tb_tag, "\n".join(lines), global_step=engine.state.epoch)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate text token IDs from prompt input_ids.

        Expects a pre-padded tensor (B, S). The caller is responsible for
        left-padding and providing an attention mask before calling this method.

        Args:
            input_ids: Prompt token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            attention_mask: Optional attention mask of shape (batch, seq_len).
        Returns:
            Tensor containing the full token IDs (prompt + newly generated tokens)
            of shape (batch, prompt_len + new_tokens).
        """
        self.model.eval()

        eos_id = self.tokenizer.eos_token_id

        device = self.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.model(generated, attention_mask=attention_mask)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if eos_id is not None:
                eos_mask = generated[:, -1:].eq(eos_id)
                next_token = torch.where(eos_mask, torch.tensor(eos_id, device=device), next_token)
            generated = torch.cat([generated, next_token], dim=-1)

            next_mask = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=device,
            )

            already_ended = generated[:, -2:-1].eq(eos_id)
            next_mask = torch.where(
                already_ended,
                torch.tensor(0, dtype=attention_mask.dtype, device=device),
                next_mask,
            )
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

            if generated[:, -1].eq(eos_id).all():
                break

        return generated

    def test(self, test_loader: DataLoader | None = None) -> None:
        loader = test_loader or self.test_loader
        if loader is None:
            self.logger.warning("No test loader provided. Skipping testing.")
            return

        # Load best model if available
        checkpoint_handler = self.handlers.get("checkpoint_best") or self.handlers.get("checkpoint_last")
        if checkpoint_handler and checkpoint_handler.last_checkpoint:
            checkpoint_path = checkpoint_handler.last_checkpoint

            self.logger.info("Loading best model for testing from %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model"])

        self.logger.info("Running testing...")
        self.test_evaluator.run(loader)
        metrics = self.test_evaluator.state.metrics
        self.logger.info(
            "Test results: loss=%.4f token_acc=%.4f",
            metrics["loss"],
            metrics["token_accuracy"],
        )

    def run(self) -> None:
        # create run directory and handlers when the run actually starts
        if self.run_dir is None:
            self.run_dir = self._make_run_dir()
            dump_config(self.config, self.run_dir / "config.yaml")
            self._attach_handlers()

        self.logger.info("starting run in %s", self.run_dir)
        config_data = self.config.model_dump(by_alias=True, polymorphic_serialization=True)
        if "tensorboard" in self.handlers:
            self.handlers["tensorboard"].writer.add_text("config", str(config_data))

        self.engine.run(self.train_loader, max_epochs=self.epochs)

        if self.test_loader:
            self.test()

        if "tensorboard" in self.handlers:
            self.handlers["tensorboard"].close()
