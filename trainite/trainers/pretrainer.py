import logging
from collections.abc import Sized
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

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
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset, random_split

from trainite.config.base import (
    ComponentConfig,
    DataConfigBase,
    DataLoaderConfig,
    OptimizerConfig,
    OutputConfig,
    SplitConfig,
)

# __MODEL_IMPORT__
# __DATASET_IMPORT__
from trainite.utils import dump_config, get_target, instantiate


class GenerativeModel(Protocol):
    def generate(
        self,
        prompt: list[str],
        max_new_tokens: int,
        tokenizer: object,
        eos_token_id: int | None = None,
    ) -> list[str]: ...


class PreTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    epochs: int = Field(default=3, gt=0)
    log_every_steps: int = Field(default=10, gt=0)
    early_stopping_patience: int | None = Field(default=3, gt=0)
    # Inference logging — validated at runtime in _setup_inference
    inference_every_epochs: int | None = Field(default=None)
    inference_num_samples: int = Field(default=5)
    max_inference_new_tokens: int = Field(default=16)
    grad_clip_norm: float | None = Field(default=None, gt=0.0)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    model: ComponentConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfigBase
    trainer: PreTrainerConfig
    output: OutputConfig
    seed: int = 42
    device: str = "auto"


class PreTrainer:
    def __init__(self, config: ProjectConfig) -> None:
        self.logger: logging.Logger = setup_logger("trainer", level=logging.INFO)
        torch.manual_seed(config.seed)
        self.config: ProjectConfig = config

        self.train_loader: DataLoader
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.inference_tokenizer: object | None = None

        self.device: str | torch.device = self._resolve_device()
        self.epochs: int = config.trainer.epochs
        self.grad_clip_norm: float | None = getattr(
            config.trainer, "grad_clip_norm", None
        )
        self.set_loaders()
        self.vocab_size: int = self._resolve_vocab_size()
        self.model: nn.Module = self._build_model()

        self._setup_inference(config)

        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Optimizer = self._build_optimizer()
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

    def _get_underlying_dataset(self, dataset: Dataset) -> Dataset:
        while hasattr(dataset, "dataset"):
            dataset = getattr(dataset, "dataset")
        return dataset

    def _resolve_vocab_size(self) -> int:
        train_dataset = self._get_underlying_dataset(self.train_loader.dataset)
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
        if vocab_size <= 0:
            raise ValueError(
                f"Resolved vocab_size is {vocab_size}. This usually means your custom dataset "
                "class does not expose a 'vocab_size' attribute, and 'vocab_size' was not "
                "specified in the 'model' block of config.yaml. Please define it in either place."
            )
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
        collate_config = dl_config.collate_fn
        if collate_config:
            target_symbol = get_target(collate_config.target)
            if isinstance(target_symbol, type):
                underlying_dataset = self._get_underlying_dataset(dataset)
                tokenizer = getattr(underlying_dataset, "tokenizer", None)
                collate_fn = instantiate(collate_config, tokenizer=tokenizer)
            else:
                collate_fn = target_symbol

        return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **dl_kwargs)

    def _build_loaders_from_ratios(
        self, data_config: DataConfigBase
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        if data_config.dataset is None:
            raise ValueError(
                "Dataset config must not be None for automatic ratio splitting."
            )
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
            self.val_loader = (
                self._build_dataloader(self.config.data.val)
                if self.config.data.val
                else None
            )
            self.test_loader = (
                self._build_dataloader(self.config.data.test)
                if self.config.data.test
                else None
            )
        elif self.config.data.dataset:
            self.logger.info(
                "Automatic splitting requested with ratios: train=%s, val=%s",
                self.config.data.train_ratio,
                self.config.data.val_ratio,
            )
            self.train_loader, self.val_loader, self.test_loader = (
                self._build_loaders_from_ratios(self.config.data)
            )

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

    def _flatten_accuracy(
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

        for prefix, evaluator in [
            ("train", self.train_evaluator),
            ("val", self.val_evaluator),
            ("test", self.test_evaluator),
        ]:
            loss = Loss(self.loss_fn, output_transform=self._flatten_loss)
            exact_acc = Accuracy(output_transform=self._exact_accuracy_transform)
            token_acc = Accuracy(output_transform=self._flatten_accuracy)

            loss.attach(evaluator, "loss")
            exact_acc.attach(evaluator, "exact_accuracy")
            token_acc.attach(evaluator, "token_accuracy")

            self.metrics[f"{prefix}_loss"] = loss
            self.metrics[f"{prefix}_exact_accuracy"] = exact_acc
            self.metrics[f"{prefix}_token_accuracy"] = token_acc

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
                Events.EPOCH_COMPLETED, self._log_inference, self.train_loader, "Train"
            )
            if self.val_loader is not None:
                self.engine.add_event_handler(
                    Events.EPOCH_COMPLETED, self._log_inference, self.val_loader, "Val"
                )
            if self.test_loader is not None:
                self.engine.add_event_handler(
                    Events.EPOCH_COMPLETED,
                    self._log_inference,
                    self.test_loader,
                    "Test",
                )

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

        metric_names = ["loss", "exact_accuracy", "token_accuracy"]
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
                "epoch=%s train_loss=%.4f train_exact_match_acc=%.4f train_token_acc=%.4f val_loss=%.4f val_exact_match_acc=%.4f val_token_acc=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["exact_accuracy"],
                train_metrics["token_accuracy"],
                val_metrics["loss"],
                val_metrics["exact_accuracy"],
                val_metrics["token_accuracy"],
            )
        else:
            self.logger.info(
                "epoch=%s train_loss=%.4f train_exact_match_acc=%.4f train_token_acc=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["exact_accuracy"],
                train_metrics["token_accuracy"],
            )

    def _validate_dataloader_for_inference(self, loader: DataLoader, name: str) -> None:
        dataset = loader.dataset
        if not isinstance(dataset, Sized) or len(dataset) == 0:
            return

        item = dataset[0]
        if not isinstance(item, dict):
            raise ValueError(
                f"{name} dataset items must be dictionaries containing 'source_text' and 'target_text' keys for inference logging."
            )
        if not all(key in item for key in ["source_text", "target_text"]):
            raise ValueError(
                f"{name} dataset items must contain 'source_text' and 'target_text' keys for inference logging."
            )

    def _setup_inference(self, config: ProjectConfig) -> None:
        self.inference_every_epochs: int | None = getattr(
            config.trainer, "inference_every_epochs", None
        )
        if self.inference_every_epochs is None:
            return

        self.inference_num_samples: int = getattr(
            config.trainer, "inference_num_samples"
        )
        self.max_inference_new_tokens: int = getattr(
            config.trainer, "max_inference_new_tokens"
        )
        inference_params = (
            self.inference_every_epochs,
            self.inference_num_samples,
            self.max_inference_new_tokens,
        )
        if any(
            (not isinstance(param, int) or isinstance(param, bool))
            for param in inference_params
        ):
            raise TypeError(
                f"Inference logging parameters must be integers.\n"
                f"Got inference_every_epochs={self.inference_every_epochs}, "
                f"inference_num_samples={self.inference_num_samples}, "
                f"max_inference_new_tokens={self.max_inference_new_tokens}"
            )

        if any(param <= 0 for param in inference_params):
            raise ValueError(
                f"Inference logging parameters must be greater than 0.\n"
                f"Got inference_every_epochs={self.inference_every_epochs}, "
                f"inference_num_samples={self.inference_num_samples}, "
                f"max_inference_new_tokens={self.max_inference_new_tokens}"
            )

        if not hasattr(self.model, "generate"):
            raise ValueError(
                "Model must implement 'generate' method for inference logging."
            )

        # Retrieve and check training tokenizer
        train_ds = self._get_underlying_dataset(self.train_loader.dataset)
        if not hasattr(train_ds, "tokenizer"):
            raise ValueError(
                "Dataset must have a 'tokenizer' attribute for inference logging."
            )
        self.inference_tokenizer = getattr(train_ds, "tokenizer")

        # Validate active loaders
        self._validate_dataloader_for_inference(self.train_loader, "Train")
        if self.val_loader is not None:
            self._validate_dataloader_for_inference(self.val_loader, "Val")
        if self.test_loader is not None:
            self._validate_dataloader_for_inference(self.test_loader, "Test")

    def _log_inference(self, engine: Engine, loader: DataLoader, name: str) -> None:
        if (
            not self.inference_every_epochs
            or engine.state.epoch % self.inference_every_epochs != 0
        ):
            return

        tokenizer = self.inference_tokenizer
        if tokenizer is None:
            return

        self.logger.info(
            f"Epoch {engine.state.epoch}: Running inference on {name} samples..."
        )

        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        dataset = loader.dataset
        if not isinstance(dataset, Sized):
            raise TypeError(f"{name} dataset must be Sized to run inference logging.")
        total_samples = len(dataset)
        num_samples = min(self.inference_num_samples, total_samples)
        samples = [dataset[i] for i in range(num_samples)]
        prompts = [sample["source_text"] for sample in samples]
        targets = [sample["target_text"] for sample in samples]

        self.model.eval()
        generative_model = cast(GenerativeModel, self.model)
        decoded_strs = generative_model.generate(
            prompts,
            max_new_tokens=self.max_inference_new_tokens,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
        )
        for idx in range(num_samples):
            self.logger.info(
                "    Sample %d | Prompt: %r | Target: %r | Prediction: %r",
                idx + 1,
                prompts[idx],
                targets[idx],
                decoded_strs[idx],
            )

        if "tensorboard" in self.handlers:
            tb_writer = self.handlers["tensorboard"].writer
            tb_table = [
                "| Sample | Prompt | Target | Prediction |",
                "|---|---|---|---|",
            ]
            for idx in range(num_samples):
                prompt_escaped = prompts[idx].replace("|", "\\|").replace("\n", "<br>")
                target_escaped = targets[idx].replace("|", "\\|").replace("\n", "<br>")
                pred_escaped = (
                    decoded_strs[idx].replace("|", "\\|").replace("\n", "<br>")
                )
                tb_table.append(
                    f"| {idx + 1} | {prompt_escaped} | {target_escaped} | {pred_escaped} |"
                )
            name_map = {"Train": "training", "Val": "validation", "Test": "testing"}
            tb_tag = f"inference/{name_map.get(name, name.lower())}"
            tb_writer.add_text(
                tb_tag, "\n".join(tb_table), global_step=engine.state.epoch
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
            "Test results: loss=%.4f exact_match_acc=%.4f token_acc=%.4f",
            metrics["loss"],
            metrics["exact_accuracy"],
            metrics["token_accuracy"],
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
