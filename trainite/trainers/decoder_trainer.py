import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    create_lr_scheduler_with_warmup,
)
import ignite.distributed as idist
from ignite.handlers.fbresearch_logger import FBResearchLogger
from ignite.handlers.param_scheduler import ParamScheduler
from ignite.handlers.tensorboard_logger import TensorboardLogger
from ignite.handlers.wandb_logger import WandBLogger
from ignite.metrics import Accuracy, Loss, Metric, RunningAverage
from ignite.utils import setup_logger
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset, random_split

from trainite.config.base import (
    DataConfigBase,
    DataWithAutoSplit,
    OptimizerConfig,
    OutputConfig,
)
from trainite.datasets.transformed import TransformedDataset

# __MODEL_IMPORT__
# __DATASET_IMPORT__
# __PREPROCESSOR_IMPORT__
from trainite.shared.utils import dump_config, get_target, instantiate


class TrainerConfig(BaseModel):
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
    preprocessor: BaseModel
    model: BaseModel
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfigBase | DataWithAutoSplit
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    output: OutputConfig
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    seed: int = 42
    device: str | None = None


# Resolve the vocabulary size for the model based on the tokenizer and model configuration.
def resolve_vocab_size(tokenizer: Any, model_config: Any) -> int:
    if tokenizer is None or not hasattr(tokenizer, "vocab_size"):
        raise ValueError("Tokenizer is missing or does not define a 'vocab_size' attribute.")
    vocab_size = tokenizer.vocab_size
    model_params = model_config.model_dump(by_alias=True)
    configured_vocab_size: int | None = model_params.get("vocab_size")
    if configured_vocab_size is not None:
        if configured_vocab_size < vocab_size:
            raise ValueError(
                f"Configured model vocab_size ({configured_vocab_size}) is smaller than "
                f"the tokenizer vocabulary size ({vocab_size}). "
                f"Please increase model vocab_size or remove it from config.yaml "
                f"to let it resolve automatically."
            )
        vocab_size = configured_vocab_size
    return vocab_size


# Inspects the target symbol's signature and filters the candidates to
# only include those that are accepted by the target symbol.
def _inject_if_accepted(target_symbol: Any, **candidates: Any) -> dict[str, Any]:
    try:
        sig = inspect.signature(target_symbol)
        return {k: v for k, v in candidates.items() if k in sig.parameters}
    except Exception:
        return {}


# Builds the model based on the provided configuration, tokenizer, and device.
def build_model(model_config: Any, tokenizer: Any, vocab_size: int, device: str | torch.device) -> nn.Module:
    target_symbol = get_target(model_config.target)
    kwargs = _inject_if_accepted(
        target_symbol,
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = instantiate(model_config, **kwargs)
    model.to(device)
    return model


def build_dataset(dataset_config: Any, transform_config: Any, tokenizer: Any) -> Dataset:
    ds = get_target(dataset_config.target)
    dataset = instantiate(dataset_config, **_inject_if_accepted(ds, preprocessor=tokenizer, tokenizer=tokenizer))
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
) -> DataLoader:
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
    if isinstance(data_config, DataWithAutoSplit):
        return _loaders_from_ratios(data_config, tokenizer, seed)
    return _loaders_from_splits(data_config, tokenizer)


class Trainer:
    def __init__(self, config: ProjectConfig) -> None:
        self.logger: logging.Logger = setup_logger("trainer", level=logging.INFO)
        torch.manual_seed(config.seed)
        self.config: ProjectConfig = config
        self.device: str | torch.device = idist.device() if config.device is None else config.device
        self.tokenizer = instantiate(config.preprocessor)
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            config.data, self.tokenizer, config.seed
        )
        vocab_size = resolve_vocab_size(self.tokenizer, config.model)
        self.model = build_model(config.model, self.tokenizer, vocab_size, self.device)
        self.optimizer = instantiate(config.optimizer, params=self.model.parameters())
        self.epochs: int = config.trainer.epochs
        self.grad_clip_norm: float | None = getattr(config.trainer, "grad_clip_norm", None)
        self.inference_every_epochs = config.trainer.inference_every_epochs
        self.inference_num_samples = config.trainer.inference_num_samples
        self.max_inference_new_tokens = config.trainer.max_inference_new_tokens
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.total_iters: int = len(self.train_loader) * self.epochs
        self.checkpointers: dict[str, Checkpoint] = {}
        self.engine = Engine(self._train_step)
        self.train_evaluator = Engine(self._eval_step)
        self.val_evaluator = Engine(self._eval_step)
        self.test_evaluator = Engine(self._eval_step)
        self.metrics: dict = self._attach_metrics()

        # Create run directory for outputs
        self.run_dir = self._make_run_dir()
        dump_config(self.config, self.run_dir / "config.yaml")

        # Attach loggers for console
        self.logger = self.attach_loggers()

        # Attach learning rate scheduler
        self.attach_lr_scheduler()

        # Attach early stopping
        self.setup_early_stopping()

        # Attach checkpointing
        self.checkpointers = self.setup_checkpointing()

        # Attach experiment logger (TensorBoard or Weights & Biases)
        self.exp_logger = self._setup_experiment_logger()

        # Run evaluations at the end of each epoch to log training and validation metrics
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, self._run_evaluations)

        # Attach inference logger if inference logging is enabled
        if self.inference_every_epochs is not None:
            self.attach_inference_logger()

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

    def attach_loggers(self) -> logging.Logger:
        logger = setup_logger(
            "trainer",
            level=logging.INFO,
            filepath=str(self.run_dir / "output.log") if self.run_dir else None,
            reset=True,
        )
        train_fb_logger: FBResearchLogger = FBResearchLogger(logger=logger, show_output=True)
        train_fb_logger.attach(
            self.engine,
            name="Train",
            every=self.config.trainer.log_every_steps,
            optimizer=self.optimizer,
            output_transform=lambda output: {"loss": output["loss"].item()},
        )
        return logger

    def attach_lr_scheduler(self) -> None:
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

    def attach_inference_logger(self) -> None:
        self.engine.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.inference_every_epochs),
            self._log_inference,
            self.train_loader,
            "Train",
        )

        self.engine.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.inference_every_epochs),
            self._log_inference,
            self.val_loader,
            "Val",
        )

    def setup_early_stopping(self) -> None:
        patience = self.config.trainer.early_stopping_patience
        if patience is not None:
            early_stopping = EarlyStopping(
                patience=patience,
                score_function=lambda engine: engine.state.metrics["loss"],
                trainer=self.engine,
                min_delta=0.0,
                mode="min",
            )
            self.val_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    def setup_checkpointing(self) -> dict[str, Checkpoint]:
        to_save = {"model": self.model, "optimizer": self.optimizer}
        checkpointers = {}

        def score_function(engine):
            loss = engine.state.metrics["loss"]
            return -loss

        checkpoint = Checkpoint(
            to_save=to_save,
            save_handler=DiskSaver(dirname=str(self.run_dir), require_empty=False),
            filename_prefix="best",
            score_function=score_function,
            score_name="val_loss",
            n_saved=1,
            global_step_transform=lambda *_: self.engine.state.iteration,
        )
        self.val_evaluator.add_event_handler(Events.COMPLETED, checkpoint)
        checkpointers["checkpoint_best"] = checkpoint

        last_checkpoint = Checkpoint(
            to_save=to_save,
            save_handler=DiskSaver(dirname=str(self.run_dir), require_empty=False),
            filename_prefix="last",
            n_saved=1,
            global_step_transform=lambda *_: self.engine.state.iteration,
        )
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, last_checkpoint)

        checkpointers["checkpoint_last"] = last_checkpoint
        return checkpointers

    def _setup_experiment_logger(self) -> TensorboardLogger | WandBLogger:
        if self.config.logger == "wandb":
            exp_logger: TensorboardLogger | WandBLogger = WandBLogger(
                project=self.config.output.run_name, dir=str(self.run_dir), name=str(self.run_dir).split("/")[-1]
            )

        else:
            log_dir = self.run_dir / "tensorboard" if self.run_dir else None
            exp_logger = TensorboardLogger(log_dir=log_dir)
        metric_names = ["loss", "token_accuracy"]

        # Log training iteration loss
        exp_logger.attach_output_handler(
            self.engine,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda output: {"batch_loss": output["loss"]},
        )

        # Log training epoch metrics
        exp_logger.attach_output_handler(
            self.train_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="training",
            metric_names=metric_names,
            global_step_transform=lambda *_: self.engine.state.iteration,
        )

        # Log validation epoch metrics
        exp_logger.attach_output_handler(
            self.val_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=metric_names,
            global_step_transform=lambda *_: self.engine.state.iteration,
        )

        # Log test metrics if applicable
        if self.test_loader:
            exp_logger.attach_output_handler(
                self.test_evaluator,
                event_name=Events.COMPLETED,
                tag="testing",
                metric_names=metric_names,
                global_step_transform=lambda *_: self.engine.state.iteration,
            )

        # Log optimizer learning rates
        exp_logger.attach_opt_params_handler(
            self.engine,
            event_name=Events.ITERATION_STARTED,
            optimizer=self.optimizer,
        )
        return exp_logger

    def _log_text(self, tag: str, text: str, step: int) -> None:
        # Both backends escape HTML in the caller; wandb renders it, TB uses markdown.
        if self.config.logger == "wandb":
            self.exp_logger.log({tag: self.exp_logger.Html(f"<pre>{text}</pre>")}, step=step)
        else:
            self.exp_logger.writer.add_text(tag, text, global_step=step)

    def _run_evaluations(self, engine: Engine) -> None:
        self.logger.info("Evaluating on training set...")
        self.train_evaluator.run(self.train_loader)
        train_metrics = self.train_evaluator.state.metrics
        epoch = engine.state.epoch

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

    def _log_inference(self, engine: Engine, loader: DataLoader, name: str) -> None:
        self.logger.info(f"Epoch {engine.state.epoch}: Running inference on {name} samples...")

        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

        dataset = loader.dataset
        total_samples = len(dataset)  # type: ignore
        num_samples = min(self.inference_num_samples, total_samples)

        # Read generation prompts straight off the DatapointModel contract.
        prompt_ids_list: list[torch.Tensor] = []
        prompt_attn_list: list[torch.Tensor] = []
        targets: list[str] = []
        sources: list[str] = []
        for i in range(num_samples):
            item = dataset[i]
            source = item.source
            target = item.target

            input_ids = item.eval_input_ids
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            prompt_ids_list.append(input_ids)
            prompt_attn_list.append(attention_mask)
            targets.append(target)
            sources.append(source)
        self.model.eval()

        # Left-pad the prompts (reverse -> pad_sequence -> reverse) so every
        # sequence in the batch ends at the same position and the newly
        # generated tokens line up in a single column.
        #
        # NOTE: this is safe for the default model because it uses rotary
        # (relative) positions, where shifting the whole prompt right by the
        # pad width preserves the distances between real tokens and the padding
        # is masked out of attention. If you swap in a model with absolute or
        # sinusoidal position embeddings indexed by slot, left-padding will
        # offset every real token's position and silently degrade generation --
        # you would then need to right-pad, or derive position_ids from the
        # attention mask instead of using the raw slot index.
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

        lines = []
        for idx in range(num_samples):
            lines.append(f"Sample {idx + 1}")
            prompt = sources[idx].strip() or "(empty)"
            target = targets[idx].strip() or "(empty)"
            pred = decoded_strs[idx].strip() or "(empty)"
            for key, val in [("Prompt", prompt), ("Target", target), ("Prediction", pred)]:
                # Escape markup so TB markdown / wandb HTML don't swallow <, >, & in outputs.
                val = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                lines.append(f"  {key}:     {val}")
            lines.append("")
        name_map = {"Train": "training", "Val": "validation", "Test": "testing"}
        tag = f"inference/{name_map.get(name, name.lower())}"
        self._log_text(tag, "\n".join(lines), engine.state.iteration)

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

            # Once a sequence has emitted EOS we keep re-emitting it (above) and
            # mask the appended token out of attention, so a finished sequence
            # stays frozen while the rest of the batch keeps generating.
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
        checkpoint_handler = self.checkpointers.get("checkpoint_best", None)
        if checkpoint_handler and checkpoint_handler.last_checkpoint:
            checkpoint_path = checkpoint_handler.last_checkpoint

            self.logger.info("Loading best model for testing from %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.logger.warning("No best model checkpoint found. Using current model for testing.")

        self.logger.info("Running testing...")
        self.test_evaluator.run(loader)
        metrics = self.test_evaluator.state.metrics
        self.logger.info(
            "Test results: loss=%.4f token_acc=%.4f",
            metrics["loss"],
            metrics["token_accuracy"],
        )

    def run(self) -> None:
        self.logger.info("starting run in %s", self.run_dir)
        config_data = self.config.model_dump(by_alias=True, polymorphic_serialization=True)
        self._log_text("config", str(config_data), 0)

        self.engine.run(self.train_loader, max_epochs=self.epochs)

        if self.test_loader:
            self.test()

        if self.config.logger == "wandb":
            self._upload_model_to_wandb()

        self.exp_logger.close()

    def _upload_model_to_wandb(self) -> None:
        checkpoint = self.checkpointers.get("checkpoint_best") or self.checkpointers.get("checkpoint_last")
        checkpoint_path = checkpoint.last_checkpoint if checkpoint else None
        if not checkpoint_path:
            self.logger.warning("No checkpoint found; skipping model upload to Weights & Biases.")
            return
        # WandBLogger forwards attribute access to the wandb module (same as .log/.Html used above).
        artifact = self.exp_logger.Artifact(name=f"{self.config.output.run_name}-model".replace("/", "-"), type="model")
        artifact.add_file(str(checkpoint_path))
        self.exp_logger.log_artifact(artifact)
        self.logger.info("Uploaded model checkpoint to Weights & Biases: %s", checkpoint_path)
