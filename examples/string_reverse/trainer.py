import itertools
from itertools import zip_longest
import logging

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, Metric, RunningAverage
from ignite.utils import setup_logger
from torch import nn
from torch.utils.data import DataLoader
from ignite.handlers import DiskSaver
from ignite.handlers.clearml_logger import ClearMLSaver

from config import (
    ProjectConfig,
    TrainerConfig,
)

from utils import (
    attach_early_stopping,
    attach_lr_floor_stopping,
    attach_perfect_accuracy_stopping,
    attach_lr_scheduler,
    build_dataloaders,
    build_model,
    dump_config,
    instantiate,
    make_run_dir,
    setup_best_model_checkpoint,
    setup_console_logger,
    setup_experiment_tracking,
    setup_training_checkpointing,
)


def _flatten(output: dict[str, torch.Tensor], ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    logits = output["logits"].reshape(-1, output["logits"].size(-1))
    targets = output["targets"].reshape(-1)
    mask = targets != ignore_index
    return logits[mask], targets[mask]


def _exact_accuracy_transform(
    output: dict[str, torch.Tensor], ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = output["logits"]
    targets = output["targets"]

    preds = torch.argmax(logits, dim=-1)
    mask = targets != ignore_index

    correct = (preds == targets) | ~mask
    seq_correct = correct.all(dim=-1)

    y_pred = seq_correct.long()
    y = torch.ones_like(seq_correct)

    return y_pred, y


class Trainer:
    def __init__(self, config: ProjectConfig) -> None:
        self.logger: logging.Logger = setup_logger("trainer", level=logging.INFO)
        torch.manual_seed(config.seed)
        self.config: ProjectConfig = config
        self.trainer_config: TrainerConfig = config.trainer
        self.device: str | torch.device = idist.device() if config.device is None else config.device
        self.tokenizer = instantiate(config.preprocessor)
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            config.data, self.tokenizer, config.seed
        )
        self.model = build_model(
            config.model, self.device, vocab_size=self.tokenizer.vocab_size, pad_token_id=self.tokenizer.pad_token_id
        )
        self.optimizer = instantiate(config.optimizer, params=self.model.parameters())
        self.epochs: int = self.trainer_config.epochs
        self.grad_clip_norm: float | None = getattr(self.trainer_config, "grad_clip_norm", None)
        self.inference_every_epochs = self.trainer_config.inference_every_epochs
        self.inference_num_samples = self.trainer_config.inference_num_samples
        self.max_inference_new_tokens = self.trainer_config.max_inference_new_tokens
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.total_iters: int = len(self.train_loader) * self.epochs
        self.trainer = Engine(self._train_step)
        self.train_evaluator = Engine(self._eval_step)
        self.val_evaluator = Engine(self._eval_step)
        self.test_evaluator = Engine(self._eval_step)
        self.metrics: dict = self._attach_metrics()

        # Run evaluations at the end of each epoch to log training and validation metrics
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._run_evaluations)

        # Create run directory for outputs
        self.run_dir = make_run_dir(config.output)
        dump_config(self.config, self.run_dir / "config.yaml")

        # Attach loggers for console
        self.logger = setup_console_logger(
            self.run_dir, self.trainer, self.trainer_config.log_every_steps, self.optimizer
        )

        # Attach learning rate scheduler
        attach_lr_scheduler(
            self.val_evaluator,
            self.optimizer,
            config.scheduler.metric_name,
            config.scheduler.mode,
            config.scheduler.patience,
            config.scheduler.factor,
            config.scheduler.min_lr,
        )

        # Attach early stopping
        if self.trainer_config.early_stopping_patience is not None:
            attach_early_stopping(self.val_evaluator, self.trainer, self.trainer_config.early_stopping_patience)
        if self.trainer_config.stop_on_perfect_acc:
            attach_perfect_accuracy_stopping(self.val_evaluator, self.trainer)
        if self.trainer_config.stop_on_lr_floor:
            attach_lr_floor_stopping(self.val_evaluator, self.trainer, self.optimizer, config.scheduler.min_lr)

        # Define validation scoring function for best model
        def score_function(engine_val):
            loss = engine_val.state.metrics["loss"]
            return -loss

        # Attach experiment logger (TensorBoard or ClearML)
        self.exp_logger = setup_experiment_tracking(
            config.logger,
            self.trainer,
            self.val_evaluator,
            self.train_evaluator,
            self.test_evaluator,
            self.optimizer,
            self.run_dir,
            ["loss", "token_accuracy", "exact_accuracy"],
            bool(self.test_loader),
            config.output.run_name,
            project=config.output.project,
        )

        # Setup save handler
        if config.logger == "clearml":
            save_handler = ClearMLSaver(logger=self.exp_logger, dirname=str(self.run_dir), require_empty=False)
        else:
            save_handler = DiskSaver(dirname=str(self.run_dir), require_empty=False)

        # Attach checkpointing
        self.best_checkpoint = setup_best_model_checkpoint(
            self.trainer,
            self.val_evaluator,
            {"model": self.model, "optimizer": self.optimizer},
            save_handler,
            score_function=score_function,
            score_name="val_loss",
        )
        self.last_checkpoint = setup_training_checkpointing(
            self.trainer,
            {"model": self.model, "optimizer": self.optimizer},
            save_handler,
        )
        # Attach inference logger if inference logging is enabled
        if self.inference_every_epochs is not None:
            self.attach_inference_logger()

    def _train_step(self, engine: Engine, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.model.train()
        inputs = batch["input_ids"].to(self.device)
        targets = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(inputs, attention_mask=attention_mask)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
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

    def _run_evaluations(self, engine: Engine) -> None:
        self.logger.info("Evaluating on training set...")
        eval_loader = itertools.islice(self.train_loader, len(self.val_loader))
        self.train_evaluator.run(eval_loader)
        train_metrics = self.train_evaluator.state.metrics
        epoch = engine.state.epoch

        self.logger.info("Evaluating on validation set...")
        self.val_evaluator.run(self.val_loader)
        val_metrics = self.val_evaluator.state.metrics
        self.logger.info(
            "epoch=%s train_loss=%.4f train_token_acc=%.4f train_exact_acc=%.4f val_loss=%.4f val_token_acc=%.4f val_exact_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["token_accuracy"],
            train_metrics["exact_accuracy"],
            val_metrics["loss"],
            val_metrics["token_accuracy"],
            val_metrics["exact_accuracy"],
        )

    def run(self) -> None:
        self.logger.info("starting run in %s", self.run_dir)
        try:
            self.trainer.run(self.train_loader, max_epochs=self.epochs)

            if self.test_loader:
                self.test()
        finally:
            self.exp_logger.close()
            if self.config.logger == "clearml":
                task = self.exp_logger.get_task()
                if task:
                    task.close()

    def evaluate_autoregressive(
        self,
        loader: DataLoader,
        name: str,
        step: int,
        num_log_samples: int = 8,
    ) -> dict[str, float]:
        pad_token_id = self.tokenizer.pad_token_id
        batch_size = loader.batch_size
        dataset = loader.dataset
        total = len(dataset)  # type: ignore[arg-type]

        items = [dataset[i] for i in range(total)]
        prompt_ids = [it.eval_input_ids for it in items]
        sources = [it.source for it in items]
        targets = [it.target for it in items]

        self.logger.info("AR eval [%s]: decoding %d samples...", name, total)

        decoded_strs: list[str] = []
        for start in range(0, total, batch_size):
            chunk = prompt_ids[start : start + batch_size]
            # Left-pad so all prompts in the batch end at the same column.
            batch_input_ids = (
                torch.nn.utils.rnn.pad_sequence(
                    [t.flip(0) for t in chunk], batch_first=True, padding_value=pad_token_id
                )
                .flip(1)
                .to(self.device)
            )
            batch_attention_mask = (batch_input_ids != pad_token_id).long()
            sequences = self.generate(
                input_ids=batch_input_ids,
                max_new_tokens=self.max_inference_new_tokens,
                attention_mask=batch_attention_mask,
            )
            new_tokens = sequences[:, batch_input_ids.shape[1] :]
            decoded_strs.extend(
                self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True) for tokens in new_tokens
            )

        ar_exact_match_acc = sum(p == t for p, t in zip(decoded_strs, targets)) / total if total else 0.0

        # Compute token-level accuracy by comparing each character in the decoded string to the target string.
        correct_chars = total_chars = 0
        for p, t in zip(decoded_strs, targets):
            # Use zip_longest to handle cases where the predicted and target strings have different lengths.
            pairs = list(zip_longest(p, t))
            correct_chars += sum(a == b for a, b in pairs)
            total_chars += len(pairs)
        ar_token_acc = correct_chars / total_chars if total_chars else 0.0

        self.logger.info(
            "AR eval [%s] step=%d  exact_match=%.4f  token_acc=%.4f",
            name,
            step,
            ar_exact_match_acc,
            ar_token_acc,
        )

        split_tag = {"Train": "training", "Val": "validation"}.get(name, name.lower())
        payload = {
            f"ar_eval/{split_tag}/exact_match_acc": ar_exact_match_acc,
            f"ar_eval/{split_tag}/token_acc": ar_token_acc,
        }
        if self.config.logger == "clearml":
            for tag, value in payload.items():
                parts = tag.split("/")
                if len(parts) >= 2:
                    title = "/".join(parts[:-1])
                    series = parts[-1]
                else:
                    title = tag
                    series = tag
                self.exp_logger.report_scalar(title=title, series=series, value=value, iteration=step)
        else:
            for tag, value in payload.items():
                self.exp_logger.writer.add_scalar(tag, value, global_step=step)

        if num_log_samples > 0:

            def _esc(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            lines: list[str] = []
            for idx in range(min(num_log_samples, total)):
                lines += [
                    f"Sample {idx + 1}",
                    f"  Prompt:     {_esc(sources[idx].strip() or '(empty)')}",
                    f"  Target:     {_esc(targets[idx].strip() or '(empty)')}",
                    f"  Prediction: {_esc(decoded_strs[idx].strip() or '(empty)')}",
                    "",
                ]
            self._log_text(f"ar_eval/{split_tag}/samples", "\n".join(lines), step)

        return {"ar_exact_match_acc": ar_exact_match_acc, "ar_token_acc": ar_token_acc}

    def test(self, test_loader: DataLoader | None = None) -> None:
        loader = test_loader or self.test_loader
        if loader is None:
            self.logger.warning("No test loader provided. Skipping testing.")
            return

        # Load best model if available
        checkpoint_handler = self.best_checkpoint
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
            "Test results: loss=%.4f token_acc=%.4f exact_acc=%.4f",
            metrics["loss"],
            metrics["token_accuracy"],
            metrics["exact_accuracy"],
        )

        # Autoregressive evaluation — logged directly into the experiment tracker
        step = self.trainer.state.iteration if self.trainer.state else 0
        self.evaluate_autoregressive(loader, "Test", step=step)

    def _attach_metrics(self) -> dict[str, Metric]:
        RunningAverage(output_transform=lambda output: output["loss"]).attach(self.trainer, "loss")

        ignore_index = self.criterion.ignore_index

        def transform_fn(output):
            return _flatten(output, ignore_index=ignore_index)

        def exact_transform_fn(output):
            return _exact_accuracy_transform(output, ignore_index=ignore_index)

        metrics = {}
        for prefix, evaluator in [
            ("train", self.train_evaluator),
            ("val", self.val_evaluator),
            ("test", self.test_evaluator),
        ]:
            loss = Loss(self.criterion, output_transform=transform_fn)
            token_acc = Accuracy(output_transform=transform_fn)
            exact_acc = Accuracy(output_transform=exact_transform_fn)

            loss.attach(evaluator, "loss")
            token_acc.attach(evaluator, "token_accuracy")
            exact_acc.attach(evaluator, "exact_accuracy")

            metrics[f"{prefix}_loss"] = loss
            metrics[f"{prefix}_token_accuracy"] = token_acc

        return metrics

    def attach_inference_logger(self) -> None:
        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.inference_every_epochs),
            self._log_inference,
            self.train_loader,
            "Train",
        )

        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.inference_every_epochs),
            self._log_inference,
            self.val_loader,
            "Val",
        )

    def _log_text(self, tag: str, text: str, step: int) -> None:
        # Both backends escape HTML/text in the caller; clearml uses report_text, TB uses markdown.
        if self.config.logger == "clearml":
            self.exp_logger.report_text(f"[{tag}] Step {step}:\n{text}")
        else:
            self.exp_logger.writer.add_text(tag, text, global_step=step)

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
                # Escape markup so TB markdown / clearml console don't swallow <, >, & in outputs.
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
