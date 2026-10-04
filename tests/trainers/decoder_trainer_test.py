import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sized, Any, Callable
from unittest import mock

import pytest
import torch
import torch.nn as nn
from accelerate.state import AcceleratorState
from pydantic import ValidationError
from trainite.config.base import (
    DataConfigBase,
    DataLoaderConfig,
    DataWithAutoSplit,
    LossConfig,
    OptimizerConfig,
    OutputConfig,
    SplitConfig,
    ModelConfig,
    PreprocessorConfig,
    DatasetConfig,
    TransformConfig,
)
from trainite.datasets.string_reverse import DatapointModel
from trainite.trainers.decoder_trainer import Trainer, _flatten
from trainite.config.base import ProjectConfig, TrainerConfig
from ignite.engine import Events
from ignite.handlers import EarlyStopping
import ignite.distributed as idist


def create_trainer_from_config(config: ProjectConfig) -> Trainer:
    return Trainer(config)


class MockComponent(ModelConfig, PreprocessorConfig, DatasetConfig, TransformConfig):
    pass


def target_path(obj: type | Callable[..., Any] | str) -> str:
    if isinstance(obj, str):
        return obj
    return f"{obj.__module__}.{obj.__qualname__}"


def cc(
    target: type | Callable[..., Any] | str | None = None,
    **kwargs: object,
) -> MockComponent:
    if target:
        kwargs["_target_"] = target_path(target)
    return MockComponent.model_validate(kwargs)


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=10, hidden_size=8, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, attention_mask=None, **kwargs):
        return self.fc(self.embedding(x))


class SimpleModelWithTokenizer(SimpleModel):
    tokenizer = "mock_tokenizer"


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=16, seq_len=4, vocab_size=10, **kwargs):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
        }


class EmptyDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("This dataset is empty")


class DummyClassCollateFn:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return batch


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.sep_token_id = 2
        self.eos_token_id = 3
        self.vocab_size = 10

    def encode(self, text):
        return [5, 6]

    def decode(self, ids, skip_special_tokens=True):
        return "decoded_prediction"

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}


class GenerativeModel(SimpleModel):
    tokenizer = DummyTokenizer()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = DummyTokenizer()

    def generate(
        self,
        input_ids,
        max_new_tokens,
        attention_mask=None,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    ):
        dummy_new = torch.tensor([[7]], dtype=torch.long, device=input_ids.device).repeat(input_ids.shape[0], 1)
        return torch.cat([input_ids, dummy_new], dim=-1)


class DummyTransform:
    """Transform emitting the DatapointModel contract (train tensors + eval prompt)."""

    def __init__(self, tokenizer: Any = None):
        self.tokenizer = tokenizer

    def __call__(self, sample):
        input_ids = sample["input_ids"]
        prompt_ids = (
            [self.tokenizer.bos_token_id] + self.tokenizer.encode(sample["source"]) + [self.tokenizer.sep_token_id]
        )
        return DatapointModel(
            source=sample["source"],
            target=sample["target"],
            train_input_ids=input_ids,
            train_label_ids=sample["labels"],
            attention_mask=torch.ones(len(input_ids), dtype=torch.long),
            eval_input_ids=torch.tensor(prompt_ids, dtype=torch.long),
        )


class GenerativeDataset(SimpleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["source"] = f"source_{index}"
        item["target"] = f"target_{index}"
        return item


class GenerativeModelNoTokenizer(SimpleModel):
    """Like GenerativeModel but without a tokenizer — used to test the missing-tokenizer error."""

    def generate(self, input_ids, max_new_tokens, **kwargs):
        dummy_new = torch.tensor([[7]], dtype=torch.long, device=input_ids.device).repeat(input_ids.shape[0], 1)
        return torch.cat([input_ids, dummy_new], dim=-1)


def dummy_collate_fn(batch):
    return batch


@pytest.fixture(autouse=True)
def reset_accelerator_state():
    AcceleratorState._reset_state(reset_partial_state=True)
    os.environ.pop("ACCELERATE_TORCH_DEVICE", None)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    os.environ.pop("ACCELERATE_TORCH_DEVICE", None)


@pytest.fixture
def temp_run_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    logging.shutdown()  # Ensure all logging handlers are flushed and closed before removing the directory
    shutil.rmtree(temp_dir)


@pytest.fixture
def project_config(temp_run_dir):
    return ProjectConfig(
        project_name="test_project",
        preprocessor=cc(DummyTokenizer),
        model=cc(
            SimpleModel,
            vocab_size=10,
            hidden_size=8,
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-3),
        data=DataConfigBase(
            train=SplitConfig(
                dataset=cc(
                    SimpleDataset,
                    size=16,
                    seq_len=4,
                    vocab_size=10,
                ),
                dataloader=DataLoaderConfig(batch_size=4, num_workers=0),
            ),
            val=SplitConfig(
                dataset=cc(
                    SimpleDataset,
                    size=8,
                    seq_len=4,
                    vocab_size=10,
                ),
                dataloader=DataLoaderConfig(batch_size=4, num_workers=0),
            ),
        ),
        trainer=TrainerConfig(
            epochs=1,
            log_every_steps=1,
            inference_every_epochs=None,
            inference_num_samples=4,
            max_inference_new_tokens=10,
        ),
        output=OutputConfig(root=str(temp_run_dir), run_name="test_run"),
        device=None,
    )


def test_flatten():
    # Mock some data
    logits = torch.randn(2, 3, 5)  # B=2, S=3, V=5
    targets = torch.tensor([[1, 2, -100], [0, -100, 3]])

    output = {"logits": logits, "targets": targets}
    flat_logits, flat_targets = _flatten(output, ignore_index=-100)

    assert flat_logits.shape == (4, 5)  # 6 tokens total, 2 are masked
    assert flat_targets.shape == (4,)
    assert (flat_targets == torch.tensor([1, 2, 0, 3])).all()


def test_decoder_trainer_init(project_config):
    trainer = create_trainer_from_config(project_config)
    assert trainer.epochs == 1
    assert isinstance(trainer.model, SimpleModel)
    assert trainer.train_loader is not None
    assert trainer.val_loader is not None
    assert len(trainer.train_loader) == 4  # 16 / 4
    assert len(trainer.val_loader) == 2  # 8 / 4


class DoubleCELoss(nn.Module):
    """Custom loss doubling CE — proves Trainer honors loss._target_."""

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        return 2 * self.ce(logits, targets)


def test_decoder_trainer_custom_loss_from_config(project_config):
    project_config.loss = LossConfig.model_validate({"_target_": target_path(DoubleCELoss)})
    trainer = create_trainer_from_config(project_config)
    assert isinstance(trainer.criterion, DoubleCELoss)

    batch = next(iter(trainer.train_loader))
    output = trainer._train_step(trainer.trainer, batch)
    flat_logits = output["logits"].reshape(-1, output["logits"].size(-1))
    flat_targets = batch["labels"].to(trainer.device).reshape(-1)
    expected = 2 * nn.CrossEntropyLoss()(flat_logits, flat_targets)
    assert torch.isclose(output["loss"], expected)


def test_device_auto_selection(project_config):
    trainer = create_trainer_from_config(project_config)
    if isinstance(trainer.device, torch.device):
        device_str = trainer.device.type
    elif isinstance(trainer.device, str):
        device_str = trainer.device
    else:
        raise ValueError("trainer.device should be either torch.device or str")
    device = idist.device()
    assert device_str == device.type


def test_decoder_trainer_run_with_val(project_config, temp_run_dir):
    trainer = create_trainer_from_config(project_config)
    trainer.run()

    # Check if run directory was created
    run_dirs = list((temp_run_dir / "test_run").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "output.log").exists()
    assert (run_dir / "tensorboard").exists()

    # Check for checkpoints
    # ModelCheckpoint n_saved=1, and best checkpoint for val so we expect at least 2
    checkpoints = list(run_dir.glob("*.pt"))
    assert len(checkpoints) >= 2

    event_handlers = trainer.val_evaluator._event_handlers.get(Events.COMPLETED, [])
    assert any(isinstance(h[0], EarlyStopping) for h in event_handlers)
    assert trainer.best_checkpoint is not None


def test_decoder_trainer_test_no_loader(project_config):
    # Ensure test split is None (default in fixture is None)
    project_config.data.test = None
    trainer = create_trainer_from_config(project_config)
    trainer.run()

    with mock.patch.object(trainer.logger, "warning") as mock_warning:
        trainer.test()

    mock_warning.assert_called_with("No test loader provided. Skipping testing.")


@mock.patch("trainite.trainers.decoder_trainer.ClearMLSaver")
@mock.patch("trainite.trainers.decoder_trainer.setup_clearml_logging")
@mock.patch("trainite.trainers.decoder_trainer.setup_best_model_checkpoint")
@mock.patch("trainite.trainers.decoder_trainer.setup_training_checkpointing")
def test_clearml_saver_is_used(
    mock_setup_training, mock_setup_best, mock_setup_logging, mock_clearml_saver, project_config, temp_run_dir
):
    project_config.logger = "clearml"
    mock_logger = mock.MagicMock()
    mock_setup_logging.return_value = mock_logger

    trainer = create_trainer_from_config(project_config)

    mock_setup_logging.assert_called_once_with(
        trainer=trainer.trainer,
        optimizers=trainer.optimizer,
        evaluators=mock.ANY,
        log_every_iters=project_config.trainer.log_every_steps,
        trainer_metric_names=["batch_loss"],
        evaluator_metric_names=["loss", "token_accuracy"],
        project_name="test_project",
        task_name=trainer.run_dir.name,
    )

    # Assert ClearMLSaver was initialized with the exp_logger
    mock_clearml_saver.assert_called_once_with(
        logger=mock_logger,
        dirname=str(trainer.run_dir),
        output_uri=True,
        require_empty=False,
    )

    # Assert setup_best_model_checkpoint was called with the clearml saver
    mock_setup_best.assert_called_once_with(
        trainer.trainer,
        trainer.val_evaluator,
        {"model": trainer.model, "optimizer": trainer.optimizer},
        mock_clearml_saver.return_value,
        score_function=mock.ANY,
        score_name="val_loss",
    )

    # Assert setup_training_checkpointing was called with the clearml saver
    mock_setup_training.assert_called_once_with(
        trainer.trainer,
        {"model": trainer.model, "optimizer": trainer.optimizer},
        mock_clearml_saver.return_value,
    )


def test_decoder_trainer_test_method(project_config, temp_run_dir):
    # Add test split to config
    project_config.data.test = SplitConfig(
        dataset=cc(
            SimpleDataset,
            size=4,
            seq_len=4,
            vocab_size=10,
        ),
        dataloader=DataLoaderConfig(batch_size=4, num_workers=0),
    )

    trainer = create_trainer_from_config(project_config)
    trainer.run()
    trainer.test()

    # Test evaluators should have run
    assert "loss" in trainer.test_evaluator.state.metrics
    assert "token_accuracy" in trainer.test_evaluator.state.metrics


def test_decoder_trainer_test_loads_best_checkpoint(project_config, temp_run_dir):
    # Add test split
    project_config.data.test = SplitConfig(
        dataset=cc(
            SimpleDataset,
            size=4,
            seq_len=4,
            vocab_size=10,
        ),
        dataloader=DataLoaderConfig(batch_size=4, num_workers=0),
    )

    trainer = create_trainer_from_config(project_config)
    trainer.run()

    assert trainer.best_checkpoint is not None

    with mock.patch("torch.load", side_effect=torch.load) as mock_load:
        trainer.test()

    # Verify that it loaded the best checkpoint
    best_checkpoint_path = trainer.best_checkpoint.last_checkpoint
    mock_load.assert_any_call(best_checkpoint_path, map_location=trainer.device, weights_only=True)


def test_decoder_trainer_dataloader_collate_fn(project_config):
    project_config.model.collate_fn_target = target_path(dummy_collate_fn)
    trainer = create_trainer_from_config(project_config)
    assert trainer.train_loader is not None
    assert trainer.train_loader.collate_fn is dummy_collate_fn


def test_decoder_trainer_explicit_split_shuffle(project_config):
    project_config.data.train.dataloader.shuffle = True
    trainer = create_trainer_from_config(project_config)
    # PyTorch DataLoader uses RandomSampler when shuffle is True;
    # Accelerate wraps it in DataLoaderShard with batch_sampler.sampler as RandomSampler
    assert isinstance(trainer.train_loader.batch_sampler.sampler, torch.utils.data.RandomSampler)


def test_decoder_trainer_builds_train_and_val_loaders_from_ratios(tmp_path):
    config = ProjectConfig(
        project_name="test_project",
        preprocessor=cc(DummyTokenizer),
        model=cc(
            SimpleModel,
            vocab_size=100,
            hidden_size=32,
        ),
        data=DataWithAutoSplit(
            dataset=cc(
                SimpleDataset,
                size=100,
                seq_len=10,
                vocab_size=100,
            ),
            test_ratio=0.0,
            val_ratio=0.2,
        ),
        trainer=TrainerConfig(epochs=1),
        output=OutputConfig(root=str(tmp_path), run_name="test"),
    )

    trainer = create_trainer_from_config(config)

    assert trainer.train_loader is not None
    assert trainer.val_loader is not None
    assert isinstance(trainer.train_loader.dataset, Sized)
    assert isinstance(trainer.val_loader.dataset, Sized)
    assert len(trainer.train_loader.dataset) == 80
    assert len(trainer.val_loader.dataset) == 20
    assert trainer.test_loader is None


def test_decoder_trainer_builds_train_val_and_test_loaders_from_ratios(tmp_path):
    config = ProjectConfig(
        project_name="test_project",
        preprocessor=cc(DummyTokenizer),
        model=cc(
            SimpleModel,
            vocab_size=100,
            hidden_size=32,
        ),
        data=DataWithAutoSplit(
            dataset=cc(
                SimpleDataset,
                size=100,
                seq_len=10,
                vocab_size=100,
            ),
            test_ratio=0.2,
            val_ratio=0.2,
        ),
        trainer=TrainerConfig(epochs=1),
        output=OutputConfig(root=str(tmp_path), run_name="test"),
    )

    trainer = create_trainer_from_config(config)

    assert trainer.train_loader is not None
    assert trainer.val_loader is not None
    assert trainer.test_loader is not None

    assert isinstance(trainer.train_loader.dataset, Sized)
    assert isinstance(trainer.val_loader.dataset, Sized)
    assert isinstance(trainer.test_loader.dataset, Sized)

    assert len(trainer.train_loader.dataset) == 60
    assert len(trainer.val_loader.dataset) == 20
    assert len(trainer.test_loader.dataset) == 20

    assert isinstance(trainer.train_loader.batch_sampler.sampler, torch.utils.data.RandomSampler)
    assert isinstance(trainer.val_loader.sampler, torch.utils.data.SequentialSampler)
    assert isinstance(trainer.test_loader.sampler, torch.utils.data.SequentialSampler)


def test_decoder_trainer_dataset_is_empty(project_config):
    project_config.data = DataWithAutoSplit(
        dataset=cc(
            EmptyDataset,
        ),
        test_ratio=0.0,
        val_ratio=0.2,
    )
    with pytest.raises(ValueError, match="Training dataset is empty"):
        create_trainer_from_config(project_config)


def test_decoder_trainer_early_stopping_patience(project_config):
    with pytest.raises(ValidationError):
        project_config.trainer.early_stopping_patience = 0

    with pytest.raises(ValidationError):
        project_config.trainer.early_stopping_patience = -1

    project_config.trainer.early_stopping_patience = 1
    project_config.trainer.epochs = 3
    trainer = create_trainer_from_config(project_config)

    # Mock validation run to simulate increasing validation loss
    losses = [1.0, 2.0, 3.0]
    original_run = trainer.val_evaluator.run

    def mock_run(data=None, max_epochs=None, epoch_length=None):
        state = original_run(data, max_epochs, epoch_length)
        epoch = trainer.trainer.state.epoch
        trainer.val_evaluator.state.metrics["loss"] = losses[epoch - 1]
        return state

    with mock.patch.object(trainer.val_evaluator, "run", side_effect=mock_run):
        trainer.run()

    # Since patience is 1 and loss went 1.0 (epoch 1) -> 2.0 (epoch 2),
    # early stopping should trigger at the end of epoch 2, stopping the trainer.
    assert trainer.trainer.state.epoch == 2


def test_decoder_trainer_dataloader_class_collate_fn(project_config):
    project_config.model = cc(
        SimpleModel,
        vocab_size=10,
        hidden_size=8,
        collate_fn_target=target_path(DummyClassCollateFn),
    )
    trainer = create_trainer_from_config(project_config)
    assert trainer.train_loader is not None
    assert isinstance(trainer.train_loader.collate_fn, DummyClassCollateFn)
    assert isinstance(trainer.train_loader.collate_fn.tokenizer, DummyTokenizer)


# Inference param validation now lives on TrainerConfig (Field(gt=0)), so bad
# values (non-positive or non-int) are rejected at config construction.
@pytest.mark.parametrize(
    "kwargs",
    [
        {"inference_every_epochs": 0},
        {"max_inference_new_tokens": 0},
        {"inference_num_samples": 0},
        {"inference_every_epochs": -1},
        {"max_inference_new_tokens": -1},
        {"inference_num_samples": -1},
        {"inference_every_epochs": 2.5},
        {"inference_num_samples": "0.3"},
    ],
)
def test_invalid_inference_params_rejected(kwargs):
    with pytest.raises(ValidationError):
        TrainerConfig(**kwargs)


def test_setup_inference_and_log_success(project_config, temp_run_dir):
    project_config.trainer.inference_every_epochs = 1
    project_config.trainer.max_inference_new_tokens = 32
    project_config.model = cc(
        GenerativeModel,
        vocab_size=10,
        hidden_size=8,
        collate_fn_target="trainite.models.rope_transformer.CausalLMCollateFn",
    )
    transform = cc(DummyTransform)
    project_config.data.train.dataset = cc(
        GenerativeDataset,
        size=16,
        seq_len=4,
        vocab_size=10,
    )
    project_config.data.train.transform = transform
    project_config.data.val.dataset = cc(
        GenerativeDataset,
        size=8,
        seq_len=4,
        vocab_size=10,
    )
    project_config.data.val.transform = transform
    trainer = create_trainer_from_config(project_config)
    assert trainer.max_inference_new_tokens == 32
    trainer.run()


def test_decoder_trainer_grad_clip_norm(project_config):
    project_config.trainer.grad_clip_norm = 1.0
    trainer = create_trainer_from_config(project_config)
    with mock.patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
        trainer.run()
    assert mock_clip.called


def test_decoder_trainer_generate(project_config):
    trainer = create_trainer_from_config(project_config)
    trainer.model.eval()

    with mock.patch.object(trainer.model, "forward") as mock_forward:

        def mock_forward_fn(x, attention_mask=None):
            logits = torch.zeros(x.shape[0], x.shape[1], trainer.tokenizer.vocab_size, device=idist.device())
            logits[:, -1, 7] = 10.0
            return logits

        mock_forward.side_effect = mock_forward_fn

        input_ids = torch.tensor([[5, 6]], dtype=torch.long, device=idist.device())
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=idist.device())

        generated = trainer.generate(input_ids, max_new_tokens=1, attention_mask=attention_mask)
        assert isinstance(generated, torch.Tensor)
        assert generated[0].tolist() == [5, 6, 7]


@pytest.mark.parametrize(
    "precision,expected_mixed_precision",
    [
        ("float32", "no"),
        ("fp16", "fp16"),
        ("bf16", "bf16"),
    ],
)
def test_accelerator_initialization_precisions(project_config, precision, expected_mixed_precision):
    project_config.trainer.precision = precision
    trainer = create_trainer_from_config(project_config)
    assert trainer.accelerator.mixed_precision == expected_mixed_precision


def test_accelerator_backward_called_in_train_step(project_config):
    trainer = create_trainer_from_config(project_config)
    batch = {
        "input_ids": torch.randint(0, 10, (2, 4), device=trainer.device),
        "labels": torch.randint(0, 10, (2, 4), device=trainer.device),
    }
    with mock.patch.object(trainer.accelerator, "backward", wraps=trainer.accelerator.backward) as spy_backward:
        output = trainer._train_step(trainer.trainer, batch)
        assert spy_backward.called
        assert "loss" in output
        assert "logits" in output


def test_accelerator_clip_grad_norm_called(project_config):
    project_config.trainer.grad_clip_norm = 1.0
    trainer = create_trainer_from_config(project_config)
    batch = {
        "input_ids": torch.randint(0, 10, (2, 4), device=trainer.device),
        "labels": torch.randint(0, 10, (2, 4), device=trainer.device),
    }
    with mock.patch.object(trainer.accelerator, "clip_grad_norm_") as mock_clip:
        trainer._train_step(trainer.trainer, batch)
        mock_clip.assert_called_once()
        args, kwargs = mock_clip.call_args
        assert args[1] == 1.0


def test_checkpoint_uses_unwrapped_model(project_config):
    with (
        mock.patch("trainite.trainers.decoder_trainer.setup_best_model_checkpoint") as mock_best,
        mock.patch("trainite.trainers.decoder_trainer.setup_training_checkpointing") as mock_last,
    ):
        trainer = create_trainer_from_config(project_config)
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)

        assert mock_best.called
        to_save_best = mock_best.call_args[0][2]
        assert to_save_best["model"] is unwrapped

        assert mock_last.called
        to_save_last = mock_last.call_args[0][1]
        assert to_save_last["model"] is unwrapped


def test_eval_step_returns_expected_outputs(project_config):
    trainer = create_trainer_from_config(project_config)
    batch = {
        "input_ids": torch.randint(0, 10, (2, 4), device=trainer.device),
        "labels": torch.randint(0, 10, (2, 4), device=trainer.device),
    }
    output = trainer._eval_step(trainer.val_evaluator, batch)
    assert "logits" in output
    assert "targets" in output
    assert output["logits"].shape[:2] == (2, 4)


def test_run_dir_broadcast_object_list_called(project_config):
    with mock.patch("trainite.shared.utils.broadcast_object_list") as mock_broadcast:
        trainer = create_trainer_from_config(project_config)
        mock_broadcast.assert_called_once()
        args, kwargs = mock_broadcast.call_args
        assert args[0][0] == trainer.run_dir
        assert kwargs.get("from_process") == 0


def test_trainer_respects_user_selected_device(project_config):
    # When device="cpu", force_cpu is True and ACCELERATE_TORCH_DEVICE is cleared
    project_config.device = "cpu"
    trainer = create_trainer_from_config(project_config)
    assert os.environ.get("ACCELERATE_TORCH_DEVICE") is None
    assert trainer.device.type == "cpu"

    # When device="cuda:1", ACCELERATE_TORCH_DEVICE is set during init and restored after
    project_config.device = "cuda:1"
    with mock.patch("trainite.shared.utils.Accelerator") as mock_accel_cls:
        mock_accel = mock.MagicMock()
        mock_accel.device = torch.device("cpu")
        mock_accel.is_main_process = True
        mock_accel.use_distributed = False
        mock_accel.prepare.side_effect = lambda *args: args

        def check_env(*args, **kwargs):
            assert os.environ.get("ACCELERATE_TORCH_DEVICE") == "cuda:1"
            return mock_accel

        mock_accel_cls.side_effect = check_env

        Trainer(project_config)
        mock_accel_cls.assert_called_once_with(cpu=False, mixed_precision="no")
        assert os.environ.get("ACCELERATE_TORCH_DEVICE") is None

    # In distributed mode (WORLD_SIZE > 1), setting a specific GPU device raises ValueError
    project_config.device = "cuda:1"
    with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
        with pytest.raises(ValueError, match="Choose distributed GPUs through the launcher"):
            Trainer(project_config)


def test_eval_step_calls_gather_for_metrics(project_config):
    trainer = create_trainer_from_config(project_config)
    batch = {
        "input_ids": torch.randint(0, 10, (2, 4), device=trainer.device),
        "labels": torch.randint(0, 10, (2, 4), device=trainer.device),
    }
    with mock.patch.object(
        trainer.accelerator, "gather_for_metrics", wraps=trainer.accelerator.gather_for_metrics
    ) as spy_gather:
        output = trainer._eval_step(trainer.val_evaluator, batch)
        spy_gather.assert_called_once()
        assert "logits" in output
        assert "targets" in output


def test_eval_step_deduplicates_uneven_batches(project_config):
    trainer = create_trainer_from_config(project_config)
    batch = {
        "input_ids": torch.randint(0, 10, (4, 6), device=trainer.device),
        "labels": torch.randint(0, 10, (4, 6), device=trainer.device),
    }

    # Simulate distributed gather where the last batch had 4 items due to even_batches=True padding,
    # but gather_for_metrics drops the duplicate items and returns 2 items.
    def mock_gather_for_metrics(tensors):
        logits, targets = tensors
        return logits[:2], targets[:2]

    with mock.patch.object(trainer.accelerator, "gather_for_metrics", side_effect=mock_gather_for_metrics):
        output = trainer._eval_step(trainer.val_evaluator, batch)
        assert output["logits"].shape[0] == 2
        assert output["targets"].shape[0] == 2
        assert output["logits"].shape[1:] == (6, 10)


def test_eval_step_pads_across_processes_in_distributed(project_config):
    trainer = create_trainer_from_config(project_config)
    batch = {
        "input_ids": torch.randint(0, 10, (2, 4), device=trainer.device),
        "labels": torch.randint(0, 10, (2, 4), device=trainer.device),
    }
    with (
        mock.patch("accelerate.Accelerator.use_distributed", new_callable=mock.PropertyMock(return_value=True)),
        mock.patch.object(
            trainer.accelerator, "pad_across_processes", wraps=trainer.accelerator.pad_across_processes
        ) as spy_pad,
    ):
        output = trainer._eval_step(trainer.val_evaluator, batch)
        assert spy_pad.call_count == 2
        assert spy_pad.call_args_list[0].kwargs.get("pad_index") == 0
        assert spy_pad.call_args_list[1].kwargs.get("pad_index") == -100
        assert "logits" in output
        assert "targets" in output


def test_decoder_trainer_distributed_two_workers(tmp_path):
    worker_script = """
import sys
from trainite.trainers.decoder_trainer import Trainer
from trainite.config.base import (
    ProjectConfig, ModelConfig, PreprocessorConfig, DatasetConfig,
    OutputConfig, OptimizerConfig, LossConfig,
    SplitConfig, DataLoaderConfig, TrainerConfig, DataConfigBase
)

def run():
    run_dir = sys.argv[1]
    cfg = ProjectConfig(
        project_name="dist_test",
        preprocessor=PreprocessorConfig(_target_="tests.trainers.decoder_trainer_test.DummyTokenizer"),
        model=ModelConfig(_target_="tests.trainers.decoder_trainer_test.SimpleModel"),
        optimizer=OptimizerConfig(_target_="torch.optim.SGD", lr=0.01),
        loss=LossConfig(_target_="torch.nn.CrossEntropyLoss"),
        data=DataConfigBase(
            train=SplitConfig(
                dataset=DatasetConfig(_target_="tests.trainers.decoder_trainer_test.SimpleDataset", size=8),
                dataloader=DataLoaderConfig(batch_size=4, num_workers=0)
            ),
            val=SplitConfig(
                dataset=DatasetConfig(_target_="tests.trainers.decoder_trainer_test.SimpleDataset", size=4),
                dataloader=DataLoaderConfig(batch_size=4, num_workers=0)
            ),
            test=SplitConfig(
                dataset=DatasetConfig(_target_="tests.trainers.decoder_trainer_test.SimpleDataset", size=4),
                dataloader=DataLoaderConfig(batch_size=4, num_workers=0)
            ),
        ),
        trainer=TrainerConfig(epochs=1, log_every_steps=1),
        output=OutputConfig(root=run_dir, run_name="run"),
        logger="tensorboard",
        device=None,
    )
    trainer = Trainer(cfg)
    trainer.run()
    if trainer.accelerator.is_main_process:
        assert (trainer.run_dir / "best.pt").exists()

if __name__ == "__main__":
    run()
"""
    script_path = tmp_path / "worker.py"
    script_path.write_text(worker_script)
    output_dir = tmp_path / "outputs"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        str(script_path),
        str(output_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Distributed run failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
