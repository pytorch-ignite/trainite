import logging
import shutil
import tempfile
from pathlib import Path
from typing import Sized, Any
from unittest import mock

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from trainite.config import (
    DataConfigBase,
    DataLoaderConfig,
    DataWithAutoSplit,
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
from trainite.config import ProjectConfig, TrainerConfig
from ignite.engine import Events
from ignite.handlers import EarlyStopping
import ignite.distributed as idist


def create_trainer_from_config(config: ProjectConfig) -> Trainer:
    return Trainer(config)


class MockComponent(ModelConfig, PreprocessorConfig, DatasetConfig, TransformConfig):
    pass


def cc(target: str | None = None, **kwargs: object) -> MockComponent:
    """Helper to create MockComponent with extra arguments without type errors."""
    if target:
        kwargs["_target_"] = target
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
        preprocessor=cc("tests.trainers.decoder_trainer_test.DummyTokenizer"),
        model=cc(
            "tests.trainers.decoder_trainer_test.SimpleModel",
            vocab_size=10,
            hidden_size=8,
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-3),
        data=DataConfigBase(
            train=SplitConfig(
                dataset=cc(
                    "tests.trainers.decoder_trainer_test.SimpleDataset",
                    size=16,
                    seq_len=4,
                    vocab_size=10,
                ),
                dataloader=DataLoaderConfig(batch_size=4),
            ),
            val=SplitConfig(
                dataset=cc(
                    "tests.trainers.decoder_trainer_test.SimpleDataset",
                    size=8,
                    seq_len=4,
                    vocab_size=10,
                ),
                dataloader=DataLoaderConfig(batch_size=4),
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
            "tests.trainers.decoder_trainer_test.SimpleDataset",
            size=4,
            seq_len=4,
            vocab_size=10,
        ),
        dataloader=DataLoaderConfig(batch_size=4),
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
            "tests.trainers.decoder_trainer_test.SimpleDataset",
            size=4,
            seq_len=4,
            vocab_size=10,
        ),
        dataloader=DataLoaderConfig(batch_size=4),
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
    project_config.model.collate_fn_target = "tests.trainers.decoder_trainer_test.dummy_collate_fn"
    trainer = create_trainer_from_config(project_config)
    assert trainer.train_loader is not None
    assert trainer.train_loader.collate_fn is dummy_collate_fn


def test_decoder_trainer_explicit_split_shuffle(project_config):
    project_config.data.train.dataloader.shuffle = True
    trainer = create_trainer_from_config(project_config)
    assert trainer.train_loader is not None
    # PyTorch DataLoader uses RandomSampler when shuffle is True
    assert isinstance(trainer.train_loader.sampler, torch.utils.data.RandomSampler)


def test_decoder_trainer_builds_train_and_val_loaders_from_ratios(tmp_path):
    config = ProjectConfig(
        project_name="test_project",
        preprocessor=cc("tests.trainers.decoder_trainer_test.DummyTokenizer"),
        model=cc(
            "tests.trainers.decoder_trainer_test.SimpleModel",
            vocab_size=100,
            hidden_size=32,
        ),
        data=DataWithAutoSplit(
            dataset=cc(
                "tests.trainers.decoder_trainer_test.SimpleDataset",
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
        preprocessor=cc("tests.trainers.decoder_trainer_test.DummyTokenizer"),
        model=cc(
            "tests.trainers.decoder_trainer_test.SimpleModel",
            vocab_size=100,
            hidden_size=32,
        ),
        data=DataWithAutoSplit(
            dataset=cc(
                "tests.trainers.decoder_trainer_test.SimpleDataset",
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

    assert isinstance(trainer.train_loader.sampler, torch.utils.data.RandomSampler)
    assert isinstance(trainer.val_loader.sampler, torch.utils.data.SequentialSampler)
    assert isinstance(trainer.test_loader.sampler, torch.utils.data.SequentialSampler)


def test_decoder_trainer_dataset_is_empty(project_config):
    project_config.data = DataWithAutoSplit(
        dataset=cc(
            "tests.trainers.decoder_trainer_test.EmptyDataset",
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
        "tests.trainers.decoder_trainer_test.SimpleModel",
        vocab_size=10,
        hidden_size=8,
        collate_fn_target="tests.trainers.decoder_trainer_test.DummyClassCollateFn",
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
        "tests.trainers.decoder_trainer_test.GenerativeModel",
        vocab_size=10,
        hidden_size=8,
        collate_fn_target="trainite.models.rope_transformer.CausalLMCollateFn",
    )
    transform = cc("tests.trainers.decoder_trainer_test.DummyTransform")
    project_config.data.train.dataset = cc(
        "tests.trainers.decoder_trainer_test.GenerativeDataset",
        size=16,
        seq_len=4,
        vocab_size=10,
    )
    project_config.data.train.transform = transform
    project_config.data.val.dataset = cc(
        "tests.trainers.decoder_trainer_test.GenerativeDataset",
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
