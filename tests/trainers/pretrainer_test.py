import shutil
import tempfile
from pathlib import Path
from typing import Any, Sized
from unittest import mock

import pytest
import torch
import torch.nn as nn

from trainite.config import (
    ComponentConfig,
    DataConfig,
    DataLoaderConfig,
    OptimizerConfig,
    OutputConfig,
    ProjectConfig,
    SplitConfig,
)
from trainite.config.trainer import PreTrainerConfig
from trainite.trainers.pretrainer import PreTrainer


def cc(target: str | None = None, **kwargs: Any) -> ComponentConfig:
    """Helper to create ComponentConfig with extra arguments without type errors."""
    if target:
        kwargs["_target_"] = target
    return ComponentConfig.model_validate(kwargs)


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=10, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        return self.fc(self.embedding(x))


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=16, seq_len=4, vocab_size=10):
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
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("This dataset is empty")


def dummy_collate_fn(batch):
    return batch


@pytest.fixture
def temp_run_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def project_config(temp_run_dir):
    return ProjectConfig(
        model=cc(
            "tests.trainers.pretrainer_test.SimpleModel",
            vocab_size=10,
            hidden_size=8,
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-3),
        data=DataConfig(
            train=SplitConfig(
                dataset=cc(
                    "tests.trainers.pretrainer_test.SimpleDataset",
                    size=16,
                    seq_len=4,
                    vocab_size=10,
                ),
                dataloader=DataLoaderConfig(batch_size=4),
            ),
            val=SplitConfig(
                dataset=cc(
                    "tests.trainers.pretrainer_test.SimpleDataset",
                    size=8,
                    seq_len=4,
                    vocab_size=10,
                ),
                dataloader=DataLoaderConfig(batch_size=4),
            ),
        ),
        trainer=PreTrainerConfig(epochs=1, log_every_steps=1),
        output=OutputConfig(root=str(temp_run_dir), run_name="test_run"),
        device="auto",
    )


def test_flatten_loss():
    # Mock some data
    logits = torch.randn(2, 3, 5)  # B=2, S=3, V=5
    targets = torch.tensor([[1, 2, -100], [0, -100, 3]])

    trainer = PreTrainer.__new__(PreTrainer)
    trainer.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    output = {"logits": logits, "targets": targets}
    flat_logits, flat_targets = trainer._flatten_loss(output)

    assert flat_logits.shape == (4, 5)  # 6 tokens total, 2 are masked
    assert flat_targets.shape == (4,)
    assert (flat_targets == torch.tensor([1, 2, 0, 3])).all()


def test_exact_accuracy_transform():
    logits = torch.tensor(
        [
            [[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]],  # Preds: 0, 1, 0
            [[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]],  # Preds: 1, 0, 1
        ]
    )
    targets = torch.tensor(
        [
            [0, 1, -100],  # Correct if we ignore -100
            [1, 0, 0],  # Last one is wrong (pred 1, target 0)
        ]
    )

    trainer = PreTrainer.__new__(PreTrainer)
    trainer.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    output = {"logits": logits, "targets": targets}
    y_pred, y = trainer._exact_accuracy_transform(output)

    # Sequence 1: 0==0, 1==1, -100 is masked. All correct -> 1
    # Sequence 2: 1==1, 0==0, 1!=0. Not all correct -> 0
    assert (y_pred == torch.tensor([1, 0])).all()
    assert (y == torch.tensor([1, 1])).all()


def test_pretrainer_init(project_config):
    trainer = PreTrainer(project_config)
    assert trainer.epochs == 1
    assert isinstance(trainer.model, SimpleModel)
    assert trainer.train_loader is not None
    assert trainer.val_loader is not None
    assert len(trainer.train_loader) == 4  # 16 / 4
    assert len(trainer.val_loader) == 2  # 8 / 4


def test_device_auto_selection(project_config):
    trainer = PreTrainer(project_config)
    if isinstance(trainer.device, torch.device):
        device_str = trainer.device.type
    elif isinstance(trainer.device, str):
        device_str = trainer.device
    else:
        raise ValueError("trainer.device should be either torch.device or str")
    if torch.cuda.is_available():
        assert device_str == "cuda"
    else:
        assert device_str == "cpu"


def test_pretrainer_auto_vocab_size(project_config):
    # Remove vocab_size from model config
    model_conf = project_config.model.model_dump(by_alias=True)
    model_conf.pop("vocab_size", None)
    project_config.model = cc(**model_conf)

    # Ensure dataset has vocab_size
    trainer = PreTrainer(project_config)
    assert trainer.vocab_size == 10
    assert isinstance(trainer.model, SimpleModel)
    assert trainer.model.embedding.num_embeddings == 10


def test_pretrainer_vocab_size_mismatch(project_config):
    # Set model vocab_size smaller than dataset
    model_conf = project_config.model.model_dump(by_alias=True)
    model_conf["vocab_size"] = 5
    project_config.model = cc(**model_conf)

    with pytest.raises(ValueError, match="is smaller than the dataset vocabulary size"):
        PreTrainer(project_config)


def test_pretrainer_run_with_val(project_config, temp_run_dir):
    trainer = PreTrainer(project_config)
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

    # Check for handlers
    assert "early_stopping" in trainer.handlers
    assert "checkpoint_best" in trainer.handlers


def test_pretrainer_run_without_val(project_config, temp_run_dir):
    project_config.data.val = None
    trainer = PreTrainer(project_config)
    trainer.run()

    # Check if run directory was created
    run_dirs = list((temp_run_dir / "test_run").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "output.log").exists()
    assert (run_dir / "tensorboard").exists()

    # Check for checkpoints
    # ModelCheckpoint n_saved=1, so we expect at least one
    checkpoints = list(run_dir.glob("*.pt"))
    assert len(checkpoints) >= 1

    # Early stopping and best checkpoint should NOT be in handlers
    assert "early_stopping" not in trainer.handlers
    assert "checkpoint_best" not in trainer.handlers


def test_pretrainer_test_no_loader(project_config):
    # Ensure test split is None (default in fixture is None)
    project_config.data.test = None
    trainer = PreTrainer(project_config)
    trainer.run()

    with mock.patch.object(trainer.logger, "warning") as mock_warning:
        trainer.test()

    mock_warning.assert_called_with("No test loader provided. Skipping testing.")


def test_pretrainer_test_method(project_config, temp_run_dir):
    # Add test split to config
    project_config.data.test = SplitConfig(
        dataset=cc(
            "tests.trainers.pretrainer_test.SimpleDataset",
            size=4,
            seq_len=4,
            vocab_size=10,
        ),
        dataloader=DataLoaderConfig(batch_size=4),
    )

    trainer = PreTrainer(project_config)
    trainer.run()
    trainer.test()

    # Test evaluators should have run
    assert "loss" in trainer.test_evaluator.state.metrics
    assert "exact_accuracy" in trainer.test_evaluator.state.metrics


def test_pretrainer_test_without_val(project_config, temp_run_dir):
    # Remove validation split
    project_config.data.val = None
    # Add test split
    project_config.data.test = SplitConfig(
        dataset=cc(
            "tests.trainers.pretrainer_test.SimpleDataset",
            size=4,
            seq_len=4,
            vocab_size=10,
        ),
        dataloader=DataLoaderConfig(batch_size=4),
    )

    trainer = PreTrainer(project_config)
    trainer.run()

    # checkpoint_best should not exist, it should use checkpoint_last
    assert "checkpoint_best" not in trainer.handlers
    assert "checkpoint_last" in trainer.handlers

    with mock.patch("torch.load", side_effect=torch.load) as mock_load:
        trainer.test()

    # Verify that it loaded the last checkpoint
    last_checkpoint_path = trainer.handlers["checkpoint_last"].last_checkpoint
    mock_load.assert_any_call(
        last_checkpoint_path, map_location=trainer.device, weights_only=True
    )


def test_pretrainer_dataloader_collate_fn(project_config):
    project_config.data.train.dataloader.collate_fn = cc(
        "tests.trainers.pretrainer_test.dummy_collate_fn"
    )
    trainer = PreTrainer(project_config)
    assert trainer.train_loader is not None
    assert trainer.train_loader.collate_fn is dummy_collate_fn


def test_pretrainer_explicit_split_shuffle(project_config):
    project_config.data.train.dataloader.shuffle = True
    trainer = PreTrainer(project_config)
    assert trainer.train_loader is not None
    # PyTorch DataLoader uses RandomSampler when shuffle is True
    assert isinstance(trainer.train_loader.sampler, torch.utils.data.RandomSampler)


def test_pretrainer_builds_train_and_val_loaders_from_ratios(tmp_path):
    config = ProjectConfig(
        model=cc(
            "tests.trainers.pretrainer_test.SimpleModel",
            vocab_size=100,
            hidden_size=32,
        ),
        data=DataConfig(
            dataset=cc(
                "tests.trainers.pretrainer_test.SimpleDataset",
                size=100,
                seq_len=10,
                vocab_size=100,
            ),
            train_ratio=0.8,
            val_ratio=0.2,
        ),
        trainer=PreTrainerConfig(epochs=1),
        output=OutputConfig(root=str(tmp_path), run_name="test"),
    )

    trainer = PreTrainer(config)

    assert trainer.train_loader is not None
    assert trainer.val_loader is not None
    assert isinstance(trainer.train_loader.dataset, Sized)
    assert isinstance(trainer.val_loader.dataset, Sized)
    assert len(trainer.train_loader.dataset) == 80
    assert len(trainer.val_loader.dataset) == 20
    assert trainer.test_loader is None


def test_pretrainer_builds_train_val_and_test_loaders_from_ratios(tmp_path):
    config = ProjectConfig(
        model=cc(
            "tests.trainers.pretrainer_test.SimpleModel",
            vocab_size=100,
            hidden_size=32,
        ),
        data=DataConfig(
            dataset=cc(
                "tests.trainers.pretrainer_test.SimpleDataset",
                size=100,
                seq_len=10,
                vocab_size=100,
            ),
            train_ratio=0.6,
            val_ratio=0.2,
        ),
        trainer=PreTrainerConfig(epochs=1),
        output=OutputConfig(root=str(tmp_path), run_name="test"),
    )

    trainer = PreTrainer(config)

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


def test_pretrainer_dataset_is_empty(project_config):
    project_config.data = DataConfig(
        dataset=cc(
            "tests.trainers.pretrainer_test.EmptyDataset",
        ),
        train_ratio=0.8,
        val_ratio=0.2,
    )
    with pytest.raises(ValueError, match="Training dataset is empty"):
        PreTrainer(project_config)


def test_pretrainer_early_stopping_patience(project_config):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        project_config.trainer.early_stopping_patience = 0

    with pytest.raises(ValidationError):
        project_config.trainer.early_stopping_patience = -1

    project_config.trainer.early_stopping_patience = 1
    trainer = PreTrainer(project_config)
    trainer.run()
