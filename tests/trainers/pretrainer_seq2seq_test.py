import shutil
import tempfile
from pathlib import Path
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
from trainite.trainers.pretrainer_seq2seq import PreTrainer


class SimpleSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size=10, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_input_ids, decoder_input_ids):
        # Dummy encoder-decoder forward pass
        enc = self.embedding(encoder_input_ids)
        dec = self.embedding(decoder_input_ids)
        # Combine encoder context (mean) with decoder representations
        context = enc.mean(dim=1, keepdim=True)
        combined = dec + context
        return self.fc(combined)


class SimpleSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, size=16, seq_len=4, vocab_size=10):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "encoder_input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "decoder_input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "decoder_labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
        }


@pytest.fixture
def temp_run_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def seq2seq_project_config(temp_run_dir):
    return ProjectConfig(
        model=ComponentConfig(
            _target_="tests.trainers.pretrainer_seq2seq_test.SimpleSeq2SeqModel",
            vocab_size=10,
            hidden_size=8,
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-3),
        data=DataConfig(
            train=SplitConfig(
                dataset=ComponentConfig(
                    _target_="tests.trainers.pretrainer_seq2seq_test.SimpleSeq2SeqDataset",
                    size=16,
                    seq_len=4,
                    vocab_size=10,
                ),
                dataloader=DataLoaderConfig(batch_size=4),
            ),
            val=SplitConfig(
                dataset=ComponentConfig(
                    _target_="tests.trainers.pretrainer_seq2seq_test.SimpleSeq2SeqDataset",
                    size=8,
                    seq_len=4,
                    vocab_size=10,
                ),
                dataloader=DataLoaderConfig(batch_size=4),
            ),
        ),
        trainer=PreTrainerConfig(epochs=1, log_every_steps=1),
        output=OutputConfig(root=str(temp_run_dir), run_name="test_seq2seq_run"),
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


def test_seq2seq_pretrainer_init(seq2seq_project_config):
    trainer = PreTrainer(seq2seq_project_config)
    assert trainer.epochs == 1
    assert isinstance(trainer.model, SimpleSeq2SeqModel)
    assert len(trainer.train_loader) == 4  # 16 / 4
    assert len(trainer.val_loader) == 2  # 8 / 4


def test_device_auto_selection(seq2seq_project_config):
    trainer = PreTrainer(seq2seq_project_config)
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


def test_seq2seq_pretrainer_auto_vocab_size(seq2seq_project_config):
    # Remove vocab_size from model config
    model_conf = seq2seq_project_config.model.model_dump(by_alias=True)
    model_conf.pop("vocab_size", None)
    seq2seq_project_config.model = ComponentConfig(**model_conf)

    # Ensure dataset has vocab_size
    trainer = PreTrainer(seq2seq_project_config)
    assert trainer.vocab_size == 10
    assert trainer.model.embedding.num_embeddings == 10


def test_seq2seq_pretrainer_vocab_size_mismatch(seq2seq_project_config):
    # Set model vocab_size smaller than dataset
    model_conf = seq2seq_project_config.model.model_dump(by_alias=True)
    model_conf["vocab_size"] = 5
    seq2seq_project_config.model = ComponentConfig(**model_conf)

    with pytest.raises(ValueError, match="is smaller than the dataset vocabulary size"):
        PreTrainer(seq2seq_project_config)


def test_seq2seq_pretrainer_run_with_val(seq2seq_project_config, temp_run_dir):
    trainer = PreTrainer(seq2seq_project_config)
    trainer.run()

    # Check if run directory was created
    run_dirs = list((temp_run_dir / "test_seq2seq_run").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "output.log").exists()
    assert (run_dir / "tensorboard").exists()

    # Check checkpoints
    checkpoints = list(run_dir.glob("*.pt"))
    assert len(checkpoints) >= 2

    # Check handlers
    assert "early_stopping" in trainer.handlers
    assert "checkpoint_best" in trainer.handlers


def test_seq2seq_pretrainer_run_without_val(seq2seq_project_config, temp_run_dir):
    seq2seq_project_config.data.val = None
    trainer = PreTrainer(seq2seq_project_config)
    trainer.run()

    # Check if run directory was created
    run_dirs = list((temp_run_dir / "test_seq2seq_run").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "output.log").exists()
    assert (run_dir / "tensorboard").exists()

    checkpoints = list(run_dir.glob("*.pt"))
    assert len(checkpoints) >= 1

    assert "early_stopping" not in trainer.handlers
    assert "checkpoint_best" not in trainer.handlers


def test_seq2seq_pretrainer_test_no_loader(seq2seq_project_config):
    seq2seq_project_config.data.test = None
    trainer = PreTrainer(seq2seq_project_config)
    trainer.run()

    with mock.patch.object(trainer.logger, "warning") as mock_warning:
        trainer.test()

    mock_warning.assert_called_with("No test loader provided. Skipping testing.")


def test_seq2seq_pretrainer_test_method(seq2seq_project_config, temp_run_dir):
    # Add test split to config
    seq2seq_project_config.data.test = SplitConfig(
        dataset=ComponentConfig(
            _target_="tests.trainers.pretrainer_seq2seq_test.SimpleSeq2SeqDataset",
            size=4,
            seq_len=4,
            vocab_size=10,
        ),
        dataloader=DataLoaderConfig(batch_size=4),
    )

    trainer = PreTrainer(seq2seq_project_config)
    trainer.run()
    trainer.test()

    # Test evaluators should have run
    assert "loss" in trainer.test_evaluator.state.metrics
    assert "exact_accuracy" in trainer.test_evaluator.state.metrics


def test_seq2seq_pretrainer_test_without_val(seq2seq_project_config, temp_run_dir):
    # Remove validation split
    seq2seq_project_config.data.val = None
    # Add test split
    seq2seq_project_config.data.test = SplitConfig(
        dataset=ComponentConfig(
            _target_="tests.trainers.pretrainer_seq2seq_test.SimpleSeq2SeqDataset",
            size=4,
            seq_len=4,
            vocab_size=10,
        ),
        dataloader=DataLoaderConfig(batch_size=4),
    )

    trainer = PreTrainer(seq2seq_project_config)
    trainer.run()

    assert "checkpoint_best" not in trainer.handlers
    assert "checkpoint_last" in trainer.handlers

    with mock.patch("torch.load", side_effect=torch.load) as mock_load:
        trainer.test()

    last_checkpoint_path = trainer.handlers["checkpoint_last"].last_checkpoint
    mock_load.assert_any_call(
        last_checkpoint_path, map_location=trainer.device, weights_only=True
    )
