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


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str) -> list[int]:
        return [int(c) + 4 for c in text if c.isdigit()]

    def decode(
        self, ids: list[int] | torch.Tensor, skip_special_tokens=True, ignore_index=-100
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(str(i - 4) for i in ids if i >= 4)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=16, seq_len=4, vocab_size=10):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokenizer = MockTokenizer()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        src = "".join(
            str(x)
            for x in torch.randint(0, self.vocab_size - 4, (self.seq_len,)).tolist()
        )
        tgt = src[::-1]
        return {
            "source_text": src,
            "target_text": tgt,
        }


class PreTrainerTestCollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        input_ids_list = []
        labels_list = []
        for item in batch:
            src_ids = self.tokenizer.encode(item["source_text"])
            tgt_ids = self.tokenizer.encode(item["target_text"])

            bos_t = torch.tensor([self.tokenizer.bos_token_id])
            eos_t = torch.tensor([self.tokenizer.eos_token_id])
            src_t = torch.tensor(src_ids)
            tgt_t = torch.tensor(tgt_ids)

            src = torch.cat([bos_t, src_t, eos_t])
            tgt = torch.cat([tgt_t, eos_t])

            full_seq = torch.cat([src, tgt])

            input_ids = full_seq[:-1]
            labels = full_seq[1:].clone()
            labels[: len(src) - 1] = -100

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        padded_input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
        }


class SimpleDatasetWithDecode(SimpleDataset):
    def decode(self, ids, skip_special_tokens=True, ignore_index=-100):
        return self.tokenizer.decode(
            ids, skip_special_tokens=skip_special_tokens, ignore_index=ignore_index
        )


class SimpleModelWithGenerate(SimpleModel):
    def generate(self, prompt: str, max_new_tokens: int, tokenizer, eos_token_id=None):
        return "generated_output"


class EmptyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("This dataset is empty")


class DatasetWithInvalidType(torch.utils.data.Dataset):
    def __init__(self, tokenizer: object = None) -> None:
        self.tokenizer = MockTokenizer()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> str:
        return "not_a_dict"


class DatasetWithMissingKeys(torch.utils.data.Dataset):
    def __init__(self, tokenizer: object = None) -> None:
        self.tokenizer = MockTokenizer()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"source_text": "hello"}


class SimpleDatasetWithoutTokenizer(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"source_text": "hello", "target_text": "world"}


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
                dataloader=DataLoaderConfig(
                    batch_size=4,
                    collate_fn=cc(
                        "tests.trainers.pretrainer_test.PreTrainerTestCollateFn"
                    ),
                ),
            ),
            val=SplitConfig(
                dataset=cc(
                    "tests.trainers.pretrainer_test.SimpleDataset",
                    size=8,
                    seq_len=4,
                    vocab_size=10,
                ),
                dataloader=DataLoaderConfig(
                    batch_size=4,
                    collate_fn=cc(
                        "tests.trainers.pretrainer_test.PreTrainerTestCollateFn"
                    ),
                ),
            ),
        ),
        trainer=PreTrainerConfig(
            epochs=1, log_every_steps=1, inference_every_epochs=None
        ),
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
        dataloader=DataLoaderConfig(
            batch_size=4,
            collate_fn=cc("tests.trainers.pretrainer_test.PreTrainerTestCollateFn"),
        ),
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
        dataloader=DataLoaderConfig(
            batch_size=4,
            collate_fn=cc("tests.trainers.pretrainer_test.PreTrainerTestCollateFn"),
        ),
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
        trainer=PreTrainerConfig(epochs=1, inference_every_epochs=None),
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


def test_pretrainer_log_inference_validation(project_config):
    project_config.trainer.inference_every_epochs = 1

    # 1. SimpleModel has no generate, should raise ValueError
    with pytest.raises(
        ValueError,
        match="Model must implement 'generate' method for inference logging.",
    ):
        PreTrainer(project_config)

    # Use a model with generate method
    project_config.model = cc(
        "tests.trainers.pretrainer_test.SimpleModelWithGenerate",
        vocab_size=10,
        hidden_size=8,
    )

    # Use a dataset without tokenizer
    project_config.data.train.dataset = cc(
        "tests.trainers.pretrainer_test.SimpleDatasetWithoutTokenizer",
    )

    with pytest.raises(
        ValueError,
        match="Dataset must have a 'tokenizer' attribute for inference logging.",
    ):
        PreTrainer(project_config)


def test_pretrainer_log_inference(project_config):
    project_config.trainer.inference_every_epochs = 1
    project_config.trainer.inference_num_samples = 2
    project_config.trainer.max_inference_steps = 3

    project_config.model = cc(
        "tests.trainers.pretrainer_test.SimpleModelWithGenerate",
        vocab_size=10,
        hidden_size=8,
    )
    project_config.data.train.dataset = cc(
        "tests.trainers.pretrainer_test.SimpleDatasetWithDecode",
        size=16,
        seq_len=4,
        vocab_size=10,
    )
    project_config.data.val.dataset = cc(
        "tests.trainers.pretrainer_test.SimpleDatasetWithDecode",
        size=8,
        seq_len=4,
        vocab_size=10,
    )

    trainer = PreTrainer(project_config)

    with mock.patch.object(trainer.logger, "info") as mock_log_info:
        trainer._log_inference(trainer.engine, trainer.train_loader, "Train")

        # Verify that it logged sample predictions
        calls = [c[0][0] for c in mock_log_info.call_args_list]
        assert any("Running inference on Train samples" in call for call in calls)


def test_pretrainer_log_inference_validation_formats(project_config):
    project_config.trainer.inference_every_epochs = 1
    project_config.model = cc(
        "tests.trainers.pretrainer_test.SimpleModelWithGenerate",
        vocab_size=10,
        hidden_size=8,
    )

    # 1. Invalid item type (not a dict)
    project_config.data.train.dataset = cc(
        "tests.trainers.pretrainer_test.DatasetWithInvalidType",
    )
    with pytest.raises(
        ValueError,
        match="Train dataset items must be dictionaries",
    ):
        PreTrainer(project_config)

    # 2. Missing keys in training dataset items
    project_config.data.train.dataset = cc(
        "tests.trainers.pretrainer_test.DatasetWithMissingKeys",
    )
    with pytest.raises(
        ValueError,
        match="Train dataset items must contain 'source_text' and 'target_text' keys",
    ):
        PreTrainer(project_config)


def test_pretrainer_log_inference_tokenizer_fallback(project_config):
    project_config.trainer.inference_every_epochs = 1
    project_config.trainer.inference_num_samples = 2
    project_config.trainer.max_inference_steps = 3

    project_config.model = cc(
        "tests.trainers.pretrainer_test.SimpleModelWithGenerate",
        vocab_size=10,
        hidden_size=8,
    )
    # Training dataset has tokenizer
    project_config.data.train.dataset = cc(
        "tests.trainers.pretrainer_test.SimpleDatasetWithDecode",
        size=16,
        seq_len=4,
        vocab_size=10,
    )
    # Validation dataset does NOT have tokenizer
    project_config.data.val.dataset = cc(
        "tests.trainers.pretrainer_test.SimpleDatasetWithoutTokenizer",
    )

    # PreTrainer initialization should not raise any tokenizer error because of fallback
    trainer = PreTrainer(project_config)

    with mock.patch.object(trainer.logger, "info") as mock_log_info:
        assert trainer.val_loader is not None
        trainer._log_inference(trainer.engine, trainer.val_loader, "Val")

        # Verify that it ran inference on Val samples successfully without raising error
        calls = [c[0][0] for c in mock_log_info.call_args_list]
        assert any("Running inference on Val samples" in call for call in calls)
