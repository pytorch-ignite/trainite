import pytest
from pydantic import ValidationError

from trainite.config.base import (
    ComponentConfig,
    DataConfigBase,
    SplitConfig,
)


def test_data_config_option_1():
    # Valid Option 1
    config = DataConfigBase(
        train=SplitConfig(dataset=ComponentConfig(_target_="dataset.train")),
        val=SplitConfig(dataset=ComponentConfig(_target_="dataset.val")),
    )
    assert config.train is not None
    assert config.val is not None
    assert config.dataset is None


def test_data_config_option_1_missing_train():
    # Invalid Option 1: missing train
    with pytest.raises(
        ValidationError,
        match="requires at least the 'train' split",
    ):
        DataConfigBase(val=SplitConfig(dataset=ComponentConfig(_target_="dataset.val")))


def test_data_config_option_2():
    # Valid Option 2
    config = DataConfigBase(
        dataset=ComponentConfig(_target_="dataset.all"), train_ratio=0.8, val_ratio=0.2
    )
    assert config.dataset is not None
    assert config.train_ratio == 0.8
    assert config.train is None


def test_data_config_option_2_missing_dataset():
    # Invalid Option 2: missing dataset (but having ratios)
    # Actually, if I provide just train_ratio, it might think it's Option 2 but missing dataset
    with pytest.raises(
        ValidationError,
        match="requires the 'dataset' field",
    ):
        DataConfigBase(train_ratio=0.8)


def test_data_config_mixed_modes_dataset_and_splits():
    # Mixed: dataset and train split
    with pytest.raises(
        ValidationError,
        match="when 'dataset' or 'dataloader' is provided",
    ):
        DataConfigBase(
            dataset=ComponentConfig(_target_="dataset.all"),
            train=SplitConfig(dataset=ComponentConfig(_target_="dataset.train")),
        )


def test_data_config_mixed_modes_ratios_and_splits():
    # Mixed: train_ratio and train split
    with pytest.raises(
        ValidationError,
        match="Cannot provide train_ratio or val_ratio at the data level when explicit splits are used",
    ):
        DataConfigBase(
            train_ratio=0.8,
            train=SplitConfig(dataset=ComponentConfig(_target_="dataset.train")),
        )


def test_data_config_invalid_ratios():
    with pytest.raises(
        ValidationError,
        match="Input should be greater than 0",
    ):
        DataConfigBase(dataset=ComponentConfig(_target_="dataset.all"), train_ratio=0.0)

    with pytest.raises(
        ValidationError,
        match="Input should be greater than 0",
    ):
        DataConfigBase(
            dataset=ComponentConfig(_target_="dataset.all"), train_ratio=-1.0
        )

    with pytest.raises(
        ValidationError,
        match="Input should be less than or equal to 1",
    ):
        DataConfigBase(dataset=ComponentConfig(_target_="dataset.all"), train_ratio=1.1)

    with pytest.raises(
        ValidationError,
        match="Input should be greater than or equal to 0",
    ):
        DataConfigBase(dataset=ComponentConfig(_target_="dataset.all"), val_ratio=-1.0)

    with pytest.raises(
        ValidationError,
        match="Input should be less than 1",
    ):
        DataConfigBase(dataset=ComponentConfig(_target_="dataset.all"), val_ratio=1.0)

    with pytest.raises(
        ValidationError,
        match="Input should be less than 1",
    ):
        DataConfigBase(dataset=ComponentConfig(_target_="dataset.all"), val_ratio=1.1)

    with pytest.raises(
        ValidationError,
        match="exceeds 1.0",
    ):
        DataConfigBase(
            dataset=ComponentConfig(_target_="dataset.all"),
            train_ratio=0.8,
            val_ratio=0.3,
        )

    with pytest.raises(
        ValidationError,
        match="exceeds 1.0",
    ):
        DataConfigBase(
            dataset=ComponentConfig(_target_="dataset.all"),
            val_ratio=0.2,
        )


def test_data_config_empty():
    with pytest.raises(
        ValidationError,
        match="Must provide either explicit splits",
    ):
        DataConfigBase()
