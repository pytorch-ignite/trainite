import pytest
from pydantic import ValidationError

from trainite.config.base import (
    DataConfigBase,
    DataWithAutoSplit,
    SplitConfig,
    DatasetConfig,
)


class MockComponent(DatasetConfig):
    pass


def test_data_config_option_1():
    # Valid Option 1
    config = DataConfigBase(
        train=SplitConfig(dataset=MockComponent(_target_="dataset.train")),
        val=SplitConfig(dataset=MockComponent(_target_="dataset.val")),
    )
    assert config.train is not None
    assert config.val is not None
    assert config.test is None


def test_data_config_option_1_missing_train():
    # Invalid Option 1: missing train
    with pytest.raises(
        ValidationError,
        match="Field required",
    ):
        DataConfigBase(val=SplitConfig(dataset=MockComponent(_target_="dataset.val")))  # type: ignore


def test_data_config_option_2():
    # Valid Option 2
    config = DataWithAutoSplit(
        dataset=MockComponent(_target_="dataset.all"),
        test_ratio=0.2,
        val_ratio=0.1,
    )
    assert config.dataset is not None
    assert config.test_ratio == 0.2
    assert config.val_ratio == 0.1


def test_data_config_option_2_missing_dataset():
    # Invalid Option 2: missing dataset (but having ratios)
    with pytest.raises(
        ValidationError,
        match="Field required",
    ):
        DataWithAutoSplit(test_ratio=0.2, val_ratio=0.1)  # type: ignore


def test_data_config_invalid_ratios():
    with pytest.raises(
        ValidationError,
        match="greater than or equal to 0",
    ):
        DataWithAutoSplit(
            dataset=MockComponent(_target_="dataset.all"),
            test_ratio=-0.1,
        )

    with pytest.raises(
        ValidationError,
        match="less than 0.5",
    ):
        DataWithAutoSplit(
            dataset=MockComponent(_target_="dataset.all"),
            test_ratio=0.5,
        )

    with pytest.raises(
        ValidationError,
        match="greater than 0",
    ):
        DataWithAutoSplit(
            dataset=MockComponent(_target_="dataset.all"),
            val_ratio=0.0,
        )

    with pytest.raises(
        ValidationError,
        match="less than 0.3",
    ):
        DataWithAutoSplit(
            dataset=MockComponent(_target_="dataset.all"),
            val_ratio=0.3,
        )

    with pytest.raises(
        ValidationError,
        match="must be greater than 0.5",
    ):
        DataWithAutoSplit(
            dataset=MockComponent(_target_="dataset.all"),
            test_ratio=0.3,
            val_ratio=0.25,  # test_ratio + val_ratio = 0.55, train = 0.45 <= 0.5
        )

    with pytest.raises(
        ValidationError,
        match="must be less than or equal to 0.9",
    ):
        DataWithAutoSplit(
            dataset=MockComponent(_target_="dataset.all"),
            test_ratio=0.01,
            val_ratio=0.01,  # test_ratio + val_ratio = 0.02, train = 0.98 > 0.9
        )


def test_data_config_empty():
    with pytest.raises(
        ValidationError,
    ):
        DataConfigBase()  # type: ignore

    with pytest.raises(
        ValidationError,
    ):
        DataWithAutoSplit()  # type: ignore
