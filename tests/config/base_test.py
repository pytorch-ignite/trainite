import pytest
from pydantic import ValidationError
import optuna

from trainite.config.base import (
    ComponentConfig,
    DataConfig,
    SplitConfig,
    ProjectConfig,
)
from trainite.config.sweep import SweepConfig, ParameterRange
from trainite.sweep_utils import (
    validate_sweep_params,
    apply_overrides,
    suggest_params,
)


def test_data_config_option_1():
    # Valid Option 1
    config = DataConfig(
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
        DataConfig(val=SplitConfig(dataset=ComponentConfig(_target_="dataset.val")))


def test_data_config_option_2():
    # Valid Option 2
    config = DataConfig(
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
        DataConfig(train_ratio=0.8)


def test_data_config_mixed_modes_dataset_and_splits():
    # Mixed: dataset and train split
    with pytest.raises(
        ValidationError,
        match="when 'dataset' or 'dataloader' is provided",
    ):
        DataConfig(
            dataset=ComponentConfig(_target_="dataset.all"),
            train=SplitConfig(dataset=ComponentConfig(_target_="dataset.train")),
        )


def test_data_config_mixed_modes_ratios_and_splits():
    # Mixed: train_ratio and train split
    with pytest.raises(
        ValidationError,
        match="Cannot provide train_ratio or val_ratio at the data level when explicit splits are used",
    ):
        DataConfig(
            train_ratio=0.8,
            train=SplitConfig(dataset=ComponentConfig(_target_="dataset.train")),
        )


def test_data_config_invalid_ratios():
    with pytest.raises(
        ValidationError,
        match="Input should be greater than 0",
    ):
        DataConfig(dataset=ComponentConfig(_target_="dataset.all"), train_ratio=0.0)

    with pytest.raises(
        ValidationError,
        match="Input should be greater than 0",
    ):
        DataConfig(dataset=ComponentConfig(_target_="dataset.all"), train_ratio=-1.0)

    with pytest.raises(
        ValidationError,
        match="Input should be less than or equal to 1",
    ):
        DataConfig(dataset=ComponentConfig(_target_="dataset.all"), train_ratio=1.1)

    with pytest.raises(
        ValidationError,
        match="Input should be greater than or equal to 0",
    ):
        DataConfig(dataset=ComponentConfig(_target_="dataset.all"), val_ratio=-1.0)

    with pytest.raises(
        ValidationError,
        match="Input should be less than 1",
    ):
        DataConfig(dataset=ComponentConfig(_target_="dataset.all"), val_ratio=1.0)

    with pytest.raises(
        ValidationError,
        match="Input should be less than 1",
    ):
        DataConfig(dataset=ComponentConfig(_target_="dataset.all"), val_ratio=1.1)

    with pytest.raises(
        ValidationError,
        match="exceeds 1.0",
    ):
        DataConfig(
            dataset=ComponentConfig(_target_="dataset.all"),
            train_ratio=0.8,
            val_ratio=0.3,
        )

    with pytest.raises(
        ValidationError,
        match="exceeds 1.0",
    ):
        DataConfig(
            dataset=ComponentConfig(_target_="dataset.all"),
            val_ratio=0.2,
        )


def test_data_config_empty():
    with pytest.raises(
        ValidationError,
        match="Must provide either explicit splits",
    ):
        DataConfig()


def test_sweep_config_valid_grid():
    # A valid grid sweep config
    config = SweepConfig(
        strategy="grid",
        direction="maximize",
        metric="exact_accuracy",
        parameters={
            "model.num_heads": [2, 4],
            "optimizer.lr": [0.01, 0.001],
        },
    )
    assert config.strategy == "grid"
    assert config.direction == "maximize"
    assert config.metric == "exact_accuracy"
    assert config.n_trials is None
    assert config.parameters == {
        "model.num_heads": [2, 4],
        "optimizer.lr": [0.01, 0.001],
    }


def test_sweep_config_grid_rejects_range():
    # Grid strategy should not accept ParameterRange
    with pytest.raises(ValidationError, match="uses a range, but strategy 'grid' requires explicit lists"):
        SweepConfig(
            strategy="grid",
            parameters={
                "optimizer.lr": ParameterRange(type="float", low=0.0001, high=0.01)
            },
        )


def test_sweep_config_random_tpe_requires_n_trials():
    # Random or TPE strategy must have n_trials defined
    with pytest.raises(ValidationError, match="n_trials is required when strategy is"):
        SweepConfig(
            strategy="random",
            parameters={
                "model.num_heads": [2, 4]
            },
        )

    with pytest.raises(ValidationError, match="n_trials is required when strategy is"):
        SweepConfig(
            strategy="tpe",
            parameters={
                "model.num_heads": [2, 4]
            },
        )


def test_sweep_config_range_validation():
    # Check invalid ranges
    with pytest.raises(ValidationError, match="low.*must be less than high"):
        ParameterRange(type="float", low=0.01, high=0.001)

    with pytest.raises(ValidationError, match="low must be positive when sample='log'"):
        ParameterRange(type="float", low=-0.01, high=0.01, sample="log")

    with pytest.raises(ValidationError, match="step must be positive"):
        ParameterRange(type="float", low=0.01, high=0.05, step=-0.1)


def test_validate_sweep_params_invalid_path():
    from trainite.config.base import ProjectConfig, TrainerConfig, OptimizerConfig, OutputConfig, DataConfig, ComponentConfig
    base_config = ProjectConfig(
        project_name="test-project",
        model=ComponentConfig(_target_="trainite.models.TransformerModel"),
        dataset="test-dataset",
        data=DataConfig(
            dataset=ComponentConfig(_target_="dataset.all"),
            train_ratio=0.8,
            val_ratio=0.2,
        ),
        trainer=TrainerConfig(epochs=5),
        optimizer=OptimizerConfig(lr=0.01),
        output=OutputConfig(root="outputs", run_name="test"),
    )

    # Valid sweep parameters
    sweep_config = SweepConfig(
        strategy="grid",
        parameters={
            "optimizer.lr": [0.01, 0.001],
            "trainer.epochs": [5, 10],
        },
    )
    # Should pass validation
    validate_sweep_params(base_config, sweep_config)

    # Invalid parameter path
    sweep_invalid_path = SweepConfig(
        strategy="grid",
        parameters={
            "optimizer.nonexistent_field": [0.01, 0.001],
        },
    )
    with pytest.raises(ValueError, match="nonexistent_field.*not found on OptimizerConfig"):
        validate_sweep_params(base_config, sweep_invalid_path)


def test_validate_sweep_params_type_mismatch():
    from trainite.config.base import ProjectConfig, TrainerConfig, OptimizerConfig, OutputConfig, DataConfig, ComponentConfig
    base_config = ProjectConfig(
        project_name="test-project",
        model=ComponentConfig(_target_="trainite.models.TransformerModel"),
        dataset="test-dataset",
        data=DataConfig(
            dataset=ComponentConfig(_target_="dataset.all"),
            train_ratio=0.8,
            val_ratio=0.2,
        ),
        trainer=TrainerConfig(epochs=5),
        optimizer=OptimizerConfig(lr=0.01),
        output=OutputConfig(root="outputs", run_name="test"),
    )

    # Invalid parameter value (type mismatch)
    sweep_invalid_type = SweepConfig(
        strategy="grid",
        parameters={
            "trainer.epochs": ["five", "ten"],
        },
    )
    with pytest.raises(ValueError, match="Invalid value 'five' for parameter 'trainer.epochs'"):
        validate_sweep_params(base_config, sweep_invalid_type)


def test_apply_overrides():
    from trainite.config.base import ProjectConfig, TrainerConfig, OptimizerConfig, OutputConfig, DataConfig, ComponentConfig
    base_config = ProjectConfig(
        project_name="test-project",
        model=ComponentConfig(_target_="trainite.models.TransformerModel"),
        dataset="test-dataset",
        data=DataConfig(
            dataset=ComponentConfig(_target_="dataset.all"),
            train_ratio=0.8,
            val_ratio=0.2,
        ),
        trainer=TrainerConfig(epochs=5),
        optimizer=OptimizerConfig(lr=0.01),
        output=OutputConfig(root="outputs", run_name="test"),
    )

    overrides = {
        "trainer.epochs": 20,
        "optimizer.lr": 0.005,
    }
    new_config = apply_overrides(base_config, overrides)

    # Original should be unchanged
    assert base_config.trainer.epochs == 5
    assert base_config.optimizer.lr == 0.01

    # New should have overrides
    assert new_config.trainer.epochs == 20
    assert new_config.optimizer.lr == 0.005


def test_suggest_params():
    parameters = {
        "categorical_param": [10, 20, 30],
        "int_param": ParameterRange(type="int", low=1.0, high=5.0),
        "float_param": ParameterRange(type="float", low=0.001, high=0.1, sample="log"),
    }

    def objective(trial):
        suggestions = suggest_params(trial, parameters)
        assert suggestions["categorical_param"] in [10, 20, 30]
        assert isinstance(suggestions["int_param"], int)
        assert 1 <= suggestions["int_param"] <= 5
        assert isinstance(suggestions["float_param"], float)
        assert 0.001 <= suggestions["float_param"] <= 0.1
        return 1.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
