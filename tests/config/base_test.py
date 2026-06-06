import pytest
from pydantic import ValidationError
import optuna

from trainite.config.base import (
    ComponentConfig,
    DataConfig,
    SplitConfig,
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
    with pytest.raises(
        ValidationError,
        match="uses a range, but strategy 'grid' requires explicit lists",
    ):
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
            parameters={"model.num_heads": [2, 4]},
        )

    with pytest.raises(ValidationError, match="n_trials is required when strategy is"):
        SweepConfig(
            strategy="tpe",
            parameters={"model.num_heads": [2, 4]},
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
    from trainite.config.base import (
        ProjectConfig,
        TrainerConfig,
        OptimizerConfig,
        OutputConfig,
        DataConfig,
        ComponentConfig,
    )

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
    with pytest.raises(
        ValueError, match="nonexistent_field.*not found on OptimizerConfig"
    ):
        validate_sweep_params(base_config, sweep_invalid_path)


def test_validate_sweep_params_type_mismatch():
    from trainite.config.base import (
        ProjectConfig,
        TrainerConfig,
        OptimizerConfig,
        OutputConfig,
        DataConfig,
        ComponentConfig,
    )

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
    with pytest.raises(
        ValueError, match="Invalid value 'five' for parameter 'trainer.epochs'"
    ):
        validate_sweep_params(base_config, sweep_invalid_type)


def test_apply_overrides():
    from trainite.config.base import (
        ProjectConfig,
        TrainerConfig,
        OptimizerConfig,
        OutputConfig,
        DataConfig,
        ComponentConfig,
    )

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


def test_sweep_config_storage_default_none():
    config = SweepConfig(parameters={"optimizer.lr": [0.01, 0.001]})
    assert config.storage is None


def test_sweep_config_pruning_default_false():
    config = SweepConfig(parameters={"optimizer.lr": [0.01, 0.001]})
    assert config.pruning.enabled is False
    assert config.pruning.type == "nop"


def test_sweep_config_with_storage_and_pruning():
    config = SweepConfig(
        storage="sweep_study.db",
        pruning={
            "enabled": True,
            "type": "percentile",
            "percentile": 25.0,
            "n_startup_trials": 10,
        },
        parameters={"optimizer.lr": [0.01, 0.001]},
    )
    assert config.storage == "sweep_study.db"
    assert config.pruning.enabled is True
    assert config.pruning.type == "percentile"
    assert config.pruning.percentile == 25.0
    assert config.pruning.n_startup_trials == 10
    assert config.pruning.n_warmup_steps == 0


def test_all_pruner_types_and_factory():
    from trainite.sweep_utils import build_pruner
    import optuna.pruners

    # 1. Median
    config = SweepConfig(
        pruning={"enabled": True, "type": "median", "n_startup_trials": 10},
        parameters={"optimizer.lr": [0.01]},
    )
    pruner = build_pruner(config.pruning)
    assert isinstance(pruner, optuna.pruners.MedianPruner)

    # 2. Percentile
    config = SweepConfig(
        pruning={"enabled": True, "type": "percentile", "percentile": 30.0},
        parameters={"optimizer.lr": [0.01]},
    )
    pruner = build_pruner(config.pruning)
    assert isinstance(pruner, optuna.pruners.PercentilePruner)

    # 3. Hyperband
    config = SweepConfig(
        pruning={"enabled": True, "type": "hyperband", "min_resource": 2},
        parameters={"optimizer.lr": [0.01]},
    )
    pruner = build_pruner(config.pruning)
    assert isinstance(pruner, optuna.pruners.HyperbandPruner)

    # 4. Successive Halving
    config = SweepConfig(
        pruning={"enabled": True, "type": "successive_halving", "reduction_factor": 3},
        parameters={"optimizer.lr": [0.01]},
    )
    pruner = build_pruner(config.pruning)
    assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)

    # 5. Threshold
    config = SweepConfig(
        pruning={"enabled": True, "type": "threshold", "lower": 0.1, "upper": 0.9},
        parameters={"optimizer.lr": [0.01]},
    )
    pruner = build_pruner(config.pruning)
    assert isinstance(pruner, optuna.pruners.ThresholdPruner)

    # 6. Wilcoxon
    config = SweepConfig(
        pruning={"enabled": True, "type": "wilcoxon", "p_threshold": 0.05},
        parameters={"optimizer.lr": [0.01]},
    )
    pruner = build_pruner(config.pruning)
    assert isinstance(pruner, optuna.pruners.WilcoxonPruner)

    # 7. Patient
    config = SweepConfig(
        pruning={
            "enabled": True,
            "type": "patient",
            "patience": 3,
            "wrapped_type": "threshold",
            "lower": 0.2,
        },
        parameters={"optimizer.lr": [0.01]},
    )
    pruner = build_pruner(config.pruning)
    assert isinstance(pruner, optuna.pruners.PatientPruner)
    assert isinstance(pruner._wrapped_pruner, optuna.pruners.ThresholdPruner)

    # 8. Nop
    config = SweepConfig(
        pruning={"enabled": False}, parameters={"optimizer.lr": [0.01]}
    )
    pruner = build_pruner(config.pruning)
    assert isinstance(pruner, optuna.pruners.NopPruner)

    # Forbid extra fields on disabled pruning
    with pytest.raises(
        ValidationError,
        match="Pruning is disabled. The following parameters are not allowed: type",
    ):
        SweepConfig(
            pruning={"enabled": False, "type": "median"},
            parameters={"optimizer.lr": [0.01]},
        )


def test_all_sampler_types_and_factory():
    from trainite.sweep_utils import build_sampler
    import optuna.samplers

    strategies = [
        "grid",
        "random",
        "tpe",
        "brute_force",
        "cmaes",
        "gp",
        "nsgaii",
        "qmc",
    ]
    for strategy in strategies:
        n_trials = 10 if strategy not in ("grid", "brute_force") else None
        config = SweepConfig(
            strategy=strategy,
            n_trials=n_trials,
            parameters={"optimizer.lr": [0.01, 0.005]},
        )
        sampler = build_sampler(config, seed=42)
        assert isinstance(sampler, optuna.samplers.BaseSampler)


def test_sweep_config_validation_errors():
    from pydantic import ValidationError

    # Extra/incorrect fields in pruning
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SweepConfig(
            pruning={"enabled": True, "type": "median", "incorrect_field_name": 123},
            parameters={"optimizer.lr": [0.01]},
        )

    # Extra/incorrect fields in sweep
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SweepConfig(extra_field="invalid", parameters={"optimizer.lr": [0.01]})

    # Missing percentile when type is percentile
    with pytest.raises(
        ValidationError, match="percentile is required when type is 'percentile'"
    ):
        SweepConfig(
            pruning={"enabled": True, "type": "percentile"},
            parameters={"optimizer.lr": [0.01]},
        )

    # Invalid percentile value
    with pytest.raises(
        ValidationError, match="percentile must be between 0.0 and 100.0 exclusive"
    ):
        SweepConfig(
            pruning={"enabled": True, "type": "percentile", "percentile": 120.0},
            parameters={"optimizer.lr": [0.01]},
        )

    # Missing patience when type is patient
    with pytest.raises(
        ValidationError, match="patience is required when type is 'patient'"
    ):
        SweepConfig(
            pruning={"enabled": True, "type": "patient"},
            parameters={"optimizer.lr": [0.01]},
        )

    # Invalid patience value
    with pytest.raises(ValidationError, match="patience must be positive"):
        SweepConfig(
            pruning={"enabled": True, "type": "patient", "patience": -1},
            parameters={"optimizer.lr": [0.01]},
        )

    # Missing bounds when type is threshold
    with pytest.raises(
        ValidationError,
        match="At least one of 'lower' or 'upper' must be specified when type is 'threshold'",
    ):
        SweepConfig(
            pruning={"enabled": True, "type": "threshold"},
            parameters={"optimizer.lr": [0.01]},
        )

    # Invalid wilcoxon values
    with pytest.raises(
        ValidationError, match="p_threshold must be between 0.0 and 1.0 exclusive"
    ):
        SweepConfig(
            pruning={"enabled": True, "type": "wilcoxon", "p_threshold": 1.5},
            parameters={"optimizer.lr": [0.01]},
        )


def test_sweep_config_pruning_enabled_nop_warning(caplog):
    import logging
    from trainite.sweep_utils import validate_sweep_params
    from trainite.config.base import (
        ProjectConfig,
        TrainerConfig,
        OptimizerConfig,
        OutputConfig,
        DataConfig,
        ComponentConfig,
    )

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

    config = SweepConfig(
        pruning={"enabled": True, "type": "nop"}, parameters={"optimizer.lr": [0.01]}
    )

    with caplog.at_level(logging.WARNING, logger="sweep"):
        validate_sweep_params(base_config, config)

    assert len(caplog.records) == 1
    assert "WARNING: Pruning is enabled" in caplog.records[0].message
