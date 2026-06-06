import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import tomlkit
import yaml

from trainite.config import get_dataset_spec, get_model_spec


@pytest.mark.parametrize(
    "model,dataset,trainer",
    [
        ("transformer", "string-reverse", "pretrainer"),
    ],
)
def test_init_generates_valid_project(model: str, dataset: str, trainer: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "demo-project"
        # Run trainite init
        cmd = [
            sys.executable,
            "-m",
            "trainite.cli",
            "init",
            "--yes",
            "--model",
            model,
            "--dataset",
            dataset,
            "--trainer",
            trainer,
            str(project_dir),
        ]
        subprocess.run(cmd, check=True, timeout=60)

        dataset_spec = get_dataset_spec(dataset)
        model_spec = get_model_spec(model)

        expected_files = [
            "config.yaml",
            "config.py",
            f"models/{model_spec.name}.py",
            f"datasets/{dataset_spec.name}.py",
            "trainer.py",
            "utils.py",
            "main.py",
            "pyproject.toml",
            "README.md",
        ]
        for filename in expected_files:
            assert (project_dir / filename).exists(), f"{filename} missing"

        # Check if python files are parseable
        python_files = [
            "config.py",
            f"models/{model_spec.name}.py",
            f"datasets/{dataset_spec.name}.py",
            "trainer.py",
            "utils.py",
            "main.py",
        ]
        for filename in python_files:
            py_compile.compile(str(project_dir / filename), doraise=True)


def test_generated_string_reversal_project_is_runnable() -> None:
    """
    Test that the generated project can actually be run for a few steps.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "runnable-project"

        # 1. Generate project
        subprocess.run(
            [
                sys.executable,
                "-m",
                "trainite.cli",
                "init",
                "--yes",
                "--model",
                "transformer",
                "--dataset",
                "string-reverse",
                "--trainer",
                "pretrainer",
                str(project_dir),
            ],
            check=True,
        )

        # 2. Modify config.yaml to run for only 1 step/epoch to keep test fast
        config_path = project_dir / "config.yaml"
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["trainer"]["epochs"] = 1
        config["trainer"]["log_every_steps"] = 1
        config["model"]["layers"] = 1
        config["model"]["hidden_size"] = 16
        config["model"]["feedforward_dim"] = 32
        config["model"]["num_heads"] = 2
        config["data"]["dataset"]["per_seq_size"] = 16

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        # 3. Run the generated main.py

        try:
            subprocess.run(
                [sys.executable, "main.py", "config.yaml"],
                cwd=project_dir,
                check=True,
                timeout=60,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(
                f"Generated project failed to run:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Generated project timed out")


def test_cli_init_with_sweep() -> None:
    """Test that trainite init --sweep generates all sweep-related files and adds optuna dependency."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "sweep-project"
        cmd = [
            sys.executable,
            "-m",
            "trainite.cli",
            "init",
            "--yes",
            "--sweep",
            "--model",
            "transformer",
            "--dataset",
            "string-reverse",
            "--trainer",
            "pretrainer",
            str(project_dir),
        ]
        subprocess.run(cmd, check=True, timeout=60)

        # Expected project files (both core and sweep files)
        expected_files = [
            "config.yaml",
            "config.py",
            "models/transformer.py",
            "datasets/string_reverse.py",
            "trainer.py",
            "utils.py",
            "main.py",
            "pyproject.toml",
            "README.md",
            "sweep.yaml",
            "sweep.py",
            "sweep_utils.py",
            "sweep_config.py",
        ]

        for filename in expected_files:
            assert (project_dir / filename).exists(), f"{filename} missing"

        # Check if all python files are parseable
        python_files = [
            "config.py",
            "models/transformer.py",
            "datasets/string_reverse.py",
            "trainer.py",
            "utils.py",
            "main.py",
            "sweep.py",
            "sweep_utils.py",
            "sweep_config.py",
        ]
        for filename in python_files:
            py_compile.compile(str(project_dir / filename), doraise=True)

        # Check that optuna is in the dependencies of pyproject.toml
        pyproject_path = project_dir / "pyproject.toml"
        assert pyproject_path.exists()
        with open(pyproject_path, "r") as f:
            data = tomlkit.parse(f.read())
        deps = data["project"]["dependencies"]
        assert any("optuna" in dep.lower() for dep in deps), (
            "optuna missing from dependencies"
        )

        # Check README contains sweep documentation
        readme_path = project_dir / "README.md"
        assert readme_path.exists()
        readme_content = readme_path.read_text()
        assert "Hyperparameter Sweep" in readme_content
        assert "sweep.yaml" in readme_content
        assert 'strategy: "grid"' in readme_content


def test_cli_init_without_sweep() -> None:
    """Test that trainite init without --sweep does not generate sweep-related files or dependencies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "no-sweep-project"
        cmd = [
            sys.executable,
            "-m",
            "trainite.cli",
            "init",
            "--yes",
            "--model",
            "transformer",
            "--dataset",
            "string-reverse",
            "--trainer",
            "pretrainer",
            str(project_dir),
        ]
        subprocess.run(cmd, check=True, timeout=60)

        # Basic expected project files should exist
        expected_files = [
            "config.yaml",
            "config.py",
            "models/transformer.py",
            "datasets/string_reverse.py",
            "trainer.py",
            "utils.py",
            "main.py",
            "pyproject.toml",
            "README.md",
        ]
        for filename in expected_files:
            assert (project_dir / filename).exists(), f"{filename} missing"

        # Sweep expected project files should NOT exist
        expected_sweep_files = [
            "sweep.yaml",
            "sweep.py",
            "sweep_utils.py",
            "sweep_config.py",
        ]
        for filename in expected_sweep_files:
            assert not (project_dir / filename).exists(), f"{filename} should not exist"

        # Check that optuna is NOT in the dependencies of pyproject.toml
        pyproject_path = project_dir / "pyproject.toml"
        assert pyproject_path.exists()
        with open(pyproject_path, "r") as f:
            data = tomlkit.parse(f.read())
        deps = data["project"]["dependencies"]
        assert not any("optuna" in dep.lower() for dep in deps), (
            "optuna should not be in dependencies"
        )

        # Check README does NOT contain sweep documentation
        readme_path = project_dir / "README.md"
        assert readme_path.exists()
        readme_content = readme_path.read_text()
        assert "Hyperparameter Sweep" not in readme_content


def test_generated_sweep_is_runnable() -> None:
    """Test that the generated sweep.py runs successfully and saves summary results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "runnable-sweep-project"

        # 1. Initialize project with sweep
        subprocess.run(
            [
                sys.executable,
                "-m",
                "trainite.cli",
                "init",
                "--yes",
                "--sweep",
                "--model",
                "transformer",
                "--dataset",
                "string-reverse",
                "--trainer",
                "pretrainer",
                str(project_dir),
            ],
            check=True,
        )

        # 2. Modify config.yaml to run with tiny parameters for fast execution
        config_path = project_dir / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["trainer"]["epochs"] = 1
        config["trainer"]["log_every_steps"] = 1
        config["model"]["layers"] = 1
        config["model"]["hidden_size"] = 16
        config["model"]["feedforward_dim"] = 32
        config["model"]["num_heads"] = 2
        config["data"]["dataset"]["per_seq_size"] = 16

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        # 3. Modify sweep.yaml to sweep over minimal options to keep test fast
        sweep_yaml_path = project_dir / "sweep.yaml"
        sweep_config = {
            "strategy": "grid",
            "direction": "maximize",
            "metric": "exact_accuracy",
            "parameters": {
                "optimizer.lr": [0.01, 0.005]  # 2 combinations/trials
            },
        }
        with open(sweep_yaml_path, "w") as f:
            yaml.safe_dump(sweep_config, f)

        # 4. Run the generated sweep.py
        try:
            subprocess.run(
                [sys.executable, "sweep.py"],
                cwd=project_dir,
                check=True,
                timeout=90,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(
                f"Generated sweep.py failed to run:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Generated sweep.py timed out")

        # 5. Verify sweep output structure and files
        output_dir = project_dir / "outputs" / "transformer__string-reverse"
        assert output_dir.exists(), "Sweep output directory not created"

        # Find the sweep_run directory (starts with sweep_)
        sweep_run_dirs = list(output_dir.glob("sweep_*"))
        assert len(sweep_run_dirs) == 1, "Expected exactly one sweep directory"
        sweep_run_dir = sweep_run_dirs[0]

        # Verify that trial directories trial_0 and trial_1 were created (using prefix matching)
        trial_0_dirs = list(sweep_run_dir.glob("trial_0*"))
        trial_1_dirs = list(sweep_run_dir.glob("trial_1*"))
        assert len(trial_0_dirs) == 1, (
            f"Expected trial_0 directory, found: {trial_0_dirs}"
        )
        assert len(trial_1_dirs) == 1, (
            f"Expected trial_1 directory, found: {trial_1_dirs}"
        )
        trial_0_dir = trial_0_dirs[0]
        trial_1_dir = trial_1_dirs[0]

        # Verify checkpoint files exist under both trial directories (using recursive glob)
        assert len(list(trial_0_dir.glob("**/*.pt"))) >= 1
        assert len(list(trial_1_dir.glob("**/*.pt"))) >= 1

        # Verify that sweep_summary.yaml and sweep.yaml copy exist
        assert (sweep_run_dir / "sweep.yaml").exists()
        summary_path = sweep_run_dir / "sweep_summary.yaml"
        assert summary_path.exists(), "sweep_summary.yaml was not generated"

        with open(summary_path, "r") as f:
            summary_data = yaml.safe_load(f)

        assert "best_trial" in summary_data
        assert "all_trials" in summary_data
        assert len(summary_data["all_trials"]) == 2
        assert summary_data["all_trials"][0]["number"] == 0
        assert summary_data["all_trials"][1]["number"] == 1
        assert "value" in summary_data["best_trial"]
        assert "params" in summary_data["best_trial"]


def test_generated_sweep_with_pruning() -> None:
    """Test that a sweep with pruning enabled runs and records the trial state (COMPLETE or PRUNED)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "pruning-sweep-project"

        # 1. Initialize project with sweep
        subprocess.run(
            [
                sys.executable,
                "-m",
                "trainite.cli",
                "init",
                "--yes",
                "--sweep",
                "--model",
                "transformer",
                "--dataset",
                "string-reverse",
                "--trainer",
                "pretrainer",
                str(project_dir),
            ],
            check=True,
        )

        # 2. Modify config.yaml to run with tiny parameters for fast execution
        config_path = project_dir / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["trainer"]["epochs"] = 3  # multiple epochs so pruning can occur
        config["trainer"]["log_every_steps"] = 1
        config["model"]["layers"] = 1
        config["model"]["hidden_size"] = 16
        config["model"]["feedforward_dim"] = 32
        config["model"]["num_heads"] = 2
        config["data"]["dataset"]["per_seq_size"] = 16

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        # 3. Modify sweep.yaml to enable pruning with a median pruner and run 2 trials
        sweep_yaml_path = project_dir / "sweep.yaml"
        sweep_config = {
            "strategy": "grid",
            "direction": "maximize",
            "metric": "exact_accuracy",
            "pruning": {
                "enabled": True,
                "type": "median",
                "n_startup_trials": 1,  # Pruning can kick in after trial 0 completes
                "n_warmup_steps": 1,
            },
            "parameters": {"optimizer.lr": [0.01, 0.005]},
        }
        with open(sweep_yaml_path, "w") as f:
            yaml.safe_dump(sweep_config, f)

        # 4. Run sweep.py
        subprocess.run(
            [sys.executable, "sweep.py"],
            cwd=project_dir,
            check=True,
            timeout=90,
        )

        # 5. Verify sweep_summary.yaml contains trial states
        output_dir = project_dir / "outputs" / "transformer__string-reverse"
        sweep_run_dir = list(output_dir.glob("sweep_*"))[0]
        summary_path = sweep_run_dir / "sweep_summary.yaml"
        assert summary_path.exists()

        with open(summary_path, "r") as f:
            summary_data = yaml.safe_load(f)

        assert "all_trials" in summary_data
        assert len(summary_data["all_trials"]) == 2
        # Check that we have recorded state for each trial, e.g. "COMPLETE" or "PRUNED"
        for t in summary_data["all_trials"]:
            assert "state" in t
            assert t["state"] in ["COMPLETE", "PRUNED"]
