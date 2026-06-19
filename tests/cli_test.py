import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from trainite.config.registry import get_dataset_spec, get_model_spec


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
        config["model"]["num_layers"] = 1
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
            pytest.fail(f"Generated project failed to run:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        except subprocess.TimeoutExpired:
            pytest.fail("Generated project timed out")


def test_cli_main_routing():
    from trainite.cli.main import main
    from unittest import mock

    with pytest.raises(SystemExit) as exc_info:
        main(argv=[])
    assert exc_info.value.code == 1

    with mock.patch("trainite.cli.main.run_interactive_mode") as mock_interactive:
        main(argv=["init"])
        mock_interactive.assert_called_once()

    with mock.patch("trainite.cli.main.init_project") as mock_init_project:
        main(
            argv=[
                "init",
                "--model",
                "transformer",
                "--dataset",
                "string-reverse",
                "--trainer",
                "pretrainer",
                "dummy_path",
            ]
        )
        mock_init_project.assert_called_once()
