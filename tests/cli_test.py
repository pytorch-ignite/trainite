import logging
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path
import yaml

import pytest

from trainite.config.registry import MODEL_SPECS, DATASET_SPECS, PREPROCESSOR_SPECS, TRAINER_SPECS


@pytest.mark.parametrize(
    "model,dataset,trainer",
    [
        ("basic-transformer", "string-reverse", "decoder-trainer"),
        ("rope-transformer", "string-reverse", "decoder-trainer"),
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

        dataset_spec = DATASET_SPECS[dataset]
        model_spec = MODEL_SPECS[model]
        trainer_spec = TRAINER_SPECS[trainer]
        preprocessor_spec = (
            PREPROCESSOR_SPECS[dataset_spec.preprocessor_spec_name] if dataset_spec.preprocessor_spec_name else None
        )
        preprocessor_file = f"preprocessors/{preprocessor_spec.name}.py" if preprocessor_spec else None

        expected_files = [
            "config.yaml",
            "config.py",
            f"models/{model_spec.name}.py",
            f"datasets/{dataset_spec.name}.py",
            "datasets/transformed.py",
            "trainer.py",
            "utils.py",
            "main.py",
            "pyproject.toml",
            "README.md",
            preprocessor_file,
        ]
        for filename in expected_files:
            if filename is not None:
                assert (project_dir / filename).exists(), f"{filename} missing"

        # Check that targets inside config.yaml are rewritten correctly
        with open(project_dir / "config.yaml", "r") as f:
            generated_config = yaml.safe_load(f)
        assert generated_config["model"]["_target_"].startswith("models.")
        assert generated_config["model"]["collate_fn_target"].startswith("models.")

        # Check if python files are parseable
        python_files = [
            "config.py",
            f"models/{model_spec.name}.py",
            f"datasets/{dataset_spec.name}.py",
            "datasets/transformed.py",
            "trainer.py",
            "utils.py",
            "main.py",
            preprocessor_file,
        ]
        for filename in python_files:
            if filename is not None:
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
                "rope-transformer",
                "--dataset",
                "string-reverse",
                "--trainer",
                "decoder-trainer",
                str(project_dir),
            ],
            check=True,
        )

        # 2. Modify config.yaml to run for only 1 step/epoch to keep test fast
        config_path = project_dir / "config.yaml"

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
                timeout=300,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Generated project failed to run:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        except subprocess.TimeoutExpired:
            pytest.fail("Generated project timed out")

        finally:
            logging.shutdown()  # Ensure all logging output is flushed before the temporary directory is cleaned up


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
                "rope-transformer",
                "--dataset",
                "string-reverse",
                "--trainer",
                "decoder-trainer",
                "dummy_path",
            ]
        )
        mock_init_project.assert_called_once()


def test_import_without_dependencies() -> None:
    """
    Test that the trainite package and CLI main entrypoint can be imported
    successfully.
    """
    code = (
        "import importlib, sys\n"
        "orig_import = importlib.import_module\n"
        "def my_import(name, package=None):\n"
        "    if name in ('torch', 'ignite') or (package and package.startswith(('torch', 'ignite'))):\n"
        "        raise ImportError(f'Mocked ImportError for {name}')\n"
        "    return orig_import(name, package)\n"
        "importlib.import_module = my_import\n"
        "import trainite\n"
        "import trainite.cli.main\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert "is not a Python type" not in result.stderr
