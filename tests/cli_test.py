import logging
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path
from trainite.cli.init import Init, init_project
import yaml

import pytest

from trainite.config.registry import MODEL_SPECS, DATASET_SPECS, PREPROCESSOR_SPECS, TRAINER_SPECS


@pytest.mark.parametrize(
    "models,dataset,trainer",
    [
        (["basic-transformer"], "string-reverse", "decoder-trainer"),
        (["rope-transformer"], "string-reverse", "decoder-trainer"),
        (["rope-transformer", "basic-transformer"], "string-reverse", "decoder-trainer"),
        (["basic-transformer"], "counting", "decoder-trainer"),
        (["rope-transformer"], "counting", "decoder-trainer"),
        (["rope-transformer", "basic-transformer"], "counting", "decoder-trainer"),
        (["rope-transformer"], "hugging-face", "decoder-trainer"),
    ],
)
def test_init_generates_valid_project(models: list[str], dataset: str, trainer: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "demo-project"
        # Run trainite init
        cmd = [
            sys.executable,
            "-m",
            "trainite.cli",
            "init",
            "--model",
            *models,
            "--dataset",
            dataset,
            "--trainer",
            trainer,
            str(project_dir),
        ]
        subprocess.run(cmd, check=True, timeout=60)

        dataset_spec = DATASET_SPECS[dataset]
        model_specs = [MODEL_SPECS[m] for m in models]
        trainer_spec = TRAINER_SPECS[trainer]
        preprocessor_spec = (
            PREPROCESSOR_SPECS[dataset_spec.preprocessor_spec_name] if dataset_spec.preprocessor_spec_name else None
        )
        preprocessor_file = f"preprocessors/{preprocessor_spec.name}.py" if preprocessor_spec else None

        expected_files = [
            "config.yaml",
            "config.py",
            *[f"models/{spec.name}.py" for spec in model_specs],
            f"dataset_impl/{dataset_spec.name}.py",
            "dataset_impl/transformed.py",
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

        # Check that target inside config.yaml is rewritten correctly and points to primary model
        with open(project_dir / "config.yaml", "r") as f:
            generated_config = yaml.safe_load(f)
        assert generated_config["project_name"] == project_dir.name
        assert generated_config["output"]["run_name"] == f"{models[0]}__{dataset}".replace("-", "_")
        assert generated_config["model"]["_target_"].startswith("models.")
        assert generated_config["model"]["collate_fn_target"].startswith("models.")

        # Check if python files are parseable
        python_files = [
            "config.py",
            *[f"models/{spec.name}.py" for spec in model_specs],
            f"dataset_impl/{dataset_spec.name}.py",
            "dataset_impl/transformed.py",
            "trainer.py",
            "utils.py",
            "main.py",
            preprocessor_file,
        ]
        for filename in python_files:
            if filename is not None:
                py_compile.compile(str(project_dir / filename), doraise=True)

        if dataset == "hugging-face":
            assert generated_config["data"]["dataset"]["_target_"] == "datasets.load_dataset"
            assert generated_config["preprocessor"]["_target_"] == ("preprocessors.gpt2_tokenizer.load_gpt2_tokenizer")
            generated_pyproject = (project_dir / "pyproject.toml").read_text()
            assert "datasets" in generated_pyproject
            assert "transformers" in generated_pyproject
        else:
            assert generated_config["data"]["dataset"]["_target_"].startswith("dataset_impl.")
        assert generated_config["data"]["transform"]["_target_"].startswith("dataset_impl.")


@pytest.mark.integration
def test_generated_project_is_runnable() -> None:
    """
    generate one representative project, install its own declared dependencies
    in an isolated uv environment, and run a minimal training workload.
    """
    model, dataset, trainer = "rope-transformer", "string-reverse", "decoder-trainer"

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
                model,
                "--dataset",
                dataset,
                "--trainer",
                trainer,
                str(project_dir),
            ],
            check=True,
        )

        # 2. Modify config.yaml to run a minimal workload
        config_path = project_dir / "config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["trainer"]["epochs"] = 1
        config["trainer"]["log_every_steps"] = 1

        config["model"]["num_layers"] = 1
        config["model"]["hidden_size"] = 16
        config["model"]["feedforward_dim"] = 32

        config["data"]["dataset"]["per_seq_size"] = 16
        config["data"]["dataloader"]["batch_size"] = 8
        config["data"]["test_ratio"] = 0.1
        config["data"]["val_ratio"] = 0.1

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        # 3. Install ONLY the generated project's dependencies
        try:
            subprocess.run(
                ["uv", "sync"],
                cwd=project_dir,
                check=True,
                timeout=600,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(
                f"uv sync failed for generated project ({model} + {dataset}) "
                f"-- likely a missing/incorrect dependency in generated pyproject.toml:\n"
                f"STDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"uv sync timed out for generated project ({model} + {dataset})")

        # 4. Run the generated project
        try:
            subprocess.run(
                ["uv", "run", "python", "main.py", "config.yaml"],
                cwd=project_dir,
                check=True,
                timeout=300,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(
                f"Generated project failed to run ({model} + {dataset}):\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"Generated project timed out ({model} + {dataset})")
        finally:
            logging.shutdown()


def test_cli_main_routing(capsys):
    from trainite.cli.main import main
    from unittest import mock

    with pytest.raises(SystemExit) as exc_info:
        main(argv=[])
    assert exc_info.value.code == 2

    main(argv=["--version"])
    output = capsys.readouterr()
    assert output.out.startswith("Trainite, https://github.com/pytorch-ignite/trainite/\nVersion: ")

    with mock.patch("trainite.cli.main.run_interactive_mode") as mock_interactive:
        main(argv=["init"])
        mock_interactive.assert_called_once()

    with mock.patch("trainite.cli.main.init_project") as mock_init_project:
        main(
            argv=[
                "init",
                "--model",
                "rope-transformer",
                "basic-transformer",
                "--dataset",
                "string-reverse",
                "--trainer",
                "decoder-trainer",
                "dummy_path",
            ]
        )
        mock_init_project.assert_called_once()

    with mock.patch("trainite.cli.main.init_sky") as mock_init_sky:
        main(argv=["add:sky", "--force"])
        mock_init_sky.assert_called_once()


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


def test_duplicate_models_raises_error():
    with pytest.raises(ValueError, match="Duplicate model entries are not allowed"):
        Init(model=("rope-transformer", "rope-transformer"))


def test_init_with_sky_flag(tmp_path):
    project_dir = tmp_path / "sky-experiment"
    config = Init(project_dir=str(project_dir), sky=True)
    init_project(config)

    assert (project_dir / "sky.yaml").exists()
    sky_content = (project_dir / "sky.yaml").read_text()
    assert 'name: "sky-experiment"' in sky_content
    assert "uv sync" in sky_content

    pyproject_content = (project_dir / "pyproject.toml").read_text()
    assert "skypilot" in pyproject_content


def test_init_without_sky_flag_default(tmp_path):
    project_dir = tmp_path / "no-sky-experiment"
    config = Init(project_dir=str(project_dir), sky=False)
    init_project(config)

    assert not (project_dir / "sky.yaml").exists()
    pyproject_content = (project_dir / "pyproject.toml").read_text()
    assert "skypilot" not in pyproject_content
