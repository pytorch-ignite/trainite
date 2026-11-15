import copy
import inspect
import json
import logging
import os
import py_compile
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from trainite.cli.init import Init, init_project, generate_uv_project
from trainite.config.registry import (
    DATASET_SPECS,
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    DEFAULT_TRAINER,
    MODEL_SPECS,
    PREPROCESSOR_SPECS,
    TRAINER_SPECS,
)


def test_generated_pyproject_has_build_system() -> None:
    pyproject = generate_uv_project(
        name="demo",
        version="0.1.0",
        dependencies=["torch"],
    )
    assert 'build-backend = "setuptools.build_meta"' in pyproject
    assert "packages = []" in pyproject


def test_interactive_defaults_are_explicit(monkeypatch):
    from trainite.cli import init

    monkeypatch.setattr(
        init,
        "DATASET_CHOICES",
        tuple(reversed(init.DATASET_CHOICES)),
    )

    captured = {}

    def fake_prompt_choice(prompt, choices, default, instruction=None):
        captured[prompt] = default
        return default

    def fake_prompt_multi_choice(
        prompt,
        choices,
        default=None,
        instruction=None,
    ):
        captured[prompt] = default
        return ["basic-transformer", init.DEFAULT_MODEL]

    monkeypatch.setattr(init, "_prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(
        init,
        "_prompt_multi_choice",
        fake_prompt_multi_choice,
    )
    monkeypatch.setattr(
        init,
        "_prompt_text",
        lambda *args, **kwargs: args[1],
    )
    monkeypatch.setattr(
        init.questionary,
        "confirm",
        lambda *args, **kwargs: type(
            "Prompt",
            (),
            {"ask": lambda self: False},
        )(),
    )
    monkeypatch.setattr(
        init,
        "init_project",
        lambda config: None,
    )

    init.run_interactive_mode()

    assert captured["Model(s):"] == [DEFAULT_MODEL]
    assert captured["Dataset:"] == DEFAULT_DATASET
    assert captured["Trainer:"] == DEFAULT_TRAINER
    assert captured["Primary active model in config.yaml:"] == init.DEFAULT_MODEL


def test_default_constants_exist_in_registry():
    assert DEFAULT_MODEL in MODEL_SPECS
    assert DEFAULT_DATASET in DATASET_SPECS
    assert DEFAULT_TRAINER in TRAINER_SPECS


def test_primary_model_default_falls_back_when_default_not_selected(monkeypatch):
    from trainite.cli import init

    monkeypatch.setattr(init, "DEFAULT_MODEL", "not-selected")

    captured = {}

    def fake_prompt_choice(prompt, choices, default, instruction=None):
        captured[prompt] = default
        return default

    def fake_prompt_multi_choice(prompt, choices, default=None, instruction=None):
        return ["basic-transformer", "rope-transformer"]

    monkeypatch.setattr(init, "_prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(
        init,
        "_prompt_multi_choice",
        fake_prompt_multi_choice,
    )
    monkeypatch.setattr(
        init,
        "_prompt_text",
        lambda *args, **kwargs: args[1],
    )
    monkeypatch.setattr(
        init.questionary,
        "confirm",
        lambda *args, **kwargs: type(
            "Prompt",
            (),
            {"ask": lambda self: False},
        )(),
    )
    monkeypatch.setattr(init, "init_project", lambda config: None)

    init.run_interactive_mode()

    assert captured["Primary active model in config.yaml:"] == "basic-transformer"


def test_cli_init_defaults_are_explicit():
    config = Init()

    assert config.model == (DEFAULT_MODEL,)
    assert config.dataset == DEFAULT_DATASET
    assert config.trainer == DEFAULT_TRAINER


def get_valid_project_combinations():
    valid_combinations = []
    for model_name in MODEL_SPECS.keys():
        for dataset_name in DATASET_SPECS.keys():
            for trainer_name in TRAINER_SPECS.keys():
                valid_combinations.append(([model_name], dataset_name, trainer_name))

    all_models = list(MODEL_SPECS.keys())
    if len(all_models) > 1:
        for dataset_name in DATASET_SPECS.keys():
            for trainer_name in TRAINER_SPECS.keys():
                valid_combinations.append((all_models, dataset_name, trainer_name))
    return valid_combinations


@pytest.mark.parametrize(
    "models, dataset, trainer",
    get_valid_project_combinations(),
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

        # Check that target inside config.yaml is rewritten correctly
        # and points to primary model
        with open(project_dir / "config.yaml", "r") as f:
            generated_config = yaml.safe_load(f)

        assert generated_config["project_name"] == project_dir.name
        assert generated_config["output"]["run_name"] == (f"{models[0]}__{dataset}".replace("-", "_"))
        assert generated_config["model"]["_target_"].startswith("models.")
        assert generated_config["model"]["collate_fn_target"].startswith("models.")
        loss_symbol = "GemmaMoECausalLMLoss" if models[0] == "gemma4-moe" else "CausalLMCrossEntropyLoss"
        assert generated_config["loss"]["_target_"] == f"models.{model_specs[0].name}.{loss_symbol}"
        assert not (project_dir / "losses").exists()

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

        # Check dataset and transform targets
        is_hf_backed = dataset_spec.builder_symbol == "datasets.load_dataset"

        data_config = generated_config["data"]
        split_configs = (
            [data_config]
            if "dataset" in data_config
            else [data_config[split] for split in ("train", "val", "test") if data_config.get(split)]
        )

        for split_config in split_configs:
            if is_hf_backed:
                assert split_config["dataset"]["_target_"] == "datasets.load_dataset"
            else:
                assert split_config["dataset"]["_target_"].startswith("dataset_impl.")

            assert split_config["transform"]["_target_"].startswith("dataset_impl.")

        # Check Hugging Face-specific configuration
        if is_hf_backed:
            assert generated_config["preprocessor"]["_target_"] == ("preprocessors.gpt2_tokenizer.load_gpt2_tokenizer")

            generated_pyproject = (project_dir / "pyproject.toml").read_text()
            assert "datasets" in generated_pyproject
            assert "transformers" in generated_pyproject


TRAINING_CASES = [(model, dataset) for model in MODEL_SPECS for dataset in DATASET_SPECS if dataset != "hugging-face"]


def run_checked(command: list[str], project_dir: Path, timeout: int, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(command, cwd=project_dir, env=env, check=True, timeout=timeout, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        pytest.fail(f"{command} failed in {project_dir}:\nSTDOUT:\n{error.stdout}\nSTDERR:\n{error.stderr}")
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"{command} timed out after {timeout}s:\nSTDOUT:\n{error.stdout}\nSTDERR:\n{error.stderr}")


def create_training_project(project_dir: Path, model: str, dataset: str) -> None:
    run_checked(
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
            "decoder-trainer",
            str(project_dir),
        ],
        project_dir.parent,
        60,
    )
    config_path = project_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["device"] = "cpu"
    config["trainer"].update(epochs=1, log_every_steps=1)
    config["model"].update(hidden_size=8, num_layers=2 if model.startswith("gemma4") else 1)
    if model.startswith("gemma4"):
        config["model"].update(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            global_num_key_value_heads=1,
            global_head_dim=4,
            dense_intermediate_size=16,
            sliding_window=8,
        )
        if model == "gemma4-moe":
            config["model"].update(expert_dim=4, num_experts=3, top_k=2)
    else:
        config["model"].update(feedforward_dim=16, dropout=0.0, max_seq_len=32)

    data = config["data"]
    if dataset == "string-reverse":
        data["dataset"].update(
            per_seq_size=10,
            min_seq_len=2,
            max_seq_len=3,
            charset="abcd",
        )
    elif dataset == "counting":
        # This range permits 27 distinct k=2 examples, enough for all three splits.
        data["dataset"].update(total_size=20, min_seq_len=3, max_seq_len=8, k=2)
    else:
        for split, size in (("train", 16 if "train" in data else 20), ("val", 4), ("test", 4)):
            if "dataset" in data and split != "train":
                continue
            if not data.get(split) and "dataset" not in data:
                data[split] = copy.deepcopy(data["val"])
            split_config = data if "dataset" in data else data[split]
            rows = []
            for index in range(size):
                text = "hello " + "world " * (1 + index % 3)
                rows.append(
                    {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": text}]}
                    if dataset == "ultrachat-200k"
                    else {"text": text}
                )
            fixture = project_dir / f"{split}.jsonl"
            fixture.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            split_config["dataset"].update(path="json", name=None, split="train", data_files=str(fixture))
            split_config["transform"]["max_length"] = 24

    if "dataset" in data:
        data["dataloader"].update(batch_size=4, num_workers=0)
        data.update(val_ratio=0.1, test_ratio=0.1)
    else:
        for split_config in data.values():
            split_config["dataloader"].update(batch_size=4, num_workers=0)
    config_path.write_text(yaml.safe_dump(config))
    (project_dir / "verify_training.py").write_text(
        inspect.getsource(verify_generated_training) + "\nverify_generated_training()\n"
    )
    run_checked(["uv", "sync", "--python", sys.executable], project_dir, 600, env=training_environment(project_dir))


def training_environment(project_dir: Path) -> dict[str, str]:
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
        },
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(project_dir / ".hf-cache"),
        "OMP_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }


def verify_generated_training() -> None:
    """Copied into the generated project to run with only its declared dependencies."""
    import importlib
    import math
    import sys
    from contextlib import nullcontext
    from pathlib import Path
    from unittest.mock import patch

    def local_tokenizer_loader():
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        vocabulary = {
            token: index
            for index, token in enumerate(
                ["<unk>", "<bos>", "<eos>", "<pad>", "hello", "world", "User", "Assistant", ":"]
            )
        }
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 2)]
        )
        fixture = Path("tokenizer-fixture")
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>", pad_token="<pad>"
        ).save_pretrained(fixture)
        original_loader = AutoTokenizer.from_pretrained

        def load(name, **kwargs):
            assert name == "openai-community/gpt2"
            return original_loader(str(fixture), local_files_only=True, **kwargs)

        return patch.object(AutoTokenizer, "from_pretrained", side_effect=load)

    import torch
    from ignite.engine import Events
    import yaml

    assert importlib.util.find_spec("trainite") is None
    main = importlib.import_module("main")
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    trainer_cls = main.Trainer
    trainers = []
    initial_parameters = {}
    losses = []

    def capture_trainer(config):
        trainer = trainer_cls(config)
        trainers.append(trainer)
        model = trainer.accelerator.unwrap_model(trainer.model)
        initial_parameters.update(
            {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        )

        def check_loss(engine):
            loss = float(engine.state.output["loss"])
            assert math.isfinite(loss), f"Non-finite training loss at iteration {engine.state.iteration}: {loss}"
            losses.append(loss)

        trainer.trainer.add_event_handler(Events.ITERATION_COMPLETED, check_loss)

        def check_training(engine):
            assert engine.state.epoch == 1
            assert engine.state.iteration == len(losses) == len(trainer.train_loader)
            assert len(losses) > 1
            assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
            assert any(
                not torch.equal(initial_parameters[name], parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )

        # Check updates before Trainer.run() reloads the best checkpoint.
        trainer.trainer.add_event_handler(Events.COMPLETED, check_training)
        return trainer

    main.Trainer = capture_trainer
    tokenizer_context = local_tokenizer_loader() if "gpt2" in config["preprocessor"]["_target_"] else nullcontext()
    with tokenizer_context:
        main.main()
    assert len(trainers) == 1
    trainer = trainers[0]
    assert trainer.trainer.state.epoch == 1
    assert trainer.trainer.state.iteration == len(losses) == len(trainer.train_loader)
    for evaluator in (trainer.train_evaluator, trainer.val_evaluator, trainer.test_evaluator):
        assert evaluator.state.iteration > 0
        assert math.isfinite(float(evaluator.state.metrics["loss"]))
        assert 0.0 <= float(evaluator.state.metrics["token_accuracy"]) <= 1.0

    model = trainer.accelerator.unwrap_model(trainer.model)
    assert any(
        not torch.equal(initial_parameters[name], parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    for filename in ("last.pt", "best.pt"):
        checkpoint = torch.load(trainer.run_dir / filename, map_location="cpu", weights_only=True)
        assert checkpoint["optimizer"]["state"], f"No optimizer updates in {filename}"
        assert checkpoint["model"]
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
        model.load_state_dict(checkpoint["model"], strict=True)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])

    # Corrupt weights so successful evaluation alone cannot hide a missing checkpoint reload.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    trainer.test()
    best = torch.load(trainer.run_dir / "best.pt", map_location="cpu", weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, best["model"][name], rtol=0, atol=0)
    assert trainer.test_evaluator.state.iteration == len(trainer.test_loader)
    assert math.isfinite(float(trainer.test_evaluator.state.metrics["loss"]))
    assert 0.0 <= float(trainer.test_evaluator.state.metrics["token_accuracy"]) <= 1.0


@pytest.mark.integration
@pytest.mark.parametrize(
    "model,dataset",
    [("rope-transformer", "string-reverse")] if os.environ.get("TRAINITE_INTEGRATION_SMOKE") == "1" else TRAINING_CASES,
    ids=lambda value: value,
)
def test_generated_project_is_runnable(model: str, dataset: str, tmp_path: Path) -> None:
    project_dir = tmp_path / "runnable-project"
    create_training_project(project_dir, model, dataset)
    try:
        subprocess.run(
            ["uv", "run", "--python", sys.executable, "python", "verify_training.py", "config.yaml"],
            cwd=project_dir,
            env=training_environment(project_dir),
            check=True,
            timeout=120,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(f"Training failed ({model} + {dataset}):\nSTDOUT:\n{error.stdout}\nSTDERR:\n{error.stderr}")
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Training timed out ({model} + {dataset}):\n{error.stdout}\n{error.stderr}")
    finally:
        logging.shutdown()


def test_cli_main_routing(capsys):
    from unittest import mock

    from trainite.cli.main import main

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
        "    if name in ('torch', 'ignite') or "
        "(package and package.startswith(('torch', 'ignite'))):\n"
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


def test_generated_project_grid_search_logic() -> None:
    """Verifies that generated utils.py and main.py support parameter sweeping."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "sweep-test-project"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "trainite.cli",
                "init",
                "--model",
                "rope-transformer",
                "--dataset",
                "counting",
                "--trainer",
                "decoder-trainer",
                str(project_dir),
            ],
            check=True,
        )

        generated_utils = (project_dir / "utils.py").read_text()
        generated_main = (project_dir / "main.py").read_text()

        assert "def load_grid_configs" in generated_utils
        assert "load_grid_configs" in generated_main


def _recreation_command(project_dir: Path) -> str:
    line = next(line for line in (project_dir / "README.md").read_text().splitlines() if "trainite init" in line)
    return line.lstrip("> ").strip()


def test_recreation_command_quotes_values_containing_spaces(tmp_path):
    project_dir = tmp_path / "quoted-experiment"
    init_project(Init(project_dir=str(project_dir), run_name="my great run", output_root="my outputs"))

    readme = (project_dir / "README.md").read_text()
    assert "Windows Command Prompt (`cmd.exe`)" in readme
    assert "replace single quotes around argument values with double quotes" in readme

    command = _recreation_command(project_dir)

    assert "--run-name 'my great run'" in command
    assert "--output-root 'my outputs'" in command


@pytest.mark.parametrize("run_name", ["", "plain_run"])
def test_recreation_command_leaves_simple_values_unquoted(tmp_path, run_name):
    project_dir = tmp_path / "plain-experiment"
    init_project(Init(project_dir=str(project_dir), run_name=run_name))

    readme = (project_dir / "README.md").read_text()
    assert "POSIX shell quoting" not in readme
    assert "Windows Command Prompt (`cmd.exe`)" not in readme

    command = _recreation_command(project_dir)

    if run_name:
        assert "--run-name plain_run" in command
    assert "'" not in command


def test_recreation_command_quoting_note_for_output_root(tmp_path):
    project_dir = tmp_path / "plain-project"
    init_project(Init(project_dir=str(project_dir), run_name="plain_run", output_root="my outputs"))

    readme = (project_dir / "README.md").read_text()
    assert "Windows Command Prompt (`cmd.exe`)" in readme


def test_recreation_command_is_replayable(tmp_path):
    source_dir = tmp_path / "replay-source"
    init_project(Init(project_dir=str(source_dir), run_name="my great run", output_root="my outputs"))

    # Drop the "trainite init <project-name>" prefix and replay the flags into a new directory.
    flags = shlex.split(_recreation_command(source_dir))[3:]
    replay_dir = tmp_path / "replayed"

    result = subprocess.run(
        [sys.executable, "-m", "trainite.cli", "init", *flags, str(replay_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "Unrecognized options" not in result.stdout + result.stderr
    assert (replay_dir / "config.yaml").exists()

    replayed_config = yaml.safe_load((replay_dir / "config.yaml").read_text())
    assert replayed_config["output"]["run_name"] == "my great run"
    assert replayed_config["output"]["root"] == "my outputs"


def test_primary_model_defaults_to_first_selected_model():
    config = Init(model=("rope-transformer", "basic-transformer"))

    assert config.primary_model == "rope-transformer"


def test_primary_model_not_in_selection_raises_error():
    with pytest.raises(ValueError, match="must be one of the selected models"):
        Init(model=("rope-transformer",), primary_model="basic-transformer")


def test_interactive_run_name_uses_non_first_primary_model(monkeypatch):
    from trainite.cli import init

    models = ("rope-transformer", "basic-transformer")
    monkeypatch.setattr(init, "_prompt_multi_choice", Mock(return_value=list(models)))
    monkeypatch.setattr(
        init,
        "_prompt_choice",
        Mock(side_effect=["basic-transformer", DEFAULT_DATASET, DEFAULT_TRAINER]),
    )
    monkeypatch.setattr(init, "_prompt_text", lambda prompt, default, instruction=None: default)
    confirmation = Mock()
    confirmation.ask.return_value = False
    monkeypatch.setattr(init.questionary, "confirm", Mock(return_value=confirmation))
    generate_project = Mock()
    monkeypatch.setattr(init, "init_project", generate_project)

    init.run_interactive_mode()

    generate_project.assert_called_once()
    config = generate_project.call_args.args[0]
    assert config.model == models
    assert config.primary_model == "basic-transformer"
    assert config.run_name == "basic_transformer__string_reverse"


def test_primary_model_drives_generated_config(tmp_path):
    project_dir = tmp_path / "primary-model-experiment"
    config = Init(
        project_dir=str(project_dir),
        model=("rope-transformer", "basic-transformer"),
        primary_model="basic-transformer",
    )
    init_project(config)

    generated_config = yaml.safe_load((project_dir / "config.yaml").read_text())

    # The primary model drives config.yaml and the run name even though it is
    # not first in the selection.
    assert "basic_transformer" in generated_config["model"]["_target_"]
    assert "basic_transformer" in generated_config["model"]["collate_fn_target"]
    assert generated_config["loss"]["_target_"] == "models.basic_transformer.CausalLMCrossEntropyLoss"
    assert generated_config["output"]["run_name"] == "basic_transformer__string_reverse"

    # Both selected models are still generated.
    assert (project_dir / "models/basic_transformer.py").exists()
    assert (project_dir / "models/rope_transformer.py").exists()
    assert "--primary-model basic-transformer" in (project_dir / "README.md").read_text()
