from __future__ import annotations

import argparse
import inspect
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

from trainite.config import default_config, dump_config
from trainite.config.registry import (
    REGISTRY,
    get_dataset_spec,
    get_model_spec,
    get_trainer_spec,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent

MODEL_CHOICES = tuple(REGISTRY["models"].keys())
DATASET_CHOICES = tuple(REGISTRY["datasets"].keys())
TRAINER_CHOICES = tuple(REGISTRY["trainers"].keys())


def _replace_many(text: str, replacements: Iterable[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _render_template(path: Path, replacements: Iterable[tuple[str, str]] = ()) -> str:
    return _replace_many(path.read_text(), replacements)


def _render_class_source(cls: type) -> str:
    return textwrap.dedent(inspect.getsource(cls)).strip()


def _build_config_template(
    model_name: str, dataset_name: str, trainer_name: str
) -> str:
    model_spec = get_model_spec(model_name)
    dataset_spec = get_dataset_spec(dataset_name)
    trainer_spec = get_trainer_spec(trainer_name)
    model_source = textwrap.indent(_render_class_source(model_spec.config_cls), "")
    dataset_source = textwrap.indent(_render_class_source(dataset_spec.config_cls), "")
    trainer_source = textwrap.indent(_render_class_source(trainer_spec.config_cls), "")
    lines = [
        "from __future__ import annotations",
        "",
        "from pathlib import Path",
        "",
        "import yaml",
        "from pydantic import BaseModel, Field",
        "",
        model_source,
        "",
        dataset_source,
        "",
        trainer_source,
        "",
        "",
        "class OutputConfig(BaseModel):",
        '    root: str = "output"',
        f'    run_name: str = "{model_name}__{dataset_name}"',
        "",
        "",
        "class ProjectConfig(BaseModel):",
        f'    model_name: str = "{model_name}"',
        f'    dataset_name: str = "{dataset_name}"',
        f'    trainer_name: str = "{trainer_name}"',
        f"    model: {model_spec.config_cls.__name__} = Field(default_factory={model_spec.config_cls.__name__})",
        f"    dataset: {dataset_spec.config_cls.__name__} = Field(default_factory={dataset_spec.config_cls.__name__})",
        f"    trainer: {trainer_spec.config_cls.__name__} = Field(default_factory={trainer_spec.config_cls.__name__})",
        "    output: OutputConfig = Field(default_factory=OutputConfig)",
        "    seed: int = 42",
        "",
        "",
        "def load_yaml(path: str | Path) -> dict:",
        "    config_path = Path(path)",
        "    data = yaml.safe_load(config_path.read_text()) or {}",
        "    if not isinstance(data, dict):",
        '        raise ValueError("Config file must contain a mapping")',
        "    return data",
        "",
        "",
        "def dump_yaml(data: dict, path: str | Path) -> None:",
        "    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))",
        "",
        "",
        "def load_config(path: str | Path) -> ProjectConfig:",
        "    return ProjectConfig.model_validate(load_yaml(path))",
        "",
        "",
        "def dump_config(config: ProjectConfig, path: str | Path) -> None:",
        "    dump_yaml(config.model_dump(), path)",
        "",
        "",
        "def default_config() -> ProjectConfig:",
        "    return ProjectConfig()",
    ]
    return "\n".join(lines).strip() + "\n"


def _prompt_text(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def _prompt_choice(prompt: str, choices: Sequence[str], default: str) -> str:
    selected = _prompt_text(f"{prompt} ({' / '.join(choices)})", default)
    if selected not in choices:
        raise SystemExit(f"Unsupported choice: {selected}")
    return selected


def _project_directory(raw_project_dir: str, force: bool) -> Path:
    project_dir = Path(raw_project_dir).expanduser().resolve()
    if project_dir.exists():
        if not project_dir.is_dir():
            raise SystemExit(f"{project_dir} is not a directory")
        if any(project_dir.iterdir()) and not force:
            raise SystemExit(
                f"{project_dir} is not empty; choose an empty directory or pass --force to overwrite starter files"
            )
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def _write_file(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build_templates(
    project_name: str, model_name: str, dataset_name: str, trainer_name: str
) -> dict[str, str]:
    model_spec = get_model_spec(model_name)
    dataset_spec = get_dataset_spec(dataset_name)
    trainer_spec = get_trainer_spec(trainer_name)
    return {
        "config.py": _build_config_template(model_name, dataset_name, trainer_name),
        f"models/{model_spec.name}.py": _render_template(
            model_spec.implementation_path,
            [
                (
                    f"from trainite.config.model import {model_spec.config_cls.__name__}",
                    f"from config import {model_spec.config_cls.__name__}",
                ),
                (
                    model_spec.builder_symbol,
                    "build_model",
                ),
            ],
        ),
        f"dataset/{dataset_spec.name}.py": _render_template(
            dataset_spec.implementation_path,
            [
                (
                    f"from trainite.config.dataset import {dataset_spec.config_cls.__name__}",
                    f"from {project_name}.config import {dataset_spec.config_cls.__name__}",
                ),
                (
                    dataset_spec.builder_symbol,
                    "build_dataloaders",
                ),
            ],
        ),
        "trainer.py": _render_template(
            trainer_spec.implementation_path,
            [
                (
                    "from trainite.config import ProjectConfig, dump_config",
                    "from config import ProjectConfig, dump_config",
                ),
                (
                    f"from trainite.datasets import {dataset_spec.builder_symbol}",
                    f"from dataset.{dataset_spec.name} import build_dataloaders",
                ),
                (
                    f"from trainite.models import {model_spec.builder_symbol}",
                    f"from models.{model_spec.name} import build_model",
                ),
                ("class PreTrainer:", "class Trainer:"),
                (
                    f"{model_spec.builder_symbol}(config.model)",
                    "build_model(config.model)",
                ),
                (
                    f"{dataset_spec.builder_symbol}(config)",
                    "build_dataloaders(config.dataset)",
                ),
                ("PreTrainer", "Trainer"),
                ("trainite.utils", "utils"),
            ],
        ),
        "utils.py": _render_template(
            PROJECT_ROOT / "trainite/utils.py",
            [],
        ),
        "main.py": _render_template(
            PROJECT_ROOT / "main.py",
            [
                (
                    "from trainite.config import default_config, load_config",
                    "from config import default_config, load_config",
                ),
                (
                    "from trainite.trainers import PreTrainer",
                    "from trainer import Trainer",
                ),
                ("trainer = PreTrainer(config)", "trainer = Trainer(config)"),
            ],
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Generate a starter training project"
    )
    init_parser.add_argument(
        "project_dir", nargs="?", help="Directory to create the starter project in"
    )
    init_parser.add_argument(
        "--model", choices=MODEL_CHOICES, help="Starter model template to use"
    )
    init_parser.add_argument(
        "--dataset", choices=DATASET_CHOICES, help="Starter dataset template to use"
    )
    init_parser.add_argument("--output-root", help="Output root for generated config")
    init_parser.add_argument("--run-name", help="Run name for generated config")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing starter files",
    )
    init_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Use defaults for anything not provided and skip prompts",
    )
    init_parser.set_defaults(func=init_project)

    return parser


def init_project(args: argparse.Namespace) -> None:
    defaults = default_config()
    if args.project_dir:
        project_dir_input = args.project_dir
    elif args.yes:
        project_dir_input = "my-lm-experiment"
    else:
        project_dir_input = _prompt_text("Project directory", "my-lm-experiment")
    project_dir = _project_directory(project_dir_input, args.force)

    if args.yes:
        model_name = args.model or defaults.model_name
        dataset_name = args.dataset or defaults.dataset_name
        output_root = args.output_root or defaults.output.root
        run_name = args.run_name or f"{model_name}__{dataset_name}"
    else:
        model_name = args.model or _prompt_choice(
            "Model", MODEL_CHOICES, defaults.model_name
        )
        dataset_name = args.dataset or _prompt_choice(
            "Dataset", DATASET_CHOICES, defaults.dataset_name
        )
        output_root = args.output_root or _prompt_text(
            "Output directory", defaults.output.root
        )
        run_name = args.run_name or _prompt_text(
            "Run name", f"{model_name}__{dataset_name}"
        )

    starter_config = default_config()
    starter_config.output.root = output_root
    starter_config.output.run_name = run_name

    project_name = project_dir.name

    templates = _build_templates(project_name, model_name, dataset_name, defaults.trainer_name)
    dump_config(starter_config, project_dir / "config.yaml")
    for filename, content in templates.items():
        _write_file(project_dir / filename, content, args.force)

    print(f"Generated starter project in {project_dir}")
    for filename in ["config.yaml", *templates]:
        print(f"- {filename}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
