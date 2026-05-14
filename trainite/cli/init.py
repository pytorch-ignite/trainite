import argparse
import inspect
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

from trainite.config import OutputConfig, ProjectConfig, dump_config
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
    model_name: str, dataset_name: str, trainer_name: str
) -> dict[str, str]:
    model_spec = get_model_spec(model_name)
    dataset_spec = get_dataset_spec(dataset_name)
    trainer_spec = get_trainer_spec(trainer_name)
    return {
        f"models/{model_spec.name}.py": _render_template(
            model_spec.implementation_path,
            model_spec.template_replacements,
        ),
        f"dataset/{dataset_spec.name}.py": _render_template(
            dataset_spec.implementation_path,
            dataset_spec.template_replacements,
        ),
        "trainer.py": _render_template(
            trainer_spec.implementation_path,
            trainer_spec.template_replacements,
        ),
        "utils.py": _render_template(
            PROJECT_ROOT / "trainite/utils.py",
            [("trainite.config", "config")],
        ),
        "config.py": _render_template(PROJECT_ROOT / "trainite/config/base.py", []),
        "main.py": _render_template(
            PROJECT_ROOT / "trainite/main.py",
            [
                (
                    "trainite.config",
                    "config",
                ),
                (
                    "from trainite.trainers import PreTrainer",
                    f"from trainer import {trainer_spec.implementation_symbol}",
                ),
                (
                    "trainer = PreTrainer(config)",
                    f"trainer = {trainer_spec.implementation_symbol}(config)",
                ),
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
        "--trainer", choices=TRAINER_CHOICES, help="Starter trainer template to use"
    )
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
    if args.yes:
        project_dir = args.project_dir or "my-cool-experiment"
        model_name = args.model or MODEL_CHOICES[0]
        dataset_name = args.dataset or DATASET_CHOICES[0]
        trainer_name = args.trainer or TRAINER_CHOICES[0]
        output_root = args.output_root or "outputs"
        run_name = args.run_name or f"{model_name}__{dataset_name}"
    else:
        project_dir = args.project_dir or _prompt_text(
            "Project directory", "my-cool-experiment"
        )
        model_name = args.model or _prompt_choice(
            "Model", MODEL_CHOICES, MODEL_CHOICES[0]
        )
        dataset_name = args.dataset or _prompt_choice(
            "Dataset", DATASET_CHOICES, DATASET_CHOICES[0]
        )
        trainer_name = args.trainer or _prompt_choice(
            "Trainer", TRAINER_CHOICES, TRAINER_CHOICES[0]
        )
        output_root = args.output_root or _prompt_text("Output directory", "outputs")
        run_name = args.run_name or _prompt_text(
            "Run name", f"{model_name}__{dataset_name}"
        )

    project_dir = _project_directory(project_dir, args.force)

    output_config = OutputConfig(root=output_root, run_name=run_name)

    # Build templates for the starter project
    templates = _build_templates(model_name, dataset_name, trainer_name)

    model_spec = get_model_spec(model_name)
    dataset_spec = get_dataset_spec(dataset_name)
    trainer_spec = get_trainer_spec(trainer_name)

    # Update config to point to the correct builder functions for the model and dataset
    model_component = model_spec.config_cls()
    dataset_component = dataset_spec.config_cls()
    trainer_component = trainer_spec.config_cls()

    model_component.target = f"models.{model_spec.name}.{model_spec.builder_symbol}"
    dataset_component.target = (
        f"dataset.{dataset_spec.name}.{dataset_spec.builder_symbol}"
    )

    starter_config = ProjectConfig(
        model=model_component,
        dataset=dataset_component,
        trainer=trainer_component,
        output=output_config,
    )

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
