import py_compile
import tempfile
from pathlib import Path

from trainite.cli import main


def test_init_generates_parseable_starter_project() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "demo-project"
        main(["init", str(project_dir), "--yes"])

        expected_files = [
            "config.yaml",
            "config.py",
            "models/transformer.py",
            "dataset/string-reverse.py",
            "trainer.py",
            "main.py",
            "README.md",
            "pyproject.toml",
        ]
        for filename in expected_files:
            assert (project_dir / filename).exists(), filename

        for filename in [
            "config.py",
            "models/transformer.py",
            "dataset/string-reverse.py",
            "trainer.py",
            "main.py",
        ]:
            py_compile.compile(str(project_dir / filename), doraise=True)

        trainer_text = (project_dir / "trainer.py").read_text()
        main_text = (project_dir / "main.py").read_text()

        assert (
            "from config import ProjectConfig, SplitConfig, dump_config" in trainer_text
        )
        assert "from trainer import PreTrainer" in main_text
        assert "trainite." not in trainer_text
        assert "trainite." not in main_text
