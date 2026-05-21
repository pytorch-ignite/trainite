from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path

from trainite.cli import main


def test_init_generates_parseable_starter_project() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "demo-project"
        main(["init", str(project_dir), "-y"])

        expected_files = [
            "config.yaml",
            "config.py",
            "trainer.py",
            "main.py",
            "utils.py",
        ]
        expected_dirs = [
            "models",
            "dataset",
        ]
        for filename in expected_files:
            assert (project_dir / filename).exists(), f"Missing file: {filename}"

        for dirname in expected_dirs:
            dir_path = project_dir / dirname
            assert dir_path.exists() and dir_path.is_dir(), f"Missing dir: {dirname}"
            py_files = list(dir_path.glob("*.py"))
            assert len(py_files) > 0, f"No .py files in {dirname}/"

        # Check all Python files are parseable
        for py_file in project_dir.rglob("*.py"):
            py_compile.compile(str(py_file), doraise=True)

def test_init_generated_trainer_has_no_trainite_imports() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "test-trainer"
        main(["init", str(project_dir), "-y"])

        trainer_text = (project_dir / "trainer.py").read_text()
        assert "trainite." not in trainer_text
