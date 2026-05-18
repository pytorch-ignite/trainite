from __future__ import annotations

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
            "utils.py",
            "models/transformer.py",
            "dataset/string-reverse.py",
            "trainer.py",
            "main.py",
        ]
        for filename in expected_files:
            assert (project_dir / filename).exists(), filename

        for filename in [
            "config.py",
            "utils.py",
            "models/transformer.py",
            "dataset/string-reverse.py",
            "trainer.py",
            "main.py",
        ]:
            py_compile.compile(str(project_dir / filename), doraise=True)

        config_text = (project_dir / "config.yaml").read_text()
        trainer_text = (project_dir / "trainer.py").read_text()
        main_text = (project_dir / "main.py").read_text()

        assert "vocab_size: ${dataset.vocab_size}" in config_text
        assert "vocab_size: 26" in config_text
        assert "trainer:\n  device:" in config_text
        assert "from config import ProjectConfig, dump_config" in trainer_text
        assert "from utils import instantiate" in trainer_text
        assert "from trainer import PreTrainer" in main_text
        assert "trainite." not in trainer_text
        assert "trainite." not in main_text
