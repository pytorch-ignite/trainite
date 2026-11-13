import json
import subprocess
import sys

import pytest

from tests.generated_project_config_test import create_training_project, training_environment


@pytest.mark.distributed
@pytest.mark.skipif(
    sys.platform.startswith(("win", "darwin")), reason="Generated distributed training requires Linux CI"
)
@pytest.mark.parametrize("model", ["rope-transformer", "gemma4-moe"])
def test_generated_project_distributed(model, tmp_path):
    project_dir = tmp_path / "distributed-project"
    create_training_project(project_dir, model, "string-reverse", distributed=True)
    command = [
        "uv",
        "run",
        "--python",
        sys.executable,
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "verify_training.py",
        "config.yaml",
    ]
    try:
        subprocess.run(
            command,
            cwd=project_dir,
            env=training_environment(project_dir),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(f"Distributed {model} failed:\nSTDOUT:\n{error.stdout}\nSTDERR:\n{error.stderr}")
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Distributed {model} timed out:\n{error.stdout}\n{error.stderr}")
    reports = [json.loads((project_dir / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert [report["rank"] for report in reports] == [0, 1]
    assert all(report["iterations"] == 4 for report in reports)
    assert reports[0]["run_dir"] == reports[1]["run_dir"]
