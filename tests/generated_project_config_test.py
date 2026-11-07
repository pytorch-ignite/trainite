"""Small generated projects shared by nightly integration and distributed tests."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


TRAINING_CASES = [
    ("basic-transformer", "string-reverse"),
    ("rope-transformer", "string-reverse"),
    ("gemma4-dense", "string-reverse"),
    ("gemma4-moe", "string-reverse"),
    ("rope-transformer", "counting"),
    ("rope-transformer", "wikitext"),
    ("rope-transformer", "ultrachat-200k"),
    ("rope-transformer", "python-edu"),
]


def run_checked(command: list[str], project_dir: Path, timeout: int) -> None:
    try:
        subprocess.run(command, cwd=project_dir, check=True, timeout=timeout, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        pytest.fail(f"{command} failed in {project_dir}:\nSTDOUT:\n{error.stdout}\nSTDERR:\n{error.stderr}")
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"{command} timed out after {timeout}s:\nSTDOUT:\n{error.stdout}\nSTDERR:\n{error.stderr}")


def create_training_project(project_dir: Path, model: str, dataset: str, distributed: bool = False) -> None:
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
            per_seq_size=13 if distributed else 10,
            min_seq_len=2,
            max_seq_len=4 if distributed else 3,
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
    worker = Path(__file__).with_name("generated_project_worker_test.py")
    (project_dir / "verify_training.py").write_text(worker.read_text())
    run_checked(["uv", "sync", "--python", sys.executable], project_dir, 600)


def training_environment(project_dir: Path) -> dict[str, str]:
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(project_dir / ".hf-cache"),
        "OMP_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }
