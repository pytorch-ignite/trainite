"""Real-data smoke test; run explicitly, without starting training.

    python test_real_dataset.py
    python test_real_dataset.py --source-split train --dataset-size 1000

The default checks the first 1,000 actual source rows to keep the smoke test small.
"""

import argparse
from pathlib import Path

import torch

from dataset_impl.python_edu import parse_source
from config import ProjectConfig
from utils import build_dataloaders, instantiate, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--source-split", default="train[:1000]")
    parser.add_argument("--dataset-size", type=int, default=100)
    args = parser.parse_args()
    if args.dataset_size < 10:
        parser.error("--dataset-size must be at least 10 to exercise all three splits")
    config = load_config(args.config, ProjectConfig)
    config.data.dataset.split = args.source_split
    config.data.dataset.dataset_size = args.dataset_size
    config.data.dataloader.num_workers = 0
    tokenizer = instantiate(config.preprocessor)
    print(
        f"Loading {config.data.dataset.path}, source split {args.source_split}; "
        f"keeping at most {args.dataset_size} usable functions",
        flush=True,
    )
    loaders = build_dataloaders(config.data, tokenizer, config.seed, config.model.collate_fn_target)
    indices = []
    count = 0
    for name, loader in zip(("train", "val", "test"), loaders):
        assert loader is not None and len(loader.dataset) > 0, f"{name} split is empty"
        indices.append(set(loader.dataset.indices))
        for item in loader.dataset:
            parse_source(item.source + item.target)
            prompt = tokenizer.encode(item.source, add_special_tokens=False)
            body = tokenizer.encode(item.target, add_special_tokens=False)
            assert item.eval_input_ids.tolist() == prompt, "Inference prompt contains incorrect tokens"
            assert item.train_input_ids.tolist() == prompt + body
            assert len(prompt) + len(body) + 1 <= config.data.dataset.max_length
            expected = torch.tensor((prompt + body + [tokenizer.eos_token_id])[1:])
            expected[: len(prompt) - 1] = -100
            assert torch.equal(item.train_label_ids, expected), "Incorrect shifted or masked labels"
            assert (item.attention_mask == 1).all()
            count += 1
        batch = next(iter(loader))
        assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
        assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
        assert (batch["labels"] != -100).any(dim=1).all(), "Example has no supervised tokens"
        print(f"{name}: {len(loader.dataset)} functions; batch shape {tuple(batch['input_ids'].shape)}", flush=True)
    assert not (indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])
    assert len(set.union(*indices)) == count <= args.dataset_size
    sample = loaders[0].dataset[0]
    print(f"\nExample prompt:\n{sample.source}\nExpected completion:\n{sample.target}")
    print(f"PASS: checked {count} actual Python-Edu functions; no training or generated-code execution.")


if __name__ == "__main__":
    main()
