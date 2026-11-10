"""Body-rate probe: greedy completion of N files, scored without executing.

    PYTHONPATH=. .venv/bin/python probe_body_rate.py outputs/<run>/<timestamp> --mode docstring

The training-time benchmark scores only 2 samples, which is too few to compare
runs. This probes a larger, deterministic sample with the same scoring functions.

Two prompt modes:

``first_half``
    What ``Trainer._run_benchmark`` does: prompt with the first half of the file.
``docstring``
    The prompt the results table uses: cut the file right after the closing
    ``\"\"\"`` of the first top-level documented function, so the model has to write
    its body. Requires the raw file text, so it reads the underlying dataset.
``docstring_raw``
    Same cut, but walks the raw corpus from the first row, which is how the
    existing table numbers were produced.

Prompts are evenly spaced over the split, so two runs with the same data config
are scored on identical files. Writes <RUN_DIR>/body_probe_<split>_<mode>.json.
"""

import argparse
import ast
import json
from pathlib import Path

import torch

import trainer as trainer_mod
from config import ProjectConfig
from utils import load_config


def _raw_file_text(dataset, index: int) -> str:
    """Return the untransformed file text for one index of a split dataset."""
    node = dataset
    while hasattr(node, "indices"):
        index = node.indices[index]
        node = node.dataset
    return node.dataset[index]["text"]


def _docstring_cut(text: str) -> tuple[str, str] | None:
    """Cut a file after the first top-level function's docstring.

    Returns ``(prompt, entry_point)`` or ``None`` when the file has no documented
    top-level function.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = node.body[0] if node.body else None
        if not (isinstance(docstring, ast.Expr) and isinstance(docstring.value, ast.Constant)):
            continue
        if not isinstance(docstring.value.value, str):
            continue
        lines = text.splitlines(keepends=True)
        prompt = "".join(lines[: docstring.end_lineno])
        return prompt, node.name
    return None


def _build_items_raw(num_samples: int, dataset_path: str) -> list[dict]:
    """Walk the raw corpus from the start and cut at the first documented function.

    This mirrors the ad hoc probe the results table was measured with, so its
    numbers are comparable to the file-level rows already in RESULTS.md.
    """
    from datasets import load_dataset  # pyrefly: ignore[missing-import]

    from dataset_impl.python_file import extract_file

    items: list[dict] = []
    for row in load_dataset(dataset_path, split="train"):
        text = extract_file(row["text"])
        if not text:
            continue
        cut = _docstring_cut(text)
        if cut is None:
            continue
        items.append({"index": len(items), "prompt": cut[0], "entry_point": cut[1]})
        if len(items) >= num_samples:
            break
    return items


def _build_items(dataset, mode: str, num_samples: int) -> list[dict]:
    """Pick prompts evenly across the split, walking forward to the next eligible file."""
    size = len(dataset)  # type: ignore[arg-type]
    items: list[dict] = []
    seen: set[int] = set()
    for i in range(num_samples):
        anchor = round(i * (size - 1) / max(num_samples - 1, 1))
        for index in range(anchor, min(anchor + 500, size)):
            if index in seen:
                continue
            if mode == "first_half":
                item = dataset[index]
                entry_point = trainer_mod._prompt_entry_point(item.source)
                items.append({"index": index, "prompt": item.source, "entry_point": entry_point})
                seen.add(index)
                break
            cut = _docstring_cut(_raw_file_text(dataset, index))
            if cut is None:
                continue
            items.append({"index": index, "prompt": cut[0], "entry_point": cut[1]})
            seen.add(index)
            break
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Directory holding config.yaml and best.pt")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to RUN_DIR/best.pt")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Defaults to the run's benchmark budget")
    parser.add_argument("--split", choices=("val", "test", "train"), default="test")
    parser.add_argument("--mode", choices=("first_half", "docstring", "docstring_raw"), default="docstring")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = load_config(run_dir / "config.yaml", ProjectConfig)

    # Reuse the run directory instead of a fresh timestamped one.
    trainer_mod.create_run_dir = lambda config, accelerator=None: run_dir
    trainer = trainer_mod.Trainer(config)

    checkpoint = args.checkpoint or run_dir / "best.pt"
    state = torch.load(checkpoint, map_location=trainer.device, weights_only=True)
    trainer.accelerator.unwrap_model(trainer.model).load_state_dict(state["model"])
    trainer.logger.info("Scored %s", checkpoint)

    if args.mode == "docstring_raw":
        items = _build_items_raw(args.num_samples, config.data.dataset.path)
    else:
        dataset = (trainer.val_loader if args.split == "val" else trainer.test_loader).dataset
        if args.split == "train":
            dataset = trainer.train_loader.dataset
        items = _build_items(dataset, args.mode, args.num_samples)
    if len(items) < args.num_samples:
        trainer.logger.warning("only %d eligible files in the %s split", len(items), args.split)
    num_samples = len(items)

    pad_token_id = getattr(trainer.tokenizer, "pad_token_id", 0)
    results = []
    for start in range(0, num_samples, args.batch_size):
        chunk = items[start : start + args.batch_size]
        prompt_ids = [trainer.tokenizer.encode(item["prompt"], add_special_tokens=False) for item in chunk]
        prompt_ids = [torch.tensor(ids, dtype=torch.long) for ids in prompt_ids]
        input_ids = (
            torch.nn.utils.rnn.pad_sequence(
                [ids.flip(0) for ids in prompt_ids], batch_first=True, padding_value=pad_token_id
            )
            .flip(1)
            .to(trainer.device)
        )
        attention_mask = (
            torch.nn.utils.rnn.pad_sequence(
                [torch.ones_like(ids, dtype=torch.long).flip(0) for ids in prompt_ids],
                batch_first=True,
                padding_value=0,
            )
            .flip(1)
            .to(trainer.device)
        )
        sequences = trainer.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens or trainer.max_benchmark_new_tokens,
            attention_mask=attention_mask,
        )
        completions = [
            trainer_mod._truncate_completion(trainer.tokenizer.decode(tokens.tolist(), skip_special_tokens=True))
            for tokens in sequences[:, input_ids.shape[1] :]
        ]
        for item, completion in zip(chunk, completions):
            compiles, has_body = trainer_mod._compile_stats(item["prompt"], completion, item["entry_point"])
            results.append(
                {
                    "index": item["index"],
                    "entry_point": item["entry_point"],
                    "compiles": compiles,
                    "has_body": has_body,
                    "completion": completion,
                }
            )
        trainer.logger.info("scored %d/%d", len(results), num_samples)

    compile_rate = sum(r["compiles"] for r in results) / num_samples
    body_rate = sum(r["has_body"] for r in results) / num_samples
    trainer.logger.info(
        "body probe [%s/%s]: n=%d compile=%.2f body=%.2f", args.mode, args.split, num_samples, compile_rate, body_rate
    )

    out = run_dir / f"{args.tag or f'body_probe_{args.split}_{args.mode}'}.json"
    out.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "split": args.split,
                "mode": args.mode,
                "body_rate": body_rate,
                "compile_rate": compile_rate,
                "samples": results,
            },
            indent=2,
        )
    )
    trainer.logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
