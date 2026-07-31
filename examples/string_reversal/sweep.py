import argparse
import copy
import sys
import time
from pathlib import Path


DEFAULT_SEQ_LENS = [8, 16, 32, 64, 96, 128, 160, 192, 224]
DEFAULT_SEEDS = [42, 123, 456]


DEPTH_LAYERS = [6, 8]
DEPTH_FIXED_HIDDEN = [32]
DEPTH_FIXED_HEADS = [2]

WIDTH_HIDDEN_HEADS = [
    (16, 1),
    (32, 2),
    (64, 4),
]
WIDTH_FIXED_LAYERS = [4]

PROJECT_NAME = "string-reverse-sweep"


def run_name(seq_len: int, seed: int, layers: int, hidden: int, heads: int) -> str:
    return f"sl{seq_len}_s{seed}_l{layers}_h{hidden}_nh{heads}"


def make_config(base, seq_len: int, seed: int, layers: int, hidden: int, heads: int, logger: str, project_name: str):
    cfg = copy.deepcopy(base)

    # Dataset — single fixed sequence length per run
    cfg.data.dataset.seq_len = seq_len
    cfg.data.dataset.min_seq_len = None
    cfg.data.dataset.max_seq_len = None
    cfg.data.dataset.seed = seed

    # Model
    cfg.model.num_layers = layers
    cfg.model.hidden_size = hidden
    cfg.model.num_heads = heads
    cfg.model.feedforward_dim = hidden * 2
    cfg.model.max_seq_len = seq_len * 3  # prompt + sep + target + eos headroom

    # Reproducibility
    cfg.seed = seed

    # Run identity — shared project, unique name per combo
    cfg.output.project = project_name
    cfg.output.run_name = run_name(seq_len, seed, layers, hidden, heads)
    cfg.logger = logger

    cfg.trainer.max_inference_new_tokens = seq_len + 2
    cfg.trainer.epochs = 9999  # rely on early stopping / time budget

    return cfg


def build_combos(mode: str) -> tuple[list[tuple], str]:
    if mode == "depth":
        layers_grid = DEPTH_LAYERS
        pairs = list(zip(DEPTH_FIXED_HIDDEN, DEPTH_FIXED_HEADS))
    elif mode == "width":
        layers_grid = WIDTH_FIXED_LAYERS
        pairs = WIDTH_HIDDEN_HEADS
    else:  # full
        layers_grid = DEPTH_LAYERS
        pairs = WIDTH_HIDDEN_HEADS

    return [
        (sl, seed, layers, hidden, heads)
        for sl in DEFAULT_SEQ_LENS
        for seed in DEFAULT_SEEDS
        for layers in layers_grid
        for hidden, heads in pairs
    ], PROJECT_NAME + f"-{mode}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "mode",
        choices=["depth", "width", "full"],
        help="which axis to sweep: depth (vary layers), width (vary hidden/heads), or full (cross-product)",
    )
    p.add_argument("--config", default="config.yaml", help="path to config.yaml (default: config.yaml)")
    p.add_argument(
        "--logger",
        default="tensorboard",
        choices=["tensorboard", "clearml"],
        help="logger backend (default: tensorboard)",
    )
    p.add_argument(
        "--time-budget",
        type=int,
        default=None,
        metavar="SECONDS",
        help="max wall-clock seconds per run (default: no limit)",
    )
    p.add_argument("--dry-run", action="store_true", help="print combos, don't train")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    combos, project_name = build_combos(args.mode)
    total = len(combos)
    print(f"Sweep [{args.mode}]: {total} runs  |  logger={args.logger}  |  project={PROJECT_NAME}\n")

    if args.dry_run:
        for i, (sl, seed, layers, hidden, heads) in enumerate(combos, 1):
            print(f"  [{i}/{total}] {run_name(sl, seed, layers, hidden, heads)}")
        return

    # Import here so --dry-run works without a full training env
    from ignite.engine import Events
    from utils import load_config
    from trainer import Trainer, ProjectConfig  # noqa: F401

    base = load_config(Path(args.config), ProjectConfig)

    for i, (sl, seed, layers, hidden, heads) in enumerate(combos, 1):
        name = run_name(sl, seed, layers, hidden, heads)
        print(f"[{i}/{total}] {name}")

        cfg = make_config(base, sl, seed, layers, hidden, heads, args.logger, project_name)
        try:
            trainer = Trainer(cfg)
            if args.time_budget:
                _start = time.monotonic()

                def _time_guard(engine, budget=args.time_budget, start=_start):
                    elapsed = time.monotonic() - start
                    if elapsed >= budget:
                        print(f"  time budget {budget}s reached ({elapsed:.0f}s elapsed) — stopping")
                        engine.terminate()

                trainer.trainer.add_event_handler(Events.EPOCH_COMPLETED, _time_guard)
            trainer.run()
        except Exception as e:  # noqa: BLE001 — log and continue sweep
            print(f"  FAILED: {e}", file=sys.stderr)
        else:
            print("  done")
        finally:
            if "trainer" in locals():
                del trainer
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
