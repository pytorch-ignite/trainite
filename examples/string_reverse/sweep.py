"""Hyperparameter sweep for string-reverse experiments.

Three modes — only one axis varies at a time for clean interpretation:

  depth   Fix dim/heads, vary num_layers
  dim     Fix layers, vary (hidden_size, num_heads) pairs  [heads tied to hidden]
  full    Cross-product of both (expensive)

Usage:
    python sweep.py --mode depth              # tensorboard, default grid
    python sweep.py --mode dim --logger wandb
    python sweep.py --mode full --seq-lens 4 8 --seeds 42
    python sweep.py --mode depth --dry-run    # print combos, don't train
    python sweep.py --mode depth --time-budget 600  # 10 min per run
"""

import argparse
import copy
import sys
import time
from pathlib import Path


DEFAULT_SEQ_LENS = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
DEFAULT_SEEDS = [42, 123, 456]


DEPTH_LAYERS = [4, 6, 8]
DEPTH_FIXED_HIDDEN = [32]  # single-element list — consistent with DEPTH_LAYERS
DEPTH_FIXED_HEADS = [2]  # head_dim = 16

DIM_HIDDEN_HEADS = [
    (16, 1),
    (32, 2),
    (64, 4),
]
DIM_FIXED_LAYERS = [4]  # single-element list — consistent with pairs

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


def build_combos(mode: str, args) -> tuple[list[tuple], str]:
    seq_lens = args.seq_lens
    seeds = args.seeds

    if mode == "depth":
        layers_grid = DEPTH_LAYERS if args.layers is None else args.layers
        pairs = list(
            zip(
                DEPTH_FIXED_HIDDEN if args.fixed_hidden is None else args.fixed_hidden,
                DEPTH_FIXED_HEADS if args.fixed_heads is None else args.fixed_heads,
            )
        )
    elif mode == "dim":
        layers_grid = DIM_FIXED_LAYERS if args.fixed_layers is None else args.fixed_layers
        hidden_vals = [h for h, _ in DIM_HIDDEN_HEADS] if args.hidden is None else args.hidden
        heads_vals = [nh for _, nh in DIM_HIDDEN_HEADS] if args.heads is None else args.heads
        pairs = list(zip(hidden_vals, heads_vals))
    else:  # full
        layers_grid = DEPTH_LAYERS if args.layers is None else args.layers
        hidden_vals = [h for h, _ in DIM_HIDDEN_HEADS] if args.hidden is None else args.hidden
        heads_vals = [nh for _, nh in DIM_HIDDEN_HEADS] if args.heads is None else args.heads
        pairs = list(zip(hidden_vals, heads_vals))

    return [
        (sl, seed, layers, hidden, heads)
        for sl in seq_lens
        for seed in seeds
        for layers in layers_grid
        for hidden, heads in pairs
    ], PROJECT_NAME + f"-{mode}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode", default="depth", choices=["depth", "dim", "full"], help="which axis to sweep (default: depth)"
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb"])
    p.add_argument(
        "--seq-lens", nargs="+", type=int, default=None, help=f"sequence lengths (default: {DEFAULT_SEQ_LENS})"
    )
    p.add_argument("--seeds", nargs="+", type=int, default=None, help=f"seeds (default: {DEFAULT_SEEDS})")

    # depth-mode axes
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help=f"num_layers to sweep in depth mode (default: {DEPTH_LAYERS})",
    )
    p.add_argument(
        "--fixed-hidden",
        nargs="+",
        type=int,
        default=None,
        help=f"hidden_size fixed for depth sweep (default: {DEPTH_FIXED_HIDDEN})",
    )
    p.add_argument(
        "--fixed-heads",
        nargs="+",
        type=int,
        default=None,
        help=f"num_heads fixed for depth sweep (default: {DEPTH_FIXED_HEADS})",
    )
    p.add_argument(
        "--fixed-layers",
        nargs="+",
        type=int,
        default=None,
        help=f"num_layers fixed for dim sweep (default: {DIM_FIXED_LAYERS})",
    )

    # dim-mode axes — hidden and heads must be the same length (they're paired)
    p.add_argument(
        "--hidden", nargs="+", type=int, default=None, help="hidden_size values for dim sweep (paired with --heads)"
    )
    p.add_argument(
        "--heads", nargs="+", type=int, default=None, help="num_heads values for dim sweep (paired with --hidden)"
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
    args.seq_lens = args.seq_lens or DEFAULT_SEQ_LENS
    args.seeds = args.seeds or DEFAULT_SEEDS

    # Validate paired args for dim mode
    if args.hidden and args.heads and len(args.hidden) != len(args.heads):
        print("ERROR: --hidden and --heads must have the same number of values (they are paired).", file=sys.stderr)
        sys.exit(1)

    combos, project_name = build_combos(args.mode, args)
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


if __name__ == "__main__":
    main()
