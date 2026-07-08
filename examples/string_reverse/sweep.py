"""Grid sweep over sequence length, seed, model depth and width.

Usage:
    python sweep.py                    # all combinations, tensorboard
    python sweep.py --logger wandb     # W&B — all runs share one project
    python sweep.py --dry-run          # print run names, don't train
    python sweep.py --seq-lens 4 8 16  # override grid axes via CLI

Each (seq_len, seed, num_layers, hidden_size) combo is one run.
Run name encodes the key hyperparams so TB/W&B graphs are self-labelled.
"""

import argparse
import copy
import sys
from pathlib import Path

# ── Grid defaults (edit here or override via CLI) ──────────────────────────
DEFAULT_SEQ_LENS = [4, 8, 16]
DEFAULT_SEEDS = [42, 123]
DEFAULT_NUM_LAYERS = [2, 4]
DEFAULT_HIDDEN = [64, 128]  # hidden_size; feedforward_dim = hidden_size * 2

PROJECT_NAME = "string-reverse-sweep"
# ──────────────────────────────────────────────────────────────────────────


def run_name(seq_len: int, seed: int, layers: int, hidden: int) -> str:
    return f"sl{seq_len}_s{seed}_l{layers}_h{hidden}"


def make_config(base, seq_len: int, seed: int, layers: int, hidden: int, logger: str):
    cfg = copy.deepcopy(base)

    # Dataset — fix sequence length
    cfg.data.dataset.seq_len = seq_len
    cfg.data.dataset.min_seq_len = None
    cfg.data.dataset.max_seq_len = None
    cfg.data.dataset.seed = seed

    # Model
    cfg.model.num_layers = layers
    cfg.model.hidden_size = hidden
    cfg.model.feedforward_dim = hidden * 2
    cfg.model.max_seq_len = seq_len * 3  # prompt + sep + target + eos headroom

    # Reproducibility
    cfg.seed = seed

    # Run identity — shared project, unique run name per combo
    cfg.output.project = PROJECT_NAME
    cfg.output.run_name = run_name(seq_len, seed, layers, hidden)
    cfg.logger = logger

    cfg.trainer.max_inference_new_tokens = seq_len * 3

    return cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb"])
    p.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--num-layers", nargs="+", type=int, default=DEFAULT_NUM_LAYERS)
    p.add_argument("--hidden", nargs="+", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--dry-run", action="store_true", help="print combos, don't train")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import here so --dry-run works without a full training env
    from utils import load_config
    from trainer import Trainer, ProjectConfig  # noqa: F401 (ProjectConfig used by load_config)

    base = load_config(Path(args.config), ProjectConfig)

    combos = [
        (sl, seed, layers, hidden)
        for sl in args.seq_lens
        for seed in args.seeds
        for layers in args.num_layers
        for hidden in args.hidden
    ]

    total = len(combos)
    print(f"Sweep: {total} runs  |  logger={args.logger}  |  project={PROJECT_NAME}\n")

    for i, (sl, seed, layers, hidden) in enumerate(combos, 1):
        name = run_name(sl, seed, layers, hidden)
        print(f"[{i}/{total}] {name}")

        if args.dry_run:
            continue

        cfg = make_config(base, sl, seed, layers, hidden, args.logger)
        try:
            Trainer(cfg).run()
        except Exception as e:  # noqa: BLE001 — log and continue sweep
            print(f"  FAILED: {e}", file=sys.stderr)
        else:
            print("  done")


if __name__ == "__main__":
    main()
