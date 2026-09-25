import argparse
import os
from pathlib import Path

import ignite.distributed as idist
import torch

from trainite.shared.utils import load_config
from trainite.trainers.decoder_trainer import Trainer, ProjectConfig


def run_training(local_rank: int, config: ProjectConfig) -> None:
    trainer = Trainer(config)
    trainer.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainite Experiment Runner")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--backend", type=str, default=None, help="Distributed backend ('nccl' or 'gloo').")
    parser.add_argument(
        "--nproc-per-node", type=int, default=None, help="Number of processes per node for in-script spawning."
    )
    args = parser.parse_args()

    config = load_config(Path(args.config), ProjectConfig)

    # Determine backend if running in a distributed environment (e.g. torchrun) or when spawning
    backend = args.backend
    if backend is None and (args.nproc_per_node or "RANK" in os.environ or "WORLD_SIZE" in os.environ):
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    with idist.Parallel(backend=backend, nproc_per_node=args.nproc_per_node) as parallel:
        parallel.run(run_training, config)


if __name__ == "__main__":
    main()
