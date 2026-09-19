import argparse
from pathlib import Path

# Import your new grid search function instead of the standard one
from utils import load_grid_configs
from trainer import Trainer, ProjectConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    args = parser.parse_args()

    # Generate the list of all parameter combinations
    configs = load_grid_configs(Path(args.config), ProjectConfig)

    # Loop through each configuration and execute the training engine
    for i, config in enumerate(configs):
        print(f"\n=== Starting Grid Search Run {i + 1} of {len(configs)} ===")
        trainer = Trainer(config)
        trainer.run()


if __name__ == "__main__":
    main()
