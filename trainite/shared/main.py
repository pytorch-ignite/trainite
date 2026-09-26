import argparse
from pathlib import Path

from trainite.shared.utils import load_grid_configs
from trainite.trainers.decoder_trainer import ProjectConfig, Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    args = parser.parse_args()

    configs = load_grid_configs(Path(args.config), ProjectConfig)

    for i, (config, active_params) in enumerate(configs):
        if active_params:
            short_keys = [k.split(".")[-1] for k in active_params.keys()]
            name_suffix = "_".join([f"{k}={v}" for k, v in zip(short_keys, active_params.values())])
            config.output.run_name = f"{config.output.run_name}_{name_suffix}"

            display_params = ", ".join([f"{k}={v}" for k, v in active_params.items()])
            print(f"\n=== Starting Grid Search Run {i + 1} of {len(configs)} ({display_params}) ===")
        else:
            if len(configs) > 1:
                print(f"\n=== Starting Grid Search Run {i + 1} of {len(configs)} ===")

        trainer = Trainer(config)
        trainer.run()


if __name__ == "__main__":
    main()
