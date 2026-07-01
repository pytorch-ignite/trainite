import argparse
from pathlib import Path

from trainite.shared.utils import load_config
from trainite.trainers.decoder_trainer import DecoderTrainer, ProjectConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    args = parser.parse_args()
    config = load_config(Path(args.config), ProjectConfig)
    trainer = DecoderTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
