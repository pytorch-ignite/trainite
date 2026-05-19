import argparse
import logging
from pathlib import Path

from config import load_config
from trainer import PreTrainer
from utils import instantiate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence Ignite internal logs
    logging.getLogger("ignite.engine").setLevel(logging.WARNING)

    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    tokenizer = instantiate(config.tokenizer)
    trainer = PreTrainer(config, tokenizer=tokenizer)
    trainer.run()


if __name__ == "__main__":
    main()

