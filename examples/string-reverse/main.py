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
    train_loader, val_loader = instantiate(config.dataset)
    actual_vocab_size = getattr(train_loader, "vocab_size", None)
    if actual_vocab_size is None:
        raise RuntimeError("The instantiated dataset does not expose a 'vocab_size' attribute.")
    model_params = config.model.model_dump()
    configured_vocab_size = model_params.get("vocab_size")

    if configured_vocab_size is not None:
        if configured_vocab_size < actual_vocab_size:
            raise ValueError(
                f"Configured model vocab_size ({configured_vocab_size}) is smaller than "
                f"the dataset vocabulary size ({actual_vocab_size}).\n"
                f"Please increase model vocab_size or remove it from config.yaml "
                f"to let it resolve automatically."
            )
        vocab_size = configured_vocab_size
    else:
        vocab_size = actual_vocab_size
    model = instantiate(config.model, vocab_size=vocab_size)

    trainer = PreTrainer(
        config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.run()


if __name__ == "__main__":
    main()
