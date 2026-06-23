import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from config import load_config
from trainer import DecoderTrainer
from utils import get_target, instantiate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    return parser.parse_args()


def resolve_device(device: str) -> str | torch.device:
    resolved = device
    if resolved == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    return resolved


def resolve_vocab_size(tokenizer: Any, model_config: Any) -> int:
    if tokenizer is None or not hasattr(tokenizer, "vocab_size"):
        raise ValueError("Tokenizer is missing or does not define a 'vocab_size' attribute.")
    vocab_size = tokenizer.vocab_size

    model_params = model_config.model_dump(by_alias=True)
    configured_vocab_size: int | None = model_params.get("vocab_size")
    if configured_vocab_size is not None:
        if configured_vocab_size < vocab_size:
            raise ValueError(
                f"Configured model vocab_size ({configured_vocab_size}) is smaller than "
                f"the tokenizer vocabulary size ({vocab_size}). "
                f"Please increase model vocab_size or remove it from config.yaml "
                f"to let it resolve automatically."
            )
        vocab_size = configured_vocab_size
    return vocab_size


def build_model(model_config: Any, tokenizer: Any, vocab_size: int, device: str | torch.device) -> nn.Module:
    target_path = model_config.target
    target_symbol = get_target(target_path)

    has_vocab_size_param = False
    try:
        sig = inspect.signature(target_symbol)
        if "vocab_size" in sig.parameters:
            has_vocab_size_param = True
    except Exception:
        pass

    kwargs = {}
    if has_vocab_size_param:
        kwargs["vocab_size"] = vocab_size

    kwargs["pad_token_id"] = tokenizer.pad_token_id

    model = instantiate(model_config, **kwargs)
    model.to(device)
    return model


def create_dataloader(
    dataset: Dataset,
    dl_config: Any,
    tokenizer: Any,
    shuffle: bool | None = None,
) -> DataLoader:
    dl_kwargs = dl_config.model_dump(exclude={"collate_fn", "shuffle"})
    if shuffle is None:
        shuffle = getattr(dl_config, "shuffle", False)

    collate_fn = None
    collate_config = dl_config.collate_fn
    if collate_config:
        target_symbol = get_target(collate_config.target)
        if isinstance(target_symbol, type):
            collate_fn = instantiate(collate_config, tokenizer=tokenizer)
        else:
            collate_fn = target_symbol

    return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **dl_kwargs)


def build_dataloader(split_config: Any, tokenizer: Any) -> DataLoader:
    dataset = instantiate(split_config.dataset)
    return create_dataloader(dataset, split_config.dataloader, tokenizer)


def build_loaders_from_ratios(
    data_config: Any, tokenizer: Any, seed: int
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    dataset = instantiate(data_config.dataset)
    total_len = len(dataset)

    train_ratio = data_config.train_ratio if data_config.train_ratio is not None else 1.0
    val_ratio = data_config.val_ratio if data_config.val_ratio is not None else 0.0

    if train_ratio <= 0.0 or val_ratio < 0:
        raise ValueError(f"train_ratio must be between 0 and 1. Got train_ratio={train_ratio}, val_ratio={val_ratio}")

    if train_ratio + val_ratio > 1.0:
        raise ValueError(f"Sum of train_ratio ({train_ratio}) and val_ratio ({val_ratio}) exceeds 1.0")

    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )

    dl_config = data_config.dataloader or instantiate(None)  # fallback or default

    train_loader = create_dataloader(train_ds, dl_config, tokenizer, shuffle=True)
    val_loader = create_dataloader(val_ds, dl_config, tokenizer, shuffle=False) if val_len > 0 else None
    test_loader = create_dataloader(test_ds, dl_config, tokenizer, shuffle=False) if test_len > 0 else None

    return train_loader, val_loader, test_loader


def build_dataloaders(
    data_config: Any, tokenizer: Any, seed: int
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    if data_config.train:
        train_loader = build_dataloader(data_config.train, tokenizer)
        val_loader = build_dataloader(data_config.val, tokenizer) if data_config.val else None
        test_loader = build_dataloader(data_config.test, tokenizer) if data_config.test else None
    else:
        train_loader, val_loader, test_loader = build_loaders_from_ratios(data_config, tokenizer, seed)

    return train_loader, val_loader, test_loader


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

    device = resolve_device(config.device)
    # The example code might not define config.tokenizer. Let's resolve it carefully.
    # In examples/string_reverse/trainer.py, it got vocab_size from train_dataset.vocab_size.
    # Wait, the dataset class itself contains vocabulary / tokenizer?
    # Let's check examples/string_reverse/trainer.py lines 61-86.
    # It says:
    # "train_dataset = self.train_loader.dataset ... self.vocab_size: int = getattr(train_dataset, 'vocab_size')"
    # Wait, in the string_reverse example, is there a tokenizer in config?
    # Let's check examples/string_reverse/config.py ProjectConfig fields.
    # ProjectConfig has: model, optimizer, data, trainer, output, seed, device.
    # It does NOT have a tokenizer!
    # Ah! The string_reverse example does not have config.tokenizer!
    # Let's check how the model is instantiated in examples/string_reverse/trainer.py:
    # `self.model: nn.Module = model or instantiate(config.model, vocab_size=self.vocab_size)`
    # So the vocab_size is resolved from the dataset!
    # Okay, let's keep examples/string_reverse/trainer.py and main.py decoupled as well.
    # In examples/string_reverse/main.py:
    # 1. Resolve device.
    # 2. Build dataloaders (tokenizer is None).
    # 3. Resolve vocab_size from train_loader's dataset:
    #    train_dataset = train_loader.dataset
    #    if isinstance(train_dataset, torch.utils.data.Subset):
    #        train_dataset = train_dataset.dataset
    #    vocab_size = getattr(train_dataset, "vocab_size")
    # 4. Build model using config.model and vocab_size.
    # 5. Build optimizer.
    # 6. Run DecoderTrainer.

    train_loader, val_loader, test_loader = build_dataloaders(config.data, None, config.seed)

    train_dataset = train_loader.dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        train_dataset = train_dataset.dataset
    vocab_size = getattr(train_dataset, "vocab_size")

    # check model config configured_vocab_size
    model_params = config.model.model_dump(by_alias=True)
    configured_vocab_size = model_params.get("vocab_size")
    if configured_vocab_size is not None:
        if configured_vocab_size < vocab_size:
            raise ValueError(
                f"Configured model vocab_size ({configured_vocab_size}) is smaller than "
                f"the dataset vocabulary size ({vocab_size}). "
                f"Please increase model vocab_size or remove it from config.yaml "
                f"to let it resolve automatically."
            )
        vocab_size = configured_vocab_size

    model = instantiate(config.model, vocab_size=vocab_size)
    model.to(device)

    optimizer = instantiate(config.optimizer, params=model.parameters())

    trainer = DecoderTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        vocab_size=vocab_size,
    )
    trainer.run()


if __name__ == "__main__":
    main()
