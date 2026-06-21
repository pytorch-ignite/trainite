import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from trainite.datasets.transformed import TransformedDataset
from trainite.shared.utils import get_target, instantiate, load_config
from trainite.trainers.pretrainer import PreTrainer, ProjectConfig


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


def _inject_if_accepted(target_symbol: Any, **candidates: Any) -> dict[str, Any]:
    try:
        sig = inspect.signature(target_symbol)
        return {k: v for k, v in candidates.items() if k in sig.parameters}
    except Exception:
        return {}


def build_dataset(dataset_config: Any, transform_config: Any, tokenizer: Any) -> Dataset:
    ds = get_target(dataset_config.target)
    dataset = instantiate(dataset_config, **_inject_if_accepted(ds, preprocessor=tokenizer, tokenizer=tokenizer))
    if transform_config is not None:
        tf = get_target(transform_config.target)
        transform = instantiate(
            transform_config, **_inject_if_accepted(tf, preprocessor=tokenizer, tokenizer=tokenizer)
        )
        return TransformedDataset(dataset, transform)
    return dataset


def build_dataloader(split_config: Any, tokenizer: Any) -> DataLoader:
    dataset = build_dataset(split_config.dataset, split_config.transform, tokenizer)
    return create_dataloader(dataset, split_config.dataloader, tokenizer)


def build_loaders_from_ratios(
    data_config: Any, tokenizer: Any, seed: int
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    dataset = build_dataset(data_config.dataset, data_config.transform, tokenizer)
    total_len = len(dataset)

    if total_len == 0:
        raise ValueError("Training dataset is empty. Cannot perform train/val/test split.")

    test_ratio = data_config.test_ratio
    val_ratio = data_config.val_ratio
    train_ratio = 1.0 - test_ratio - val_ratio

    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )

    dl_config = data_config.dataloader

    train_loader = create_dataloader(train_ds, dl_config, tokenizer, shuffle=True)
    val_loader = create_dataloader(val_ds, dl_config, tokenizer, shuffle=False)
    test_loader = create_dataloader(test_ds, dl_config, tokenizer, shuffle=False) if test_len > 0 else None

    return train_loader, val_loader, test_loader


def build_dataloaders(data_config: Any, tokenizer: Any, seed: int) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    if hasattr(data_config, "train"):
        train_loader = build_dataloader(data_config.train, tokenizer)
        val_loader = build_dataloader(data_config.val, tokenizer)
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
    config = load_config(config_path, ProjectConfig)

    device = resolve_device(config.device)
    tokenizer = instantiate(config.preprocessor) if config.preprocessor is not None else None

    train_loader, val_loader, test_loader = build_dataloaders(config.data, tokenizer, config.seed)

    vocab_size = resolve_vocab_size(tokenizer, config.model)
    model = build_model(config.model, tokenizer, vocab_size, device)

    optimizer = instantiate(config.optimizer, params=model.parameters())

    # The transform owns the prompt format; hand it to the trainer for inference logging.
    ds = train_loader.dataset
    ds = ds.dataset if isinstance(ds, Subset) else ds
    prompt_transform = getattr(ds, "transform", None)

    trainer = PreTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        preprocessor=tokenizer,
        prompt_transform=prompt_transform,
    )
    trainer.run()


if __name__ == "__main__":
    main()
