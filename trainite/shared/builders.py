import inspect
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from trainite.config.base import DataConfigBase, DataWithAutoSplit
from trainite.datasets.transformed import TransformedDataset
from trainite.shared.utils import get_target, instantiate


# Resolve the vocabulary size for the model based on the tokenizer and model configuration.
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


# Inspects the target symbol's signature and filters the candidates to
# only include those that are accepted by the target symbol.
def _inject_if_accepted(target_symbol: Any, **candidates: Any) -> dict[str, Any]:
    try:
        sig = inspect.signature(target_symbol)
        return {k: v for k, v in candidates.items() if k in sig.parameters}
    except Exception:
        return {}


# Builds the model based on the provided configuration, tokenizer, and device.
def build_model(model_config: Any, tokenizer: Any, vocab_size: int, device: str | torch.device) -> nn.Module:
    target_symbol = get_target(model_config.target)
    kwargs = _inject_if_accepted(
        target_symbol,
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = instantiate(model_config, **kwargs)
    model.to(device)
    return model


def build_dataset(dataset_config: Any, transform_config: Any, tokenizer: Any) -> Dataset:
    ds = get_target(dataset_config.target)
    dataset = instantiate(dataset_config, **_inject_if_accepted(ds, preprocessor=tokenizer, tokenizer=tokenizer))
    transform = None
    if transform_config is not None:
        tf = get_target(transform_config.target)
        transform = instantiate(
            transform_config, **_inject_if_accepted(tf, preprocessor=tokenizer, tokenizer=tokenizer)
        )
    # Wraps the dataset with the transform if provided, otherwise returns the dataset as is.
    return TransformedDataset(dataset, transform)


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


def _loaders_from_splits(
    data_config: DataConfigBase, tokenizer: Any
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    def _make(split_config: Any) -> DataLoader:
        ds = build_dataset(split_config.dataset, split_config.transform, tokenizer)
        return create_dataloader(ds, split_config.dataloader, tokenizer)

    return _make(data_config.train), _make(data_config.val), _make(data_config.test) if data_config.test else None


def _loaders_from_ratios(
    data_config: DataWithAutoSplit, tokenizer: Any, seed: int
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    dataset = build_dataset(data_config.dataset, data_config.transform, tokenizer)
    total_len = len(dataset)  # type: ignore
    if total_len == 0:
        raise ValueError("Training dataset is empty. Cannot perform train/val/test split.")
    val_ratio = data_config.val_ratio
    test_ratio = data_config.test_ratio
    train_len = int(total_len * (1.0 - test_ratio - val_ratio))
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )
    dl = data_config.dataloader
    return (
        create_dataloader(train_ds, dl, tokenizer, shuffle=True),
        create_dataloader(val_ds, dl, tokenizer, shuffle=False),
        create_dataloader(test_ds, dl, tokenizer, shuffle=False) if test_len > 0 else None,
    )


def build_dataloaders(data_config: Any, tokenizer: Any, seed: int) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    if isinstance(data_config, DataWithAutoSplit):
        return _loaders_from_ratios(data_config, tokenizer, seed)
    return _loaders_from_splits(data_config, tokenizer)
