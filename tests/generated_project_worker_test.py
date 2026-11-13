"""Copied into the generated project and executed with its declared dependencies."""

import importlib
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch


def local_tokenizer_loader():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    vocabulary = {
        token: index
        for index, token in enumerate(["<unk>", "<bos>", "<eos>", "<pad>", "hello", "world", "User", "Assistant", ":"])
    }
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 2)]
    )
    fixture = Path("tokenizer-fixture")
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>", pad_token="<pad>"
    ).save_pretrained(fixture)
    original_loader = AutoTokenizer.from_pretrained

    def load(name, **kwargs):
        assert name == "openai-community/gpt2"
        return original_loader(str(fixture), local_files_only=True, **kwargs)

    return patch.object(AutoTokenizer, "from_pretrained", side_effect=load)


def run() -> None:
    import torch
    import torch.distributed as dist
    from ignite.engine import Events
    import yaml

    main = importlib.import_module("main")
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    trainer_cls = main.Trainer
    trainers = []
    initial_parameters = {}
    losses = []

    def capture_trainer(config):
        trainer = trainer_cls(config)
        trainers.append(trainer)
        model = trainer.accelerator.unwrap_model(trainer.model)
        initial_parameters.update(
            {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        )
        if distributed:
            assert trainer.accelerator.use_distributed
            assert trainer.accelerator.num_processes == 2
            assert len(trainer.train_loader.dataset) % 8 != 0
            lengths = {item.train_input_ids.numel() for item in trainer.train_loader.dataset}
            assert len(lengths) > 1
            torch.manual_seed(config.seed + dist.get_rank())

        def check_loss(engine):
            loss = float(engine.state.output["loss"])
            assert math.isfinite(loss), f"Non-finite training loss at iteration {engine.state.iteration}: {loss}"
            losses.append(loss)

        trainer.trainer.add_event_handler(Events.ITERATION_COMPLETED, check_loss)

        def check_training(engine):
            assert engine.state.epoch == 1
            assert engine.state.iteration == len(losses) == 4
            assert any(
                not torch.equal(initial_parameters[name], parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
            if trainer.requires_router_info:
                for name, parameter in model.named_parameters():
                    if name.endswith("moe.router.weight"):
                        assert parameter.grad is not None
                        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            if distributed:
                parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
                gathered = [torch.empty_like(parameters) for _ in range(2)]
                dist.all_gather(gathered, parameters)
                for other in gathered:
                    torch.testing.assert_close(parameters, other, rtol=0, atol=0)

        # Capture synchronization before Trainer.run() reloads a shared checkpoint.
        trainer.trainer.add_event_handler(Events.COMPLETED, check_training)
        return trainer

    main.Trainer = capture_trainer
    tokenizer_context = local_tokenizer_loader() if "gpt2" in config["preprocessor"]["_target_"] else nullcontext()
    with tokenizer_context:
        main.main()
    assert len(trainers) == 1
    trainer = trainers[0]
    assert trainer.trainer.state.epoch == 1
    assert trainer.trainer.state.iteration == len(losses) == 4
    for evaluator in (trainer.train_evaluator, trainer.val_evaluator, trainer.test_evaluator):
        assert evaluator.state.iteration > 0
        assert math.isfinite(float(evaluator.state.metrics["loss"]))
        assert 0.0 <= float(evaluator.state.metrics["token_accuracy"]) <= 1.0

    model = trainer.accelerator.unwrap_model(trainer.model)
    assert any(
        not torch.equal(initial_parameters[name], parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    for filename in ("last.pt", "best.pt"):
        checkpoint = torch.load(trainer.run_dir / filename, map_location="cpu", weights_only=True)
        assert checkpoint["optimizer"]["state"], f"No optimizer updates in {filename}"
        assert checkpoint["model"]
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
        model.load_state_dict(checkpoint["model"], strict=True)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])

    # Corrupt weights so successful evaluation alone cannot hide a missing checkpoint reload.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    trainer.test()
    best = torch.load(trainer.run_dir / "best.pt", map_location="cpu", weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, best["model"][name], rtol=0, atol=0)
    assert trainer.test_evaluator.state.iteration == len(trainer.test_loader)
    assert math.isfinite(float(trainer.test_evaluator.state.metrics["loss"]))
    assert 0.0 <= float(trainer.test_evaluator.state.metrics["token_accuracy"]) <= 1.0
    if distributed:
        trainer.accelerator.wait_for_everyone()
        Path(f"rank-{dist.get_rank()}.json").write_text(
            json.dumps(
                {
                    "rank": dist.get_rank(),
                    "iterations": trainer.trainer.state.iteration,
                    "run_dir": str(trainer.run_dir),
                }
            )
        )
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    run()
