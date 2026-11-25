import json
import math
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from ignite.engine import Events

from trainite.config.base import DataLoaderConfig, OptimizerConfig, OutputConfig, ProjectConfig, TrainerConfig
from trainite.config.datasets import StringReverseDataConfig, StringReverseDatasetConfig
from trainite.config.preprocessors import CharTokenizerConfig
from trainite.config.registry import MODEL_SPECS
from trainite.trainers.decoder_trainer import Trainer


def run() -> None:
    run_dir = sys.argv[1]
    model_name = sys.argv[2]
    spec = MODEL_SPECS[model_name]
    model_config = dict(hidden_size=8, num_layers=2, collate_fn_target=spec.collate_fn_target)
    if model_name == "rope-transformer":
        model_config.update(feedforward_dim=16, dropout=0.0, max_seq_len=32)
    else:
        model_config.update(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            global_num_key_value_heads=1,
            global_head_dim=4,
            dense_intermediate_size=16,
            expert_dim=4,
            num_experts=3,
            top_k=2,
            sliding_window=8,
        )
    cfg = ProjectConfig(
        project_name="dist_test",
        preprocessor=CharTokenizerConfig(charset="abcd"),
        model=spec.config_cls(**model_config),
        optimizer=OptimizerConfig(_target_="torch.optim.SGD", lr=0.01),
        loss=spec.loss_config_cls(),
        data=StringReverseDataConfig(
            dataset=StringReverseDatasetConfig(per_seq_size=13, min_seq_len=2, max_seq_len=4, charset="abcd"),
            dataloader=DataLoaderConfig(batch_size=4, num_workers=0),
        ),
        trainer=TrainerConfig(epochs=1, log_every_steps=1),
        output=OutputConfig(root=run_dir, run_name="run"),
        logger="tensorboard",
        device="cpu",
    )
    trainer = Trainer(cfg)
    assert trainer.accelerator.use_distributed
    assert trainer.accelerator.num_processes == 2
    model = trainer.accelerator.unwrap_model(trainer.model)
    initial = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    losses = []

    def check_loss(engine):
        loss = float(engine.state.output["loss"])
        assert math.isfinite(loss), f"Rank {dist.get_rank()}: non-finite loss at iteration {engine.state.iteration}"
        losses.append(loss)

    def check_training_progress(engine):
        assert engine.state.epoch == 1
        assert engine.state.iteration == len(losses) == len(trainer.train_loader)
        assert len(losses) > 1
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        assert any(
            not torch.equal(initial[name], parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )

    trainer.trainer.add_event_handler(Events.ITERATION_COMPLETED, check_loss)
    # Compare before Trainer.run() reloads best.pt for testing.
    trainer.trainer.add_event_handler(Events.COMPLETED, check_training_progress)
    trainer.run()
    for evaluator in (trainer.train_evaluator, trainer.val_evaluator, trainer.test_evaluator):
        assert evaluator.state.iteration > 0
        assert math.isfinite(float(evaluator.state.metrics["loss"]))
        assert 0.0 <= float(evaluator.state.metrics["token_accuracy"]) <= 1.0

    trainer.accelerator.wait_for_everyone()
    for filename in ("last.pt", "best.pt"):
        checkpoint = torch.load(trainer.run_dir / filename, map_location="cpu", weights_only=True)
        assert checkpoint["model"]
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
        assert "param_groups" in checkpoint["optimizer"]
        model.load_state_dict(checkpoint["model"], strict=True)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    trainer.test()
    best = torch.load(trainer.run_dir / "best.pt", map_location="cpu", weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, best["model"][name], rtol=0, atol=0)
    assert trainer.test_evaluator.state.iteration == len(trainer.test_loader)
    assert math.isfinite(float(trainer.test_evaluator.state.metrics["loss"]))
    trainer.accelerator.wait_for_everyone()
    Path(run_dir, f"rank-{dist.get_rank()}.json").write_text(
        json.dumps(
            {"rank": dist.get_rank(), "iterations": trainer.trainer.state.iteration, "run_dir": str(trainer.run_dir)}
        )
    )
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    run()
