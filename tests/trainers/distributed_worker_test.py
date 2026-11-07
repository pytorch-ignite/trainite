import json
import math
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from ignite.engine import Events

from trainite.config.base import (
    DataConfigBase,
    DataLoaderConfig,
    DatasetConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    OutputConfig,
    PreprocessorConfig,
    ProjectConfig,
    SplitConfig,
    TrainerConfig,
)
from trainite.trainers.decoder_trainer import Trainer


def run() -> None:
    run_dir = sys.argv[1]
    cfg = ProjectConfig(
        project_name="dist_test",
        preprocessor=PreprocessorConfig(_target_="tests.trainers.decoder_trainer_test.DummyTokenizer"),
        model=ModelConfig(_target_="tests.trainers.decoder_trainer_test.SimpleModel"),
        optimizer=OptimizerConfig(_target_="torch.optim.SGD", lr=0.01),
        loss=LossConfig(),
        data=DataConfigBase(
            train=SplitConfig(
                dataset=DatasetConfig(_target_="tests.trainers.decoder_trainer_test.SimpleDataset", size=32),
                dataloader=DataLoaderConfig(batch_size=4, num_workers=0),
            ),
            val=SplitConfig(
                dataset=DatasetConfig(_target_="tests.trainers.decoder_trainer_test.SimpleDataset", size=4),
                dataloader=DataLoaderConfig(batch_size=4, num_workers=0),
            ),
            test=SplitConfig(
                dataset=DatasetConfig(_target_="tests.trainers.decoder_trainer_test.SimpleDataset", size=4),
                dataloader=DataLoaderConfig(batch_size=4, num_workers=0),
            ),
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
    # Different batches ensure equal final parameters require gradient synchronization.
    torch.manual_seed(cfg.seed + dist.get_rank())
    losses = []

    def check_loss(engine):
        loss = float(engine.state.output["loss"])
        assert math.isfinite(loss), f"Rank {dist.get_rank()}: non-finite loss at iteration {engine.state.iteration}"
        losses.append(loss)

    def check_synchronized_parameters(engine):
        assert engine.state.epoch == 1
        assert engine.state.iteration == len(losses) == 4
        assert any(
            not torch.equal(initial[name], parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        # Compare before Trainer.run() reloads best.pt for testing.
        parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
        gathered = [torch.empty_like(parameters) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, parameters)
        for other in gathered:
            torch.testing.assert_close(parameters, other, rtol=0, atol=0)

    trainer.trainer.add_event_handler(Events.ITERATION_COMPLETED, check_loss)
    trainer.trainer.add_event_handler(Events.COMPLETED, check_synchronized_parameters)
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
