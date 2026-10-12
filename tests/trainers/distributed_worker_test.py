import sys

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
        loss=LossConfig(_target_="torch.nn.CrossEntropyLoss"),
        data=DataConfigBase(
            train=SplitConfig(
                dataset=DatasetConfig(_target_="tests.trainers.decoder_trainer_test.SimpleDataset", size=8),
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
        device=None,
    )
    trainer = Trainer(cfg)
    assert trainer.accelerator.use_distributed
    assert trainer.accelerator.num_processes == 2
    trainer.run()
    if trainer.accelerator.is_main_process:
        assert (trainer.run_dir / "best.pt").exists()


if __name__ == "__main__":
    run()
