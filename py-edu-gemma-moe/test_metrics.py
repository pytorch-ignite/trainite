"""Offline metric checks: python -m unittest test_metrics.py."""

import math
import tempfile
import unittest

import torch
from accelerate import Accelerator
from ignite.engine import Engine
from ignite.handlers.logger_utils import setup_tb_logging
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from models.gemma4_moe import Gemma4MoEModel, GemmaMoECausalLMLoss
from trainer import Trainer, _flatten


class MetricsTest(unittest.TestCase):
    def make_trainer(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        trainer.accelerator = Accelerator(cpu=True)
        trainer.ignore_index = -100
        trainer.requires_router_info = True
        trainer.train_forward_kwargs = {"return_router_info": True}
        trainer.grad_clip_norm = None
        trainer.model = Gemma4MoEModel(
            vocab_size=8,
            hidden_size=16,
            num_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            dense_intermediate_size=16,
            expert_dim=8,
            num_experts=4,
            top_k=2,
            head_dim=8,
            layer_pattern="sg",
            sliding_window=8,
        )
        trainer.criterion = GemmaMoECausalLMLoss(aux_loss_coef=0.1)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.001)
        trainer.trainer = Engine(trainer._train_step)
        trainer.train_evaluator = Engine(lambda engine, output: output)
        trainer.val_evaluator = Engine(lambda engine, output: output)
        trainer.test_evaluator = Engine(lambda engine, output: output)
        trainer.metrics = trainer._attach_metrics()
        return trainer

    def test_token_weighted_perplexity(self):
        trainer = self.make_trainer()
        outputs = [
            {"logits": torch.tensor([[[2.0, 0.0], [0.0, 2.0], [9.0, -9.0]]]), "targets": torch.tensor([[0, 1, -100]])},
            {"logits": torch.tensor([[[0.0, 2.0], [9.0, -9.0]]]), "targets": torch.tensor([[0, -100]])},
        ]
        flattened = [_flatten(output) for output in outputs]
        expected = torch.nn.functional.cross_entropy(
            torch.cat([x for x, y in flattened]),
            torch.cat([y for x, y in flattened]),
        ).item()
        for evaluator in (trainer.train_evaluator, trainer.val_evaluator, trainer.test_evaluator):
            metrics = evaluator.run(outputs).metrics
            self.assertAlmostEqual(metrics["loss"], expected, places=6)
            self.assertAlmostEqual(metrics["perplexity"], math.exp(expected), places=6)
        extreme = {"logits": torch.tensor([[[0.0, 1000.0]]]), "targets": torch.tensor([[0]])}
        self.assertTrue(math.isinf(trainer.val_evaluator.run([extreme]).metrics["perplexity"]))

    def test_training_routing_and_logging(self):
        torch.manual_seed(42)
        trainer = self.make_trainer()
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 3, 4], [-100, 6, 7, 0]]),
        }
        with tempfile.TemporaryDirectory() as directory:
            logger = setup_tb_logging(
                directory,
                trainer.trainer,
                log_every_iters=1,
                trainer_metric_names=trainer.trainer_metric_names,
                evaluators={"validation": trainer.val_evaluator},
                evaluator_metric_names=["loss", "token_accuracy", "perplexity"],
            )
            try:
                state = trainer.trainer.run([batch])
                output = state.output
                expected = torch.nn.functional.cross_entropy(*_flatten(output)).item()
                self.assertAlmostEqual(output["lm_loss"].item(), expected, places=5)
                self.assertAlmostEqual(output["loss"].item(), expected + 0.1 * output["aux_loss"].item(), places=5)
                # Padding contributes no routing assignments; 7 real tokens each select 2 experts.
                torch.testing.assert_close(output["expert_usage"].sum(-1), torch.ones(2))
                torch.testing.assert_close(output["expert_usage"] * 14, (output["expert_usage"] * 14).round())
                for layer in range(2):
                    for expert in range(4):
                        name = f"router/layer_{layer}/expert_{expert}"
                        self.assertAlmostEqual(state.metrics[name], output["expert_usage"][layer, expert].item())
                self.assertFalse(output["expert_usage"].requires_grad)
                evaluated = trainer._eval_step(trainer.val_evaluator, batch)
                trainer.val_evaluator.run([evaluated])
            finally:
                logger.close()
            tags = EventAccumulator(directory).Reload().Tags()["scalars"]
            for name in ("lm_loss", "aux_loss", "router/layer_0/expert_0", "perplexity"):
                self.assertTrue(any(tag.endswith("/" + name) for tag in tags), (name, tags))


if __name__ == "__main__":
    unittest.main()
