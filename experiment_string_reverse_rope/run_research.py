import itertools
import json
import logging
from pathlib import Path

import torch
from config import load_config
from trainer import PreTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Silence verbose logs
logging.getLogger("ignite.engine").setLevel(logging.WARNING)
logging.getLogger("trainer").setLevel(logging.INFO)


def run_research():
    base_cfg_path = Path("research_base.yaml")
    results_file = Path("research_results.json")

    # --- Experiment Grid ---
    layers_options = [1, 2, 4]
    heads_options = [1, 2, 4]
    # Sweep lengths to test generalization
    # Training is on 1-64. We test up to 512.
    sweep_lengths = [
        1,
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        320,
        384,
        448,
        512,
    ]
    # -----------------------

    all_results = []

    for nl, nh in itertools.product(layers_options, heads_options):
        print(f"\n{'=' * 60}")
        print(f"STARTING ARCHITECTURE: Layers={nl}, Heads={nh}")
        print(f"{'=' * 60}\n")

        cfg = load_config(base_cfg_path)

        # 1. Update Model Architecture
        cfg.model.num_layers = nl
        cfg.model.num_heads = nh
        cfg.output.run_name = f"research_L{nl}_H{nh}"

        # 2. Execute Training (Fixed range 1-64)
        trainer = PreTrainer(cfg)
        trainer.run()

        # --- LOADING BEST MODEL ---
        # We manually load the best model to ensure all metrics (Train/Val/Test)
        # are calculated using the same optimized weights.
        checkpoint_handler = trainer.handlers.get(
            "checkpoint_best"
        ) or trainer.handlers.get("checkpoint_last")
        if checkpoint_handler and checkpoint_handler.last_checkpoint:
            checkpoint_path = checkpoint_handler.last_checkpoint
            print(f">>> Loading best model from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=trainer.device, weights_only=True
            )
            trainer.model.load_state_dict(checkpoint["model"])

        # 3. Capture "Best Model" Metrics for ID (In-Distribution)
        print(">>> Evaluating Best Model on Train/Val sets...")
        trainer.train_evaluator.run(trainer.train_loader)
        best_train_acc = trainer.train_evaluator.state.metrics.get("exact_accuracy", 0)

        trainer.val_evaluator.run(trainer.val_loader)
        best_val_acc = trainer.val_evaluator.state.metrics.get("exact_accuracy", 0)

        # 4. Perform Length Sweep (OOD Analysis)
        print(">>> Starting Length Sweep...")
        sweep_data = []
        for length in sweep_lengths:
            # Create a specific loader for this length
            # We use a smaller size (200) for speed during the sweep
            sweep_cfg = cfg.data.test.model_copy(deep=True)
            sweep_cfg.dataset.min_seq_len = length
            sweep_cfg.dataset.max_seq_len = length
            sweep_cfg.dataset.size = 200

            loader = trainer._build_dataloader(sweep_cfg)
            trainer.test_evaluator.run(loader)
            acc = trainer.test_evaluator.state.metrics.get("exact_accuracy", 0)

            print(f"  Length {length:3d}: Accuracy = {acc:>7.2%}")
            sweep_data.append({"length": length, "accuracy": acc})

        res = {
            "layers": nl,
            "heads": nh,
            "best_train_acc": best_train_acc,
            "best_val_acc": best_val_acc,
            "length_sweep": sweep_data,
            "run_dir": str(trainer.run_dir),
        }
        all_results.append(res)

        # Save intermediate results
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    # --- Final Summary Table ---
    print("\n" + "#" * 40)
    print("        RESEARCH SUMMARY")
    print("#" * 40)
    print(
        f"{'L':<2} | {'H':<2} | {'Train %':<8} | {'Val %':<8} | {'Generalization Cliff'}"
    )
    print("-" * 65)
    for r in all_results:
        # Find where it drops below 80%
        cliff = "N/A"
        for s in r["length_sweep"]:
            if s["accuracy"] < 0.8:
                cliff = f"~{s['length']}"
                break

        print(
            f"{r['layers']:<2} | {r['heads']:<2} | "
            f"{r['best_train_acc']:>8.2%} | {r['best_val_acc']:>8.2%} | {cliff}"
        )


if __name__ == "__main__":
    run_research()