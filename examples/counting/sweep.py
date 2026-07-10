import argparse
import csv
import os
from pathlib import Path

from ignite.engine import Events

from config import ProjectConfig
from datasets.counting import CountingDataset, CountingTransform
from datasets.transformed import TransformedDataset
from trainer import Trainer
from utils import load_config, create_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search / hyperparameter sweep for counting example")
    parser.add_argument("--k-values", type=str, default="3,4,5,6,7,8,9,10,11,12", help="Comma-separated k values")
    parser.add_argument("--depths", type=str, default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated layer depths")
    parser.add_argument("--dims", type=str, default="256,512", help="Comma-separated model dims")
    parser.add_argument("--lrs", type=str, default="0.0001,0.00001", help="Comma-separated learning rates")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--output-csv", type=str, default="sweep_results.csv", help="Path to output CSV")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    dims = [int(x) for x in args.dims.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]

    fieldnames = ["k", "depth", "dim", "lr", "best_val_sequence_acc", "test_251_300", "test_301_350", "test_351_400"]
    file_exists = os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for k in k_values:
        for depth in depths:
            for dim in dims:
                for lr in lrs:
                    print(
                        f"\n========================================\n"
                        f"Running configuration: k={k}, depth={depth}, dim={dim}, lr={lr}\n"
                        f"========================================"
                    )

                    # Load base config from local yaml
                    config = load_config(Path("config.yaml"), ProjectConfig)

                    # Dynamic overrides
                    config.data.dataset.k = k
                    config.model.num_layers = depth
                    config.model.hidden_size = dim
                    config.model.feedforward_dim = dim * 2

                    config.optimizer.lr = lr
                    config.trainer.epochs = args.epochs
                    config.output.run_name = f"sweep_k{k}_layer{depth}_dim{dim}_lr{lr}"

                    # Instantiate trainer
                    trainer = Trainer(config)

                    best_val_acc = 0.0

                    # Register validation evaluator handler to track peak accuracy (exact match)
                    @trainer.val_evaluator.on(Events.COMPLETED)
                    def track_best_accuracy(engine):
                        nonlocal best_val_acc
                        acc = engine.state.metrics.get("sequence_accuracy", 0.0)
                        if acc > best_val_acc:
                            best_val_acc = acc

                    # Execute training run
                    trainer.run()

                    # OOD generalization evaluation on the paper's length bins
                    test_results = {}
                    test_bins = [(251, 300), (301, 350), (351, 400)]
                    for bin_min, bin_max in test_bins:
                        test_ds = CountingDataset(
                            total_size=1000,
                            k=k,
                            min_seq_len=bin_min,
                            max_seq_len=bin_max,
                            seed=42,
                        )
                        test_transformed = TransformedDataset(test_ds, CountingTransform(trainer.tokenizer))
                        test_loader = create_dataloader(
                            test_transformed,
                            config.data.dataloader,
                            trainer.tokenizer,
                            shuffle=False,
                        )
                        trainer.test(test_loader)
                        key = f"test_{bin_min}_{bin_max}"
                        test_results[key] = trainer.test_evaluator.state.metrics.get("sequence_accuracy", 0.0)

                    # Record results to CSV file
                    row = {
                        "k": k,
                        "depth": depth,
                        "dim": dim,
                        "lr": lr,
                        "best_val_sequence_acc": best_val_acc,
                        "test_251_300": test_results.get("test_251_300", 0.0),
                        "test_301_350": test_results.get("test_301_350", 0.0),
                        "test_351_400": test_results.get("test_351_400", 0.0),
                    }
                    with open(args.output_csv, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(row)

                    print(
                        f"Finished configuration: k={k}, depth={depth}, dim={dim}, lr={lr} | "
                        f"Best Val Acc: {best_val_acc:.4f}"
                    )


if __name__ == "__main__":
    main()
