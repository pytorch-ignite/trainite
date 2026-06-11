import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import ProjectConfig, load_config
from ignite.engine import Engine, Events
from trainer import PreTrainer

NUM_RUNS: int = 2
TIME_LIMIT_SECONDS: float = 30
TEST_SEQ_LENGTHS: list[int] = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]

MODEL_HIDDEN_SIZE: int = 256
MODEL_NUM_LAYERS: int = 6
MODEL_NUM_HEADS: int = 4
MODEL_FEEDFORWARD_DIM: int = 1024
MODEL_MAX_SEQ_LEN: int = max(TEST_SEQ_LENGTHS) * 3
FIXED_DATASET_SIZE: int = 10000

_MODEL_OVERRIDES = {
    "hidden_size": MODEL_HIDDEN_SIZE,
    "num_layers": MODEL_NUM_LAYERS,
    "num_heads": MODEL_NUM_HEADS,
    "feedforward_dim": MODEL_FEEDFORWARD_DIM,
    "max_seq_len": MODEL_MAX_SEQ_LEN,
}

_OUTPUT_ROOT = f"outputs/d{MODEL_HIDDEN_SIZE}_l{MODEL_NUM_LAYERS}_h{MODEL_NUM_HEADS}"


def _make_run_name(seq_len: int, seed: int) -> str:
    return f"sl{seq_len}/s{seed}"


def _apply_base_overrides(cfg: ProjectConfig) -> None:
    for attr, value in _MODEL_OVERRIDES.items():
        setattr(cfg.model, attr, value)
    cfg.model.vocab_size = None
    cfg.trainer.epochs = 9999
    cfg.trainer.log_every_steps = 1000
    cfg.output.root = _OUTPUT_ROOT


def build_fixed_config(
    base: ProjectConfig,
    seq_len: int,
    train_charset: str,
    seed: int = 42,
) -> ProjectConfig:
    cfg = ProjectConfig.model_validate(base.model_dump(by_alias=True))
    _apply_base_overrides(cfg)
    cfg.data.dataset.seq_len = seq_len
    cfg.data.dataset.per_seq_size = FIXED_DATASET_SIZE
    cfg.data.dataset.min_seq_len = None
    cfg.data.dataset.max_seq_len = None
    cfg.data.dataset.charset = train_charset
    cfg.trainer.max_inference_new_tokens = seq_len + 1
    cfg.output.run_name = _make_run_name(seq_len, seed)
    cfg.seed = seed
    return cfg


def _make_anchor_snapshot_callback(
    trainer: PreTrainer,
    store: dict[str, float],
) -> Callable[[Engine], None]:
    """Snapshot all metrics at the epoch where val-exact-accuracy peaks."""

    def handler(engine: Engine) -> None:
        val_metrics = trainer.val_evaluator.state.metrics
        current_val_exact: float = val_metrics["exact_accuracy"]

        train_metrics = trainer.train_evaluator.state.metrics
        current_train_exact: float = train_metrics["exact_accuracy"]
        prev_max_train = store.get("max_train_exact_acc", -1.0)
        if current_train_exact > prev_max_train:
            store["max_train_exact_acc"] = current_train_exact

        prev_best_val = store.get("anchor_val_exact_acc", -1.0)
        if current_val_exact <= prev_best_val:
            return

        store["anchor_val_exact_acc"] = current_val_exact
        store["anchor_epoch"] = float(engine.state.epoch)
        store["train_loss"] = train_metrics["loss"]
        store["train_exact_acc"] = train_metrics["exact_accuracy"]
        store["train_token_acc"] = train_metrics["token_accuracy"]
        store["val_loss"] = val_metrics["loss"]
        store["val_exact_acc"] = val_metrics["exact_accuracy"]
        store["val_token_acc"] = val_metrics["token_accuracy"]

    return handler


def run_single_experiment(
    cfg: ProjectConfig,
    time_limit_seconds: float,
) -> tuple[dict[str, float], float, int]:
    """Train one model; return (snapshot_metrics, elapsed_seconds, epochs_trained).

    Eval/inference metrics are anchored to the epoch where val-exact-accuracy peaked.
    ``max_train_exact_acc`` is the best train-exact at *any* epoch.
    """
    store: dict[str, float] = {}

    trainer = PreTrainer(cfg)
    start = time.time()

    def _timeout(engine: Engine) -> None:
        if time.time() - start > time_limit_seconds:
            print(
                f"  Time limit ({time_limit_seconds:.0f}s) reached - terminating early."
            )
            engine.terminate()

    trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, _timeout)

    def _early_stopping(engine: Engine) -> None:
        if engine.state.metrics.get("exact_accuracy", 0.0) >= 0.999:
            print("\n  100% train accuracy reached! Terminating early.", end="")
            trainer.engine.terminate()

    trainer.train_evaluator.add_event_handler(Events.EPOCH_COMPLETED, _early_stopping)

    original_attach = trainer._attach_handlers

    def _patched_attach() -> None:
        original_attach()
        trainer.engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            _make_anchor_snapshot_callback(trainer, store),
        )

    trainer._attach_handlers = _patched_attach  # type: ignore

    trainer.run()
    elapsed = time.time() - start
    epochs_trained = trainer.engine.state.epoch

    # Explicitly run test to ensure it runs even if run() exited abnormally or was terminated
    if getattr(trainer, "test_loader", None) is not None:
        if "ar_exact_match_acc" not in trainer.test_evaluator.state.metrics:
            print("  [experiment_script] Running final autoregressive test...")
            trainer.test()

        store["test_ar_exact_match_acc"] = trainer.test_evaluator.state.metrics.get(
            "ar_exact_match_acc", 0.0
        )
        store["test_ar_token_acc"] = trainer.test_evaluator.state.metrics.get(
            "ar_token_acc", 0.0
        )

    return store, elapsed, epochs_trained


def run_suite(
    configs: list[tuple[int, ProjectConfig]],
    time_limit_seconds: float,
    label: str,
) -> tuple[list[int], dict]:
    """Run every config ``NUM_RUNS`` times, returning x-values and per-metric stats."""
    x_vals: list[int] = []
    _METRICS = [
        "train_loss",
        "train_exact_acc",
        "train_token_acc",
        "val_loss",
        "val_exact_acc",
        "val_token_acc",
        "max_train_exact_acc",
        "epochs_trained",
        "test_ar_exact_match_acc",
        "test_ar_token_acc",
    ]
    raw: dict[str, list[list[float]]] = {k: [] for k in _METRICS}

    for x_val, base_cfg in configs:
        print(f"\n{'─' * 60}")
        print(f"[{label}]  x = {x_val}")
        print(f"{'─' * 60}")

        runs: dict[str, list[float]] = {k: [] for k in raw}

        for run_idx in range(NUM_RUNS):
            cfg = ProjectConfig.model_validate(base_cfg.model_dump(by_alias=True))
            cfg.seed = base_cfg.seed + run_idx
            print(f"  Run {run_idx + 1}/{NUM_RUNS}  (seed = {cfg.seed})", end="")

            snap, elapsed, epochs = run_single_experiment(cfg, time_limit_seconds)

            print(
                f"  -> {elapsed:.0f}s  {epochs} epochs  "
                f"v_acc={snap.get('val_exact_acc', float('nan')):.3f}  "
                f"test_ar_acc={snap.get('test_ar_exact_match_acc', float('nan')):.3f}  "
                f"max_train={snap.get('max_train_exact_acc', 0):.3f}"
            )

            for key in _METRICS:
                if key == "epochs_trained":
                    runs[key].append(float(epochs))
                else:
                    runs[key].append(snap.get(key, float("nan")))

        x_vals.append(x_val)
        for k in raw:
            raw[k].append(runs[k])

    results: dict = {}
    for k, data in raw.items():
        arr = np.array(data)
        results[k] = {
            "mean": arr.mean(axis=1).tolist(),
            "std": arr.std(axis=1).tolist(),
        }

    return x_vals, results


METRIC_COLORS = {"train": "#1f77b4", "val": "#d62728"}
INFERENCE_COLOR = "#9467bd"
TRAIN_MAX_COLOR = "#ff7f0e"


def plot_results(
    x_values: list[int],
    results: dict,
    model_info: dict,
    out_path: Path,
) -> None:
    """6-panel figure (3×2) with error bars (± 1 std)."""
    specs: list[tuple[str, str, str | None, list[float] | None]] = [
        ("Token Accuracy (anchored)", "train_token_acc", "val_token_acc", [0, 1]),
        ("Exact Match Accuracy (anchored)", "train_exact_acc", "val_exact_acc", [0, 1]),
        ("Loss (anchored)", "train_loss", "val_loss", None),
        ("Epochs Trained", "epochs_trained", None, None),
        ("Inference Token Acc (anchored)", "test_ar_token_acc", None, [0, 1]),
        (
            "Inference Exact Match Acc (anchored)",
            "test_ar_exact_match_acc",
            None,
            [0, 1],
        ),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        f"Transformer — Fixed Seq Len\n"
        f"layers={model_info['num_layers']},  d={model_info['hidden_size']},  "
        f"heads={model_info['num_heads']},  ds={FIXED_DATASET_SIZE},  "
        f"{NUM_RUNS}-run avg ± 1σ  |  anchored on best val-exact-acc",
        fontsize=12,
        fontweight="bold",
    )

    for ax, spec in zip(axes.flat, specs):
        name = spec[0]

        if spec[2] is not None:
            t_key, v_key = spec[1], spec[2]  # type: ignore[assignment]
            ylim = spec[3]
            t_mean = np.array(results[t_key]["mean"])
            t_std = np.array(results[t_key]["std"])
            v_mean = np.array(results[v_key]["mean"])
            v_std = np.array(results[v_key]["std"])

            ax.errorbar(
                x_values,
                t_mean,
                yerr=t_std,
                color=METRIC_COLORS["train"],
                marker="s",
                linewidth=1.5,
                markersize=6,
                capsize=3,
                label="Train",
            )
            ax.errorbar(
                x_values,
                v_mean,
                yerr=v_std,
                color=METRIC_COLORS["val"],
                marker="o",
                linewidth=1.5,
                markersize=6,
                capsize=3,
                label="Validation",
            )
        else:
            key = spec[1]
            ylim = spec[3]
            y_mean = np.array(results[key]["mean"])
            y_std = np.array(results[key]["std"])
            bar_width = 0.6 * (x_values[1] - x_values[0]) if len(x_values) > 1 else 10

            if key == "epochs_trained":
                ax.bar(
                    x_values,
                    y_mean,
                    yerr=y_std,
                    color="#2ca02c",
                    width=bar_width,
                    capsize=3,
                    alpha=0.8,
                    label="Epochs",
                )
            elif key.startswith("test_ar_"):
                ax.errorbar(
                    x_values,
                    y_mean,
                    yerr=y_std,
                    color=INFERENCE_COLOR,
                    marker="D",
                    linewidth=1.5,
                    markersize=6,
                    capsize=3,
                    label="Inference (test)",
                )
            else:
                ax.errorbar(
                    x_values,
                    y_mean,
                    yerr=y_std,
                    color=TRAIN_MAX_COLOR,
                    marker="^",
                    linewidth=1.5,
                    markersize=6,
                    capsize=3,
                    label="Max Train",
                )

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(name)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out_path}")


def _plot_max_train_exact(
    x_values: list[int],
    results: dict,
    model_info: dict,
    out_path: Path,
) -> None:
    y_mean = np.array(results["max_train_exact_acc"]["mean"])
    y_std = np.array(results["max_train_exact_acc"]["std"])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Max Train Exact Accuracy (best at any epoch)\n"
        f"layers={model_info['num_layers']},  d={model_info['hidden_size']},  "
        f"heads={model_info['num_heads']},  ds={FIXED_DATASET_SIZE},  "
        f"{NUM_RUNS}-run avg ± 1σ",
        fontsize=12,
        fontweight="bold",
    )

    bar_width = 0.6 * (x_values[1] - x_values[0]) if len(x_values) > 1 else 10
    ax.bar(
        x_values,
        y_mean,
        yerr=y_std,
        color=TRAIN_MAX_COLOR,
        width=bar_width,
        capsize=3,
        alpha=0.8,
    )
    ax.axhline(
        y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="100%"
    )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Max Train Exact Accuracy")
    ax.set_title(
        "Did the model have capacity to memorize training data?",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out_path}")


def _make_out_dir() -> Path:
    model_label = f"d{MODEL_HIDDEN_SIZE}_l{MODEL_NUM_LAYERS}_h{MODEL_NUM_HEADS}"
    sweep_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_label = f"{sweep_time}_{NUM_RUNS}runs_ds{FIXED_DATASET_SIZE}"
    out_dir = (
        Path(__file__).resolve().parent / "seq_len_results" / model_label / sweep_label
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout, force=True)
    logging.getLogger("ignite.engine").setLevel(logging.WARNING)

    base_config = load_config(Path(__file__).resolve().parent / "config.yaml")
    out_dir = _make_out_dir()
    train_charset = "@universal"

    model_info = {
        "hidden_size": MODEL_HIDDEN_SIZE,
        "num_layers": MODEL_NUM_LAYERS,
        "num_heads": MODEL_NUM_HEADS,
        "feedforward_dim": MODEL_FEEDFORWARD_DIM,
        "max_seq_len": MODEL_MAX_SEQ_LEN,
    }

    cfgs: list[tuple[int, ProjectConfig]] = [
        (sl, build_fixed_config(base_config, sl, train_charset, seed=42))
        for sl in TEST_SEQ_LENGTHS
    ]

    print(f"\n{'#' * 60}")
    print("#  MODE: Fixed Seq Len")
    print(f"{'#' * 60}")

    x_vals, results = run_suite(cfgs, TIME_LIMIT_SECONDS, "Fixed Seq Len")

    payload = {
        "x_values": x_vals,
        "num_runs": NUM_RUNS,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "model_info": model_info,
        "dataset_size": FIXED_DATASET_SIZE,
        "results": results,
    }
    (out_dir / "seq_len_results.json").write_text(json.dumps(payload, indent=2))
    print(f"JSON saved -> {out_dir / 'seq_len_results.json'}")

    plot_results(x_vals, results, model_info, out_dir / "seq_len_analysis.png")
    _plot_max_train_exact(x_vals, results, model_info, out_dir / "max_train_exact.png")

    config_record = {
        "NUM_RUNS": NUM_RUNS,
        "TIME_LIMIT_SECONDS": TIME_LIMIT_SECONDS,
        "TEST_SEQ_LENGTHS": TEST_SEQ_LENGTHS,
        "FIXED_DATASET_SIZE": FIXED_DATASET_SIZE,
        "model_info": model_info,
    }
    (out_dir / "experiment_config.json").write_text(json.dumps(config_record, indent=2))

    print(f"\nAll experiments complete. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
