import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import argparse
from matplotlib.patches import Patch


def format_lr(lr: float) -> str:
    """Format learning rate cleanly (e.g. 0.0001 -> 1e-4, 0.00001 -> 1e-5)."""
    s = f"{lr:.0e}"
    return s.replace("-0", "-").replace("+0", "+").replace("+", "")


def save_heatmap(df_pivot, filename, title):
    plt.figure(figsize=(12, 9))
    sns.set_theme(style="white")

    # Plot heatmap
    ax = sns.heatmap(
        df_pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Sequence Accuracy"},
        linewidths=0.5,
        linecolor="#BBBBBB",
        vmin=0.0,
        vmax=1.0,
        mask=df_pivot.isnull(),
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_facecolor("#EBEBEB")

    # Place x-axis on top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Draw red outlines around the theoretical limit boundary k = d + 2
    for col_idx, d in enumerate(df_pivot.columns):
        target_k = d + 2
        if target_k in df_pivot.index:
            row_idx = df_pivot.index.get_loc(target_k)
            rect = patches.Rectangle(
                (col_idx, row_idx),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=3,
                clip_on=False,
                zorder=5,
            )
            ax.add_patch(rect)

    # Legend placed below the x-axis
    legend_elements = [Patch(facecolor="none", edgecolor="red", linewidth=3, label="Theoretical Bound (k = d + 2)")]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), framealpha=0.9, fontsize=10)

    plt.title(title, fontsize=14, fontweight="bold", pad=45)
    plt.xlabel("Model Depth / Transformer Layers (d)", fontsize=12, labelpad=15)
    plt.ylabel("Target Language Alternating Block Count (k)", fontsize=12, labelpad=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Generated plot: {filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot counting targeted sweep results heatmaps")
    parser.add_argument("--input-csv", type=str, default="sweep.csv", help="Path to the sweep results CSV file")
    args = parser.parse_args()

    csv_path = args.input_csv
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Dictionary mapping CSV columns to output suffix and base title
    metrics = {
        "best_val_sequence_acc": ("201_250", "Validation Accuracy (Lengths 201-250)"),
        "test_251_300": ("251_300", "OOD Test Accuracy (Lengths 251-300)"),
        "test_301_350": ("301_350", "OOD Test Accuracy (Lengths 301-350)"),
        "test_351_400": ("351_400", "OOD Test Accuracy (Lengths 351-400)"),
    }

    # 1. Plot Individual Configurations
    configs = df[["dim", "lr"]].drop_duplicates()
    for _, row in configs.iterrows():
        dim = int(row["dim"])
        lr = float(row["lr"])

        # Filter for this config
        df_filtered = df[(df["dim"] == dim) & (df["lr"] == lr)]
        if df_filtered.empty:
            continue

        lr_str = format_lr(lr)
        print(f"\nPlotting configuration: dim={dim}, lr={lr_str}...")
        for metric_col, (suffix, title) in metrics.items():
            if metric_col not in df_filtered.columns:
                continue

            pivot_df = df_filtered.pivot(index="k", columns="depth", values=metric_col)
            filename = f"sweep_heatmap_{suffix}_dim{dim}_lr{lr_str}.png"
            full_title = f"{title} (Dim={dim}, LR={lr_str})\nRed Outline = Theoretical Limit Boundary (k = d + 2)"
            save_heatmap(pivot_df, filename, full_title)

    # 2. Plot Best Aggregated Configuration
    print("\nPlotting best aggregated configuration across all sweeps...")
    best_rows = []
    for (depth, k), group in df.groupby(["depth", "k"]):
        sorted_group = group.sort_values(
            by=["best_val_sequence_acc", "test_251_300", "test_301_350", "test_351_400"], ascending=False
        )
        best_rows.append(sorted_group.iloc[0])

    best_df = pd.DataFrame(best_rows)
    best_csv_path = "sweep_results_best.csv"
    best_df.to_csv(best_csv_path, index=False)
    print(f"Saved aggregated best results to {best_csv_path}.")

    for metric_col, (suffix, title) in metrics.items():
        if metric_col not in best_df.columns:
            continue
        pivot_df = best_df.pivot(index="k", columns="depth", values=metric_col)
        filename = f"sweep_heatmap_{suffix}_best.png"
        full_title = f"{title} (Aggregated Best Configuration)\nRed Outline = Theoretical Limit Boundary (k = d + 2)"
        save_heatmap(pivot_df, filename, full_title)


if __name__ == "__main__":
    main()
