import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    # 1. Load the results
    csv_path = "sweep_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # 2. Pivot the data to create a matrix of depth vs k
    pivot_df = df.pivot(index="depth", columns="k", values="best_val_sequence_acc")

    plt.figure(figsize=(12, 9))

    # Set style
    sns.set_theme(style="white")

    # Plot heatmap, masking NaNs (which will default to the facecolor)
    ax = sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Best Validation Sequence Accuracy"},
        linewidths=0.5,
        linecolor="#BBBBBB",
        vmin=0.0,
        vmax=1.0,
        mask=pivot_df.isnull(),
        annot_kws={"size": 10, "weight": "bold"},
    )

    # Make un-run configurations (NaNs) light gray to highlight targeted evaluations
    ax.set_facecolor("#EBEBEB")

    # Invert y-axis to place Depth 1 at the top
    plt.gca().invert_yaxis()

    # Draw red outline patches around the theoretical limit boundary k = d + 2
    depths = sorted(df["depth"].unique())
    for d in depths:
        # Row index in pivot: d (1-indexed index mapping)
        # Column index in pivot: k = d + 2
        # Since depth is 1 to 10 and k is 3 to 14:
        # Row index in plot coords is d - 1
        # Column index in plot coords is (d + 2) - 3 = d - 1
        rect = patches.Rectangle(
            (d - 1, d - 1),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=3,
            clip_on=False,
            zorder=5,
        )
        ax.add_patch(rect)

    # Add legend entry for the boundary outline
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor="none", edgecolor="red", linewidth=3, label="Theoretical Bound (k = d + 2)")]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=10)

    plt.title(
        "Sequence-Level Exact Match Accuracy (Targeted Sweep)\nRed Outline = Theoretical Limit Boundary (k = d + 2)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Target Language Alternating Block Count (k)", fontsize=12, labelpad=12)
    plt.ylabel("Model Depth / Transformer Layers (d)", fontsize=12, labelpad=12)

    plt.tight_layout()

    # Define paths
    artifact_dir = "/home/taha/.gemini/antigravity-cli/brain/49902a92-bf82-47eb-89a8-d38055e1254a"
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_path = os.path.join(artifact_dir, "sweep_heatmap.png")
    local_path = "sweep_heatmap.png"

    # Save plots
    plt.savefig(artifact_path, dpi=300)
    plt.savefig(local_path, dpi=300)
    plt.close()

    print("Heatmap visualization generated successfully!")
    print(f"Artifact Path: {artifact_path}")
    print(f"Local Path: {local_path}")


if __name__ == "__main__":
    main()
