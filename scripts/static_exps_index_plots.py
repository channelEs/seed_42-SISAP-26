#!/usr/bin/env python3
import argparse
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FOCUS_K = [100, 120, 140]
FOCUS_NB = [100, 150]
FOCUS_ND = [90, 110, 140]
K_COLORS = {
    100: "#1f77b4",
    120: "#ff7f0e",
    140: "#2ca02c",
}


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_and_filter(csv_paths: list[str], md: int) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        frame = ensure_numeric(
            pd.read_csv(path),
            ["k", "nb", "nd", "md", "Recall@30", "Avg_Time_Per_Query_ms", "Avg_Docs_Examined"],
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    filtered = combined[
        combined["k"].isin(FOCUS_K)
        & combined["nb"].isin(FOCUS_NB)
        & combined["nd"].isin(FOCUS_ND)
        & (combined["md"] == md)
    ].copy()

    return (
        filtered.groupby(["k", "nb", "nd"], as_index=False)
        .agg(
            {
                "Recall@30": "mean",
                "Avg_Time_Per_Query_ms": "mean",
                "Avg_Docs_Examined": "mean",
            }
        )
        .sort_values(["k", "nb", "nd"])
    )


def build_k_split_heatmap(
    grouped: pd.DataFrame,
    value_column: str,
    feasible_only: bool = False,
) -> tuple[np.ndarray, list[list[str]], list[str]]:
    """Builds a heatmap matrix splitting each (nb, nd) combination into sub-cells per k value."""
    num_rows = len(FOCUS_NB) * len(FOCUS_K)
    num_cols = len(FOCUS_ND)
    heatmap = np.full((num_rows, num_cols), np.nan)
    labels = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    row_labels = []
    for nb in FOCUS_NB:
        for k in FOCUS_K:
            row_labels.append(f"nb={nb}, k={k}")

    for nb_idx, nb in enumerate(FOCUS_NB):
        for k_idx, k in enumerate(FOCUS_K):
            row_idx = nb_idx * len(FOCUS_K) + k_idx
            for nd_idx, nd in enumerate(FOCUS_ND):
                sub = grouped[
                    (grouped["nb"] == nb) & (grouped["nd"] == nd) & (grouped["k"] == k)
                ]
                if sub.empty:
                    labels[row_idx][nd_idx] = "-"
                    continue

                row = sub.iloc[0]
                value = float(row[value_column])
                recall = float(row["Recall@30"])

                if feasible_only and recall < 0.9:
                    labels[row_idx][nd_idx] = "X"
                    continue

                heatmap[row_idx, nd_idx] = value
                if value_column == "Avg_Docs_Examined":
                    labels[row_idx][nd_idx] = f"{int(round(value))}"
                else:
                    labels[row_idx][nd_idx] = f"{value:.1f}"

    return heatmap, labels, row_labels


def annotate_heatmap_cells(
    ax: plt.Axes,
    heatmap: np.ndarray,
    labels: list[list[str]],
    cmap_name: str = "Blues",
) -> None:
    """Annotates each cell with text, adjusting contrast dynamically based on background color."""
    num_rows, num_cols = heatmap.shape
    cmap = plt.get_cmap(cmap_name)

    valid_vals = heatmap[~np.isnan(heatmap)]
    vmin = np.min(valid_vals) if len(valid_vals) > 0 else 0
    vmax = np.max(valid_vals) if len(valid_vals) > 0 else 1

    for r in range(num_rows):
        for c in range(num_cols):
            text = labels[r][c]
            val = heatmap[r, c]

            if np.isnan(val):
                ax.text(c, r, text, ha="center", va="center", fontsize=8.5, color="#555555")
            else:
                norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                rgba = cmap(norm_val)
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "black" if luminance > 0.5 else "white"

                ax.text(
                    c,
                    r,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=text_color,
                )


def save_summary_plot(grouped: pd.DataFrame, md: int, output_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

    # ---------------------------------------------------------
    # Plot 1: Recall-Time Tradeoff Scatter Plot
    # ---------------------------------------------------------
    scatter_ax = axes[0]
    for k in FOCUS_K:
        sub = grouped[grouped["k"] == k]
        if sub.empty:
            continue
        scatter_ax.scatter(
            sub["Avg_Time_Per_Query_ms"],
            sub["Recall@30"],
            s=50,
            alpha=0.9,
            color=K_COLORS[k],
            edgecolors="black",
            linewidths=0.8,
            label=f"k={k}",
        )
        for _, row in sub.iterrows():
            scatter_ax.annotate(
                f"nb={int(row['nb'])}, nd={int(row['nd'])}",
                (row["Avg_Time_Per_Query_ms"], row["Recall@30"]),
                xytext=(6, 3),
                textcoords="offset points",
                fontsize=8,
                color="black",
            )

    scatter_ax.axhline(0.9, color="red", linestyle="--", linewidth=1)
    scatter_ax.set_xlabel("Avg_Time_Per_Query_ms")
    scatter_ax.set_ylabel("Recall@30")
    scatter_ax.set_title(f"Recall-Time Tradeoff (md={md})")
    scatter_ax.legend(loc="lower right", fontsize=9)
    scatter_ax.grid(alpha=0.2)

    # ---------------------------------------------------------
    # Plot 2: Avg Docs Examined Heatmap (Split by k)
    # ---------------------------------------------------------
    docs_heatmap, docs_labels, row_y_labels = build_k_split_heatmap(grouped, "Avg_Docs_Examined")
    docs_ax = axes[1]
    docs_im = docs_ax.imshow(docs_heatmap, aspect="auto", cmap="Blues")
    docs_ax.set_title(f"Avg Docs Examined (md={md})")
    docs_ax.set_xlabel("nd")
    docs_ax.set_ylabel("Configuration (nb, k)")
    docs_ax.set_xticks(range(len(FOCUS_ND)))
    docs_ax.set_xticklabels([str(v) for v in FOCUS_ND])
    docs_ax.set_yticks(range(len(row_y_labels)))
    docs_ax.set_yticklabels(row_y_labels)

    # Add visual separators between different nb groups
    for i in range(1, len(FOCUS_NB)):
        docs_ax.axhline(i * len(FOCUS_K) - 0.5, color="black", linewidth=1.5)

    annotate_heatmap_cells(docs_ax, docs_heatmap, docs_labels, cmap_name="Blues")
    fig.colorbar(docs_im, ax=docs_ax, label="Avg_Docs_Examined", fraction=0.046, pad=0.04)

    # ---------------------------------------------------------
    # Plot 3: Feasible Time Heatmap (Recall >= 0.9)
    # ---------------------------------------------------------
    feasible_heatmap, feasible_labels, _ = build_k_split_heatmap(
        grouped,
        "Avg_Time_Per_Query_ms",
        feasible_only=True,
    )
    feasible_ax = axes[2]
    feasible_im = feasible_ax.imshow(feasible_heatmap, aspect="auto", cmap="cividis_r")
    feasible_ax.set_title("Feasible Time (Recall >= 0.9)")
    feasible_ax.set_xlabel("nd")
    feasible_ax.set_ylabel("Configuration (nb, k)")
    feasible_ax.set_xticks(range(len(FOCUS_ND)))
    feasible_ax.set_xticklabels([str(v) for v in FOCUS_ND])
    feasible_ax.set_yticks(range(len(row_y_labels)))
    feasible_ax.set_yticklabels(row_y_labels)

    # Add visual separators between different nb groups
    for i in range(1, len(FOCUS_NB)):
        feasible_ax.axhline(i * len(FOCUS_K) - 0.5, color="white", linewidth=1.5)

    annotate_heatmap_cells(feasible_ax, feasible_heatmap, feasible_labels, cmap_name="cividis_r")
    fig.colorbar(feasible_im, ax=feasible_ax, label="Avg_Time_Per_Query_ms", fraction=0.046, pad=0.04)

    fig.suptitle("Index Building Restricted Configurations", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate combined index-building plots for the paper")
    parser.add_argument("--csv", nargs="+", required=True, help="One or more CSV files to combine")
    parser.add_argument("--md", type=int, default=75000, help="md value to filter")
    parser.add_argument(
        "--outdir",
        default="figures/restricted_index_static_exps_analysis",
        help="Output directory for the combined figure",
    )
    parser.add_argument(
        "--output-name",
        default="static_exps_index_summary.png",
        help="Filename for the output PNG",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    grouped = load_and_filter(args.csv, args.md)
    if grouped.empty:
        raise SystemExit("No rows matched the selected k/nb/nd/md filter.")

    output_path = os.path.join(args.outdir, args.output_name)
    save_summary_plot(grouped, args.md, output_path)


if __name__ == "__main__":
    main()