#!/usr/bin/env python3
import argparse
import os
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_per_k_heatmaps(df: pd.DataFrame, outdir: str) -> None:
    grouped = (
        df.groupby(["k", "nb", "nd"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["k", "nb", "nd"])
    )

    for k in sorted(grouped["k"].dropna().unique()):
        sub = grouped[grouped["k"] == k]
        recall_pivot = sub.pivot_table(index="nb", columns="nd", values="Recall@30", aggfunc="mean")
        time_pivot = sub.pivot_table(index="nb", columns="nd", values="Avg_Time_Per_Query_ms", aggfunc="mean")
        feasible_pivot = time_pivot.where(recall_pivot >= 0.9)

        best_feasible = sub[sub["Recall@30"] >= 0.9].sort_values("Avg_Time_Per_Query_ms").head(1)
        best_nb = None
        best_nd = None
        if not best_feasible.empty:
            best_nb = int(best_feasible.iloc[0]["nb"])
            best_nd = int(best_feasible.iloc[0]["nd"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im1 = axes[0].imshow(recall_pivot.values, aspect="auto", vmin=0.75, vmax=1.0, cmap="viridis")
        axes[0].set_title(f"Recall Heatmap (k={int(k)})")
        axes[0].set_xlabel("nd")
        axes[0].set_ylabel("nb")
        axes[0].set_xticks(range(len(recall_pivot.columns)))
        axes[0].set_xticklabels([str(int(v)) for v in recall_pivot.columns])
        axes[0].set_yticks(range(len(recall_pivot.index)))
        axes[0].set_yticklabels([str(int(v)) for v in recall_pivot.index])
        for i, nb in enumerate(recall_pivot.index):
            for j, nd in enumerate(recall_pivot.columns):
                val = recall_pivot.loc[nb, nd]
                if pd.notna(val):
                    axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(im1, ax=axes[0], label="Recall@30")

        im2 = axes[1].imshow(time_pivot.values, aspect="auto", cmap="magma_r")
        axes[1].set_title(f"Time Heatmap (k={int(k)})")
        axes[1].set_xlabel("nd")
        axes[1].set_ylabel("nb")
        axes[1].set_xticks(range(len(time_pivot.columns)))
        axes[1].set_xticklabels([str(int(v)) for v in time_pivot.columns])
        axes[1].set_yticks(range(len(time_pivot.index)))
        axes[1].set_yticklabels([str(int(v)) for v in time_pivot.index])
        for i, nb in enumerate(time_pivot.index):
            for j, nd in enumerate(time_pivot.columns):
                val = time_pivot.loc[nb, nd]
                if pd.notna(val):
                    axes[1].text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=8)
        if best_nb is not None and best_nd is not None:
            i = list(time_pivot.index).index(best_nb)
            j = list(time_pivot.columns).index(best_nd)
            axes[1].scatter(j, i, marker="*", s=260, c="cyan", edgecolors="black", linewidths=0.8)
        plt.colorbar(im2, ax=axes[1], label="Avg_Time_Per_Query_ms")

        im3 = axes[2].imshow(feasible_pivot.values, aspect="auto", cmap="cividis_r")
        axes[2].set_title("Feasible Time (Recall>=0.9)")
        axes[2].set_xlabel("nd")
        axes[2].set_ylabel("nb")
        axes[2].set_xticks(range(len(feasible_pivot.columns)))
        axes[2].set_xticklabels([str(int(v)) for v in feasible_pivot.columns])
        axes[2].set_yticks(range(len(feasible_pivot.index)))
        axes[2].set_yticklabels([str(int(v)) for v in feasible_pivot.index])
        for i, nb in enumerate(feasible_pivot.index):
            for j, nd in enumerate(feasible_pivot.columns):
                val = feasible_pivot.loc[nb, nd]
                label = "X" if pd.isna(val) else f"{val:.1f}"
                color = "black" if pd.isna(val) else "white"
                axes[2].text(j, i, label, ha="center", va="center", color=color, fontsize=8)
        if best_nb is not None and best_nd is not None:
            i = list(feasible_pivot.index).index(best_nb)
            j = list(feasible_pivot.columns).index(best_nd)
            axes[2].scatter(j, i, marker="*", s=260, c="yellow", edgecolors="black", linewidths=0.8)
        plt.colorbar(im3, ax=axes[2], label="Avg_Time_Per_Query_ms")

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"k_{int(k)}_nb_nd_sweetspot.png"), dpi=180)
        plt.close()


def save_tradeoff_plot(df: pd.DataFrame, outdir: str) -> None:
    grouped = (
        df.groupby(["k", "nb", "nd"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["k", "Avg_Time_Per_Query_ms"])
    )
    grouped["feasible"] = grouped["Recall@30"] >= 0.9

    plt.figure(figsize=(9, 6))
    for k in sorted(grouped["k"].dropna().unique()):
        sub = grouped[grouped["k"] == k]
        plt.scatter(
            sub["Avg_Time_Per_Query_ms"],
            sub["Recall@30"],
            s=60,
            alpha=0.8,
            label=f"k={int(k)}",
        )
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Avg_Time_Per_Query_ms")
    plt.ylabel("Recall@30")
    plt.title("Round-04 Search: Recall-Time Tradeoff")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "round04_tradeoff_by_k.png"), dpi=180)
    plt.close()

    best = grouped[grouped["feasible"]].sort_values("Avg_Time_Per_Query_ms").head(20)
    best.to_csv(os.path.join(outdir, "round04_fastest_feasible_top20.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for static_exps_04_search")
    parser.add_argument("--csv", required=True, help="Path to static_exps_04_search.csv")
    parser.add_argument("--outdir", default="figures/04_static_exps_analysis", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = ensure_numeric(
        pd.read_csv(args.csv),
        ["k", "nb", "nd", "Recall@30", "Avg_Time_Per_Query_ms"],
    )

    save_per_k_heatmaps(df, args.outdir)
    save_tradeoff_plot(df, args.outdir)


if __name__ == "__main__":
    main()