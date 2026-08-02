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


def save_k_plots(df: pd.DataFrame, outdir: str) -> None:
    grouped = (
        df.groupby(["k", "md"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["md", "k"])
    )

    if grouped.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for md in sorted(grouped["md"].dropna().unique()):
        sub = grouped[grouped["md"] == md]
        axes[0].plot(sub["k"], sub["Recall@30"], marker="o", label=f"md={int(md)}")
        axes[1].plot(sub["k"], sub["Avg_Time_Per_Query_ms"], marker="o", label=f"md={int(md)}")

    axes[0].set_title("K Sweep: Recall vs k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Recall@30")
    axes[0].axhline(0.9, color="red", linestyle="--", linewidth=1)

    axes[1].set_title("K Sweep: Query Time vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Avg_Time_Per_Query_ms")

    axes[1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "k_recall_time_by_md.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        grouped["Avg_Time_Per_Query_ms"],
        grouped["Recall@30"],
        c=grouped["k"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="none",
    )
    plt.colorbar(sc, label="k")
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Avg_Time_Per_Query_ms")
    plt.ylabel("Recall@30")
    plt.title("K Sweep: Recall-Time Tradeoff")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "k_tradeoff_scatter.png"), dpi=180)
    plt.close()


def save_nbnd_plots(df: pd.DataFrame, outdir: str) -> None:
    grouped = (
        df.groupby(["nb", "nd", "md"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["md", "nb", "nd"])
    )

    if grouped.empty:
        return

    for md in sorted(grouped["md"].dropna().unique()):
        sub = grouped[grouped["md"] == md]
        recall_pivot = sub.pivot_table(index="nb", columns="nd", values="Recall@30", aggfunc="mean")
        time_pivot = sub.pivot_table(index="nb", columns="nd", values="Avg_Time_Per_Query_ms", aggfunc="mean")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        im1 = axes[0].imshow(recall_pivot.values, aspect="auto")
        axes[0].set_title(f"Recall Heatmap (md={int(md)})")
        axes[0].set_xlabel("nd")
        axes[0].set_ylabel("nb")
        axes[0].set_xticks(range(len(recall_pivot.columns)))
        axes[0].set_xticklabels([str(int(v)) for v in recall_pivot.columns])
        axes[0].set_yticks(range(len(recall_pivot.index)))
        axes[0].set_yticklabels([str(int(v)) for v in recall_pivot.index])
        plt.colorbar(im1, ax=axes[0], label="Recall@30")

        im2 = axes[1].imshow(time_pivot.values, aspect="auto")
        axes[1].set_title(f"Time Heatmap (md={int(md)})")
        axes[1].set_xlabel("nd")
        axes[1].set_ylabel("nb")
        axes[1].set_xticks(range(len(time_pivot.columns)))
        axes[1].set_xticklabels([str(int(v)) for v in time_pivot.columns])
        axes[1].set_yticks(range(len(time_pivot.index)))
        axes[1].set_yticklabels([str(int(v)) for v in time_pivot.index])
        plt.colorbar(im2, ax=axes[1], label="Avg_Time_Per_Query_ms")

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"nb_nd_heatmaps_md_{int(md)}.png"), dpi=180)
        plt.close()

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        grouped["Avg_Time_Per_Query_ms"],
        grouped["Recall@30"],
        c=grouped["md"],
        cmap="plasma",
        alpha=0.85,
        edgecolors="none",
    )
    for _, row in grouped.iterrows():
        label = f"nb={int(row['nb'])},nd={int(row['nd'])}"
        plt.annotate(label, (row["Avg_Time_Per_Query_ms"], row["Recall@30"]), fontsize=7, alpha=0.7)
    plt.colorbar(sc, label="md")
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Avg_Time_Per_Query_ms")
    plt.ylabel("Recall@30")
    plt.title("NB/ND Sweep: Recall-Time Tradeoff")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "nb_nd_tradeoff_scatter.png"), dpi=180)
    plt.close()


def save_search_plots(df: pd.DataFrame, outdir: str) -> None:
    # md behavior under fixed mqt=0, msb=150
    md_base = (
        df[(df["mqt"] == 0) & (df["msb"] == 150)]
        .groupby("md", as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values("md")
    )

    if not md_base.empty:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()
        ax1.plot(md_base["md"], md_base["Recall@30"], marker="o", color="#1f77b4", label="Recall@30")
        ax2.plot(md_base["md"], md_base["Avg_Time_Per_Query_ms"], marker="x", linestyle="--", color="#ff7f0e", label="Query Time")
        ax1.axhline(0.9, color="red", linestyle=":", linewidth=1)
        ax1.set_xlabel("md")
        ax1.set_ylabel("Recall@30")
        ax2.set_ylabel("Avg_Time_Per_Query_ms")
        ax1.set_title("Search: md effect (mqt=0, msb=150)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "search_md_effect.png"), dpi=180)
        plt.close()

    # mqt behavior under fixed msb=150
    mqt_df = (
        df[df["msb"] == 150]
        .groupby(["md", "mqt"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["md", "mqt"])
    )

    # Keep only md groups that actually sweep mqt (>= 2 points).
    if not mqt_df.empty:
        mqt_counts = mqt_df.groupby("md")["mqt"].nunique()
        valid_md_mqt = mqt_counts[mqt_counts >= 2].index
        mqt_df = mqt_df[mqt_df["md"].isin(valid_md_mqt)]

    if not mqt_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for md in sorted(mqt_df["md"].dropna().unique()):
            sub = mqt_df[mqt_df["md"] == md]
            axes[0].plot(sub["mqt"], sub["Recall@30"], marker="o", label=f"md={int(md)}")
            axes[1].plot(sub["mqt"], sub["Avg_Time_Per_Query_ms"], marker="o", label=f"md={int(md)}")
        axes[0].axhline(0.9, color="red", linestyle="--", linewidth=1)
        axes[0].set_title("Search: mqt effect on Recall")
        axes[0].set_xlabel("mqt")
        axes[0].set_ylabel("Recall@30")
        axes[1].set_title("Search: mqt effect on Query Time")
        axes[1].set_xlabel("mqt")
        axes[1].set_ylabel("Avg_Time_Per_Query_ms")
        axes[1].legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "search_mqt_effect.png"), dpi=180)
        plt.close()

    # msb behavior under fixed mqt=0
    msb_df = (
        df[df["mqt"] == 0]
        .groupby(["md", "msb"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["md", "msb"])
    )

    # Keep only md groups that actually sweep msb (>= 2 points).
    if not msb_df.empty:
        msb_counts = msb_df.groupby("md")["msb"].nunique()
        valid_md_msb = msb_counts[msb_counts >= 2].index
        msb_df = msb_df[msb_df["md"].isin(valid_md_msb)]

    if not msb_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for md in sorted(msb_df["md"].dropna().unique()):
            sub = msb_df[msb_df["md"] == md]
            axes[0].plot(sub["msb"], sub["Recall@30"], marker="o", label=f"md={int(md)}")
            axes[1].plot(sub["msb"], sub["Avg_Time_Per_Query_ms"], marker="o", label=f"md={int(md)}")
        axes[0].axhline(0.9, color="red", linestyle="--", linewidth=1)
        axes[0].set_title("Search: msb effect on Recall")
        axes[0].set_xlabel("msb")
        axes[0].set_ylabel("Recall@30")
        axes[1].set_title("Search: msb effect on Query Time")
        axes[1].set_xlabel("msb")
        axes[1].set_ylabel("Avg_Time_Per_Query_ms")
        axes[1].legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "search_msb_effect.png"), dpi=180)
        plt.close()

    # Recall/time heatmaps for md x msb with mqt=0
    joint = (
        df[df["mqt"] == 0]
        .groupby(["md", "msb"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
    )

    if not joint.empty:
        recall_pivot = joint.pivot_table(index="md", columns="msb", values="Recall@30", aggfunc="mean")
        time_pivot = joint.pivot_table(index="md", columns="msb", values="Avg_Time_Per_Query_ms", aggfunc="mean")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        im1 = axes[0].imshow(recall_pivot.values, aspect="auto")
        axes[0].set_title("Search Heatmap: Recall@30 (mqt=0)")
        axes[0].set_xlabel("msb")
        axes[0].set_ylabel("md")
        axes[0].set_xticks(range(len(recall_pivot.columns)))
        axes[0].set_xticklabels([str(int(v)) for v in recall_pivot.columns], rotation=45, ha="right")
        axes[0].set_yticks(range(len(recall_pivot.index)))
        axes[0].set_yticklabels([str(int(v)) for v in recall_pivot.index])
        plt.colorbar(im1, ax=axes[0], label="Recall@30")

        im2 = axes[1].imshow(time_pivot.values, aspect="auto")
        axes[1].set_title("Search Heatmap: Query Time (mqt=0)")
        axes[1].set_xlabel("msb")
        axes[1].set_ylabel("md")
        axes[1].set_xticks(range(len(time_pivot.columns)))
        axes[1].set_xticklabels([str(int(v)) for v in time_pivot.columns], rotation=45, ha="right")
        axes[1].set_yticks(range(len(time_pivot.index)))
        axes[1].set_yticklabels([str(int(v)) for v in time_pivot.index])
        plt.colorbar(im2, ax=axes[1], label="Avg_Time_Per_Query_ms")

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "search_md_msb_heatmaps.png"), dpi=180)
        plt.close()

    # Global tradeoff scatter
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        df["Avg_Time_Per_Query_ms"],
        df["Recall@30"],
        c=df["md"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    plt.colorbar(sc, label="md")
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Avg_Time_Per_Query_ms")
    plt.ylabel("Recall@30")
    plt.title("Search Sweep: Recall-Time Tradeoff")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "search_tradeoff_scatter.png"), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for static experiments (k, indexing, search)")
    parser.add_argument("--k-csv", required=False, help="Path to static_exps_k.csv")
    parser.add_argument("--nbnd-csv", required=False, help="Path to static_exps_nb_nd.csv")
    parser.add_argument("--search-csv", required=False, help="Path to static_exps_search_v1.csv")
    parser.add_argument("--outdir", default="figures/static_exps_analysis", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    k_df = ensure_numeric(pd.read_csv(args.k_csv), ["k", "md", "Recall@30", "Avg_Time_Per_Query_ms"]) if args.k_csv else None
    nbnd_df = ensure_numeric(pd.read_csv(args.nbnd_csv), ["nb", "nd", "md", "Recall@30", "Avg_Time_Per_Query_ms"]) if args.nbnd_csv else None
    search_df = ensure_numeric(pd.read_csv(args.search_csv), ["md", "mqt", "msb", "Recall@30", "Avg_Time_Per_Query_ms"]) if args.search_csv else None

    if k_df is not None:
        save_k_plots(k_df, args.outdir)
    if nbnd_df is not None:
        save_nbnd_plots(nbnd_df, args.outdir)
    if search_df is not None:
        save_search_plots(search_df, args.outdir)


if __name__ == "__main__":
    main()
