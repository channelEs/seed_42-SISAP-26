#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def save_scatter(df: pd.DataFrame, outdir: str) -> None:
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        df["Avg_Time_Per_Query_ms"],
        df["Recall@30"],
        c=df["md"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("md")
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1, label="Recall 0.9")
    plt.xlabel("Avg Time per Query (ms)")
    plt.ylabel("Recall@30")
    plt.title("Recall vs Query Time (color = md)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_scatter_recall_vs_time.png"), dpi=150)
    plt.close()


def save_md_tradeoff(df: pd.DataFrame, outdir: str) -> None:
    grouped = (
        df.groupby(["itr", "md"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["itr", "md"])
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for itr in sorted(grouped["itr"].dropna().unique()):
        sub = grouped[grouped["itr"] == itr]
        ax1.plot(sub["md"], sub["Recall@30"], marker="o", label=f"Recall (itr={int(itr)})")
        ax2.plot(sub["md"], sub["Avg_Time_Per_Query_ms"], marker="x", linestyle="--", label=f"Time (itr={int(itr)})")

    ax1.set_xlabel("md")
    ax1.set_ylabel("Recall@30")
    ax2.set_ylabel("Avg Time per Query (ms)")
    ax1.axhline(0.9, color="red", linestyle=":", linewidth=1)
    ax1.set_title("Trade-off by md")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_tradeoff_by_md.png"), dpi=150)
    plt.close()


def save_top10_bar(df: pd.DataFrame, outdir: str) -> None:
    top = (
        df[df["Recall@30"] >= 0.9]
        .sort_values(["Avg_Time_Per_Query_ms", "Recall@30"], ascending=[True, False])
        .head(10)
        .copy()
    )

    if top.empty:
        return

    top["label"] = top.apply(
        lambda r: f"itr={int(r['itr'])}|nb={int(r['nb'])}|nd={int(r['nd'])}|md={int(r['md'])}|hf={r['heap_factor']:.2f}",
        axis=1,
    )

    plt.figure(figsize=(12, 7))
    bars = plt.barh(top["label"], top["Avg_Time_Per_Query_ms"], color="#2a9d8f")
    plt.gca().invert_yaxis()
    plt.xlabel("Avg Time per Query (ms)")
    plt.title("Top-10 Fastest Configurations with Recall@30 >= 0.9")

    for bar, rec in zip(bars, top["Recall@30"]):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"R={rec:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_top10_fastest_ge_0_9.png"), dpi=150)
    plt.close()


def save_heatmap(df: pd.DataFrame, outdir: str) -> None:
    sub = df[(df["itr"] == 3) & (df["md"] == 60000)].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(index="nb", columns="nd", values="Recall@30", aggfunc="mean")

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Recall@30")
    plt.xticks(range(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(int(i)) for i in pivot.index])
    plt.xlabel("nd")
    plt.ylabel("nb")
    plt.title("Recall Heatmap (itr=3, md=60000)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_heatmap_itr3_md60000.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for fixed_engine_experiments results")
    parser.add_argument("--csv", required=True, help="Path to results CSV")
    parser.add_argument("--outdir", default="figures/fixed_engine_analysis", help="Output directory for generated plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    numeric_cols = [
        "k", "itr", "nb", "nd", "md", "heap_factor",
        "Recall@30", "Avg_Time_Per_Query_ms", "Avg_Docs_Examined",
        "Avg_Blocks_Entered", "Avg_Blocks_Skipped",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    save_scatter(df, args.outdir)
    save_md_tradeoff(df, args.outdir)
    save_top10_bar(df, args.outdir)
    save_heatmap(df, args.outdir)


if __name__ == "__main__":
    main()
