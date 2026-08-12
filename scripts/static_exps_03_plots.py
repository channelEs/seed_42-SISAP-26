#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


INDEX_COMBINATIONS: Dict[Tuple[int, int, int], str] = {
    (500, 500, 400): "I-1",
    (500, 750, 75): "I-2",
    (500, 100, 750): "I-3",
    (1000, 500, 400): "I-4",
    (1000, 750, 75): "I-5",
    (1000, 100, 750): "I-6",
    (1500, 500, 400): "I-7",
    (1500, 750, 75): "I-8",
    (1500, 100, 750): "I-9",
}

INDEX_COLORS: Dict[str, str] = {
    "I-1": "#1f77b4",
    "I-2": "#ff7f0e",
    "I-3": "#2ca02c",
    "I-4": "#d62728",
    "I-5": "#9467bd",
    "I-6": "#8c564b",
    "I-7": "#e377c2",
    "I-8": "#7f7f7f",
    "I-9": "#bcbd22",
}


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def annotate_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    def to_index_id(row: pd.Series) -> str:
        key = (int(row["k"]), int(row["nb"]), int(row["nd"]))
        return INDEX_COMBINATIONS.get(key, "UNKNOWN")

    df = df.copy()
    df["index_id"] = df.apply(to_index_id, axis=1)
    return df


def index_legend_label(index_id: str) -> str:
    inv = {v: k for k, v in INDEX_COMBINATIONS.items()}
    if index_id not in inv:
        return index_id
    k, nb, nd = inv[index_id]
    return f"{index_id} (k={k}, nb={nb}, nd={nd})"


def save_k_plot(k_df: pd.DataFrame, outdir: str) -> None:
    grouped = (
        k_df.groupby(["k", "md"], as_index=False)
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


def save_search_md_effect(search_df: pd.DataFrame, outdir: str) -> None:
    md_df = (
        search_df[(search_df["mqt"] == 0) & (search_df["msb"] == 150)]
        .groupby(["index_id", "md"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["index_id", "md"])
    )

    md_df = md_df[md_df["index_id"] != "UNKNOWN"]
    if md_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for index_id in [f"I-{i}" for i in range(1, 10)]:
        sub = md_df[md_df["index_id"] == index_id]
        if sub.empty:
            continue
        color = INDEX_COLORS[index_id]
        label = index_legend_label(index_id)
        axes[0].plot(sub["md"], sub["Recall@30"], marker="o", color=color, label=label)
        axes[1].plot(sub["md"], sub["Avg_Time_Per_Query_ms"], marker="o", color=color, label=label)

    axes[0].set_title("Search: md effect on Recall")
    axes[0].set_xlabel("md")
    axes[0].set_ylabel("Recall@30")
    axes[0].axhline(0.9, color="red", linestyle="--", linewidth=1)

    axes[1].set_title("Search: md effect on Query Time")
    axes[1].set_xlabel("md")
    axes[1].set_ylabel("Avg_Time_Per_Query_ms")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.86, 1])
    else:
        plt.tight_layout()

    plt.savefig(os.path.join(outdir, "search_md_effect.png"), dpi=180)
    plt.close()


def save_search_mqt_effect(search_df: pd.DataFrame, outdir: str) -> None:
    mqt_df = (
        search_df[(search_df["md"] == 100000) & (search_df["msb"] == 150)]
        .groupby(["index_id", "mqt"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["index_id", "mqt"])
    )

    mqt_df = mqt_df[mqt_df["index_id"] != "UNKNOWN"]
    if mqt_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for index_id in [f"I-{i}" for i in range(1, 10)]:
        sub = mqt_df[mqt_df["index_id"] == index_id]
        if sub.empty:
            continue
        color = INDEX_COLORS[index_id]
        label = index_legend_label(index_id)
        axes[0].plot(sub["mqt"], sub["Recall@30"], marker="o", color=color, label=label)
        axes[1].plot(sub["mqt"], sub["Avg_Time_Per_Query_ms"], marker="o", color=color, label=label)

    axes[0].set_title("Search: mqt effect on Recall")
    axes[0].set_xlabel("mqt")
    axes[0].set_ylabel("Recall@30")
    axes[0].axhline(0.9, color="red", linestyle="--", linewidth=1)

    axes[1].set_title("Search: mqt effect on Query Time")
    axes[1].set_xlabel("mqt")
    axes[1].set_ylabel("Avg_Time_Per_Query_ms")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.86, 1])
    else:
        plt.tight_layout()

    plt.savefig(os.path.join(outdir, "search_mqt_effect.png"), dpi=180)
    plt.close()


def save_search_msb_effect(search_df: pd.DataFrame, outdir: str) -> None:
    msb_df = (
        search_df[(search_df["md"] == 100000) & (search_df["mqt"] == 0)]
        .groupby(["index_id", "msb"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
        })
        .sort_values(["index_id", "msb"])
    )

    msb_df = msb_df[msb_df["index_id"] != "UNKNOWN"]
    if msb_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for index_id in [f"I-{i}" for i in range(1, 10)]:
        sub = msb_df[msb_df["index_id"] == index_id]
        if sub.empty:
            continue
        color = INDEX_COLORS[index_id]
        label = index_legend_label(index_id)
        axes[0].plot(sub["msb"], sub["Recall@30"], marker="o", color=color, label=label)
        axes[1].plot(sub["msb"], sub["Avg_Time_Per_Query_ms"], marker="o", color=color, label=label)

    axes[0].set_title("Search: msb effect on Recall")
    axes[0].set_xlabel("msb")
    axes[0].set_ylabel("Recall@30")
    axes[0].axhline(0.9, color="red", linestyle="--", linewidth=1)

    axes[1].set_title("Search: msb effect on Query Time")
    axes[1].set_xlabel("msb")
    axes[1].set_ylabel("Avg_Time_Per_Query_ms")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.86, 1])
    else:
        plt.tight_layout()

    plt.savefig(os.path.join(outdir, "search_msb_effect.png"), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate round-3 static experiment plots")
    parser.add_argument("--k-csv", required=True, help="Path to static_exps_03_k.csv")
    parser.add_argument("--search-csv", required=True, help="Path to static_exps_03_search.csv")
    parser.add_argument("--outdir", default="figures/03_static_exps_analysis", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    k_df = ensure_numeric(pd.read_csv(args.k_csv), ["k", "md", "Recall@30", "Avg_Time_Per_Query_ms"])
    search_df = ensure_numeric(
        pd.read_csv(args.search_csv),
        ["k", "nb", "nd", "md", "mqt", "msb", "Recall@30", "Avg_Time_Per_Query_ms"],
    )
    search_df = annotate_index_columns(search_df)

    save_k_plot(k_df, args.outdir)
    save_search_md_effect(search_df, args.outdir)
    save_search_mqt_effect(search_df, args.outdir)
    save_search_msb_effect(search_df, args.outdir)


if __name__ == "__main__":
    main()