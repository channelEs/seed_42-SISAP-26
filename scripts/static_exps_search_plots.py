#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


INDEX_COMBINATIONS: Dict[Tuple[int, int, int], str] = {
    (500, 500, 400): "I-1",
    (500, 750, 75): "I-2",
    (1000, 500, 400): "I-3",
    (1000, 100, 750): "I-4",
    (1500, 500, 400): "I-5",
    (1500, 100, 750): "I-6",
    # (500, 100, 750): "I-3",
    # (1000, 750, 75): "I-3",
    # (1500, 750, 75): "I-8",
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


def configured_index_ids() -> list[str]:
    # Only these index IDs are considered valid for plotting.
    ids = list(dict.fromkeys(INDEX_COMBINATIONS.values()))

    def sort_key(value: str) -> tuple[int, str]:
        if value.startswith("I-"):
            suffix = value[2:]
            if suffix.isdigit():
                return (0, f"{int(suffix):04d}")
        return (1, value)

    return sorted(ids, key=sort_key)


def color_for_index(index_id: str, idx: int):
    if index_id in INDEX_COLORS:
        return INDEX_COLORS[index_id]
    cmap = plt.get_cmap("tab20")
    return cmap(idx % 20)


def add_index_legend(fig: plt.Figure, axes: list[plt.Axes], title: str = "Static index") -> None:
    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)

    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

    if unique_handles:
        fig.legend(
            unique_handles,
            unique_labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            fontsize=16,
            title=title,
        )


def create_search_figure() -> tuple[plt.Figure, list[plt.Axes], plt.Axes]:
    fig = plt.figure(figsize=(15, 5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.7], wspace=0.18)
    plot_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.axis("off")
    return fig, plot_axes, legend_ax

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

    fig, axes, legend_ax = create_search_figure()

    for idx, index_id in enumerate(configured_index_ids()):
        sub = md_df[md_df["index_id"] == index_id]
        if sub.empty:
            continue
        color = color_for_index(index_id, idx)
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

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend_ax.legend(handles, labels, loc="center left", fontsize=16, title="Static index", frameon=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.18)

    plt.savefig(os.path.join(outdir, "search_md_effect.png"), dpi=180, bbox_inches="tight")
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

    fig, axes, legend_ax = create_search_figure()

    for idx, index_id in enumerate(configured_index_ids()):
        sub = mqt_df[mqt_df["index_id"] == index_id]
        if sub.empty:
            continue
        color = color_for_index(index_id, idx)
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

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend_ax.legend(handles, labels, loc="center left", fontsize=16, title="Static index", frameon=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.18)

    plt.savefig(os.path.join(outdir, "search_mqt_effect.png"), dpi=180, bbox_inches="tight")
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

    fig, axes, legend_ax = create_search_figure()

    for idx, index_id in enumerate(configured_index_ids()):
        sub = msb_df[msb_df["index_id"] == index_id]
        if sub.empty:
            continue
        color = color_for_index(index_id, idx)
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

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend_ax.legend(handles, labels, loc="center left", fontsize=16, title="Static index", frameon=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.18)

    plt.savefig(os.path.join(outdir, "search_msb_effect.png"), dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate round-3 static experiment plots")
    parser.add_argument("--plot", choices=["all", "md", "mqt", "msb"], default="all", help="Select which plot(s) to generate")
    parser.add_argument("--search-md-csv", required=False, help="Path to static_exps_search_md.csv")
    parser.add_argument("--search-mqt-csv", required=False, help="Path to static_exps_search_mqt.csv")
    parser.add_argument("--search-msb-csv", required=False, help="Path to static_exps_search_msb.csv")
    parser.add_argument("--outdir", default="figures/static_exps_analysis", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if (
        args.search_md_csv is None
        and args.search_mqt_csv is None
        and args.search_msb_csv is None
    ):
        parser.error("Provide at least one CSV input: --search-md-csv, --search-mqt-csv, or --search-msb-csv")

    def load_search_df(path: str) -> pd.DataFrame:
        df = ensure_numeric(
            pd.read_csv(path),
            ["k", "nb", "nd", "md", "mqt", "msb", "Recall@30", "Avg_Time_Per_Query_ms"],
        )
        return annotate_index_columns(df)

    # if args.plot in ("all", "k") and args.k_csv:
    #     k_df = ensure_numeric(pd.read_csv(args.k_csv), ["k", "md", "Recall@30", "Avg_Time_Per_Query_ms"])
    #     save_k_plot(k_df, args.outdir)

    # if args.plot in ("all", "nbnd") and args.nbnd_csv:
    #     nbnd_df = ensure_numeric(pd.read_csv(args.nbnd_csv), ["nb", "nd", "md", "Recall@30", "Avg_Time_Per_Query_ms"])
    #     save_nbnd_plots(nbnd_df, args.outdir)

    if args.plot in ("all", "md"):
        md_source = args.search_md_csv
        if md_source:
            save_search_md_effect(load_search_df(md_source), args.outdir)

    if args.plot in ("all", "mqt"):
        mqt_source = args.search_mqt_csv
        if mqt_source:
            save_search_mqt_effect(load_search_df(mqt_source), args.outdir)

    if args.plot in ("all", "msb"):
        msb_source = args.search_msb_csv
        if msb_source:
            save_search_msb_effect(load_search_df(msb_source), args.outdir)


if __name__ == "__main__":
    main()