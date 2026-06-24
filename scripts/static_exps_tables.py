#!/usr/bin/env python3
import argparse
import os
from typing import Iterable

import pandas as pd


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_k_tables(df: pd.DataFrame, outdir: str) -> None:
    by_k_md = (
        df.groupby(["k", "md"], as_index=False)
        .agg(
            recall_mean=("Recall@30", "mean"),
            recall_max=("Recall@30", "max"),
            qtime_mean=("Avg_Time_Per_Query_ms", "mean"),
            qtime_min=("Avg_Time_Per_Query_ms", "min"),
        )
    )
    by_k_md = by_k_md.sort_values(["md", "recall_mean", "qtime_mean"], ascending=[True, False, True])
    by_k_md.to_csv(os.path.join(outdir, "k_summary_by_k_md.csv"), index=False)

    best_per_md = (
        df.sort_values(["md", "Recall@30", "Avg_Time_Per_Query_ms"], ascending=[True, False, True])
        .groupby("md", as_index=False)
        .first()
    )
    best_per_md.to_csv(os.path.join(outdir, "k_best_per_md.csv"), index=False)


def save_nbnd_tables(df: pd.DataFrame, outdir: str) -> None:
    by_nb_nd_md = (
        df.groupby(["nb", "nd", "md"], as_index=False)
        .agg(
            recall_mean=("Recall@30", "mean"),
            recall_max=("Recall@30", "max"),
            qtime_mean=("Avg_Time_Per_Query_ms", "mean"),
            qtime_min=("Avg_Time_Per_Query_ms", "min"),
            indexing_mean=("Indexing_Time_s", "mean"),
            indexing_min=("Indexing_Time_s", "min"),
        )
    )
    by_nb_nd_md = by_nb_nd_md.sort_values(["md", "recall_mean", "qtime_mean"], ascending=[True, False, True])
    by_nb_nd_md.to_csv(os.path.join(outdir, "nb_nd_summary_by_md.csv"), index=False)

    best_per_md = (
        df.sort_values(["md", "Recall@30", "Avg_Time_Per_Query_ms"], ascending=[True, False, True])
        .groupby("md", as_index=False)
        .first()
    )
    best_per_md.to_csv(os.path.join(outdir, "nb_nd_best_per_md.csv"), index=False)

    recall_ge_09 = df[df["Recall@30"] >= 0.9].sort_values(["Avg_Time_Per_Query_ms", "Recall@30"], ascending=[True, False])
    recall_ge_09.to_csv(os.path.join(outdir, "nb_nd_fastest_recall_ge_0_9.csv"), index=False)


def save_search_tables(df: pd.DataFrame, outdir: str) -> None:
    md_baseline = (
        df[(df["mqt"] == 0) & (df["msb"] == 150)]
        .groupby("md", as_index=False)
        .agg(
            recall_mean=("Recall@30", "mean"),
            recall_max=("Recall@30", "max"),
            qtime_mean=("Avg_Time_Per_Query_ms", "mean"),
            qtime_min=("Avg_Time_Per_Query_ms", "min"),
            blocks_mean=("Avg_Blocks_Entered", "mean"),
            blocks_max=("Avg_Blocks_Entered", "max"),
            docs_mean=("Avg_Docs_Examined", "mean"),
            docs_max=("Avg_Docs_Examined", "max"),
        )
    )
    md_baseline = md_baseline.sort_values("md")
    md_baseline.to_csv(os.path.join(outdir, "search_md_baseline_summary.csv"), index=False)

    mqt_summary = (
        df[df["msb"] == 150]
        .groupby(["md", "mqt"], as_index=False)
        .agg(
            recall_mean=("Recall@30", "mean"),
            recall_max=("Recall@30", "max"),
            qtime_mean=("Avg_Time_Per_Query_ms", "mean"),
            qtime_min=("Avg_Time_Per_Query_ms", "min"),
        )
    )
    mqt_summary = mqt_summary.sort_values(["md", "mqt"])
    mqt_summary.to_csv(os.path.join(outdir, "search_mqt_summary.csv"), index=False)

    msb_summary = (
        df[df["mqt"] == 0]
        .groupby(["md", "msb"], as_index=False)
        .agg(
            recall_mean=("Recall@30", "mean"),
            recall_max=("Recall@30", "max"),
            qtime_mean=("Avg_Time_Per_Query_ms", "mean"),
            qtime_min=("Avg_Time_Per_Query_ms", "min"),
        )
    )
    msb_summary = msb_summary.sort_values(["md", "msb"])
    msb_summary.to_csv(os.path.join(outdir, "search_msb_summary.csv"), index=False)

    fastest_ge_09 = df[df["Recall@30"] >= 0.9].sort_values(["Avg_Time_Per_Query_ms", "Recall@30"], ascending=[True, False])
    fastest_ge_09.to_csv(os.path.join(outdir, "search_fastest_recall_ge_0_9.csv"), index=False)

    top_recall = df.sort_values(["Recall@30", "Avg_Time_Per_Query_ms"], ascending=[False, True]).head(50)
    top_recall.to_csv(os.path.join(outdir, "search_top_recall.csv"), index=False)


def save_summary_markdown(outdir: str, args: argparse.Namespace) -> None:
    lines = [
        "# Static Experiments Analysis Summary",
        "",
        "## Input Files",
        "",
        f"- k sweep: {args.k_csv}",
        f"- nb/nd sweep: {args.nbnd_csv}",
        f"- search sweep: {args.search_csv}",
        "",
        "## Generated Tables",
        "",
        "### K sweep",
        "- k_summary_by_k_md.csv",
        "- k_best_per_md.csv",
        "",
        "### NB/ND sweep",
        "- nb_nd_summary_by_md.csv",
        "- nb_nd_best_per_md.csv",
        "- nb_nd_fastest_recall_ge_0_9.csv",
        "",
        "### Search sweep",
        "- search_md_baseline_summary.csv",
        "- search_mqt_summary.csv",
        "- search_msb_summary.csv",
        "- search_fastest_recall_ge_0_9.csv",
        "- search_top_recall.csv",
    ]

    with open(os.path.join(outdir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tables for static experiments (k, indexing, search)")
    parser.add_argument("--k-csv", required=True, help="Path to static_exps_k.csv")
    parser.add_argument("--nbnd-csv", required=True, help="Path to static_exps_nb_nd.csv")
    parser.add_argument("--search-csv", required=True, help="Path to static_exps_search_v1.csv")
    parser.add_argument("--outdir", default="results/static_exps_analysis", help="Output directory for tables")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    k_df = ensure_numeric(pd.read_csv(args.k_csv), ["k", "md", "Recall@30", "Avg_Time_Per_Query_ms"])
    nbnd_df = ensure_numeric(
        pd.read_csv(args.nbnd_csv),
        ["nb", "nd", "md", "Recall@30", "Avg_Time_Per_Query_ms", "Indexing_Time_s"],
    )
    search_df = ensure_numeric(
        pd.read_csv(args.search_csv),
        ["md", "mqt", "msb", "Recall@30", "Avg_Time_Per_Query_ms", "Avg_Blocks_Entered", "Avg_Docs_Examined"],
    )

    save_k_tables(k_df, args.outdir)
    save_nbnd_tables(nbnd_df, args.outdir)
    save_search_tables(search_df, args.outdir)
    save_summary_markdown(args.outdir, args)


if __name__ == "__main__":
    main()
