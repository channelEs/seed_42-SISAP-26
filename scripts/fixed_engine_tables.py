#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary tables for fixed_engine_experiments results")
    parser.add_argument("--csv", required=True, help="Path to results CSV")
    parser.add_argument("--outdir", default="results/fixed_engine_analysis", help="Output directory for generated tables")
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

    # 1) Fastest configs with recall >= 0.9
    above_09 = df[df["Recall@30"] >= 0.9].copy()
    fastest_above_09 = above_09.sort_values(["Avg_Time_Per_Query_ms", "Recall@30"], ascending=[True, False])
    fastest_above_09.to_csv(os.path.join(args.outdir, "table_fastest_recall_ge_0_9.csv"), index=False)

    # 2) Best recall overall
    best_recall = df.sort_values(["Recall@30", "Avg_Time_Per_Query_ms"], ascending=[False, True]).head(25)
    best_recall.to_csv(os.path.join(args.outdir, "table_best_recall.csv"), index=False)

    # 3) Group-level means by md
    md_summary = (
        df.groupby(["itr", "md"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
            "Avg_Docs_Examined": "mean",
            "Avg_Blocks_Entered": "mean",
            "Avg_Blocks_Skipped": "mean",
        })
        .sort_values(["itr", "md"])
    )
    md_summary.to_csv(os.path.join(args.outdir, "table_by_itr_md.csv"), index=False)

    # 4) Group-level means by nb/nd at each md
    nb_nd_summary = (
        df.groupby(["itr", "nb", "nd", "md"], as_index=False)
        .agg({
            "Recall@30": "mean",
            "Avg_Time_Per_Query_ms": "mean",
            "Avg_Docs_Examined": "mean",
        })
        .sort_values(["itr", "md", "Avg_Time_Per_Query_ms"])
    )
    nb_nd_summary.to_csv(os.path.join(args.outdir, "table_by_itr_nb_nd_md.csv"), index=False)

    # 5) Markdown summary for reporting
    best = fastest_above_09.iloc[0] if not fastest_above_09.empty else None
    lines = []
    lines.append("# fixed_engine_experiments summary")
    lines.append("")
    lines.append(f"Source CSV: {args.csv}")
    lines.append("")
    if best is not None:
        lines.append("## Fastest run with Recall@30 >= 0.9")
        lines.append("")
        lines.append(
            f"- Recall@30: {best['Recall@30']:.6f}"
        )
        lines.append(
            f"- Avg_Time_Per_Query_ms: {best['Avg_Time_Per_Query_ms']:.4f}"
        )
        lines.append(
            f"- Params: k={int(best['k'])}, itr={int(best['itr'])}, nb={int(best['nb'])}, nd={int(best['nd'])}, md={int(best['md'])}, heap_factor={best['heap_factor']:.2f}"
        )
        lines.append("")

    lines.append("## Generated tables")
    lines.append("")
    lines.append("- table_fastest_recall_ge_0_9.csv")
    lines.append("- table_best_recall.csv")
    lines.append("- table_by_itr_md.csv")
    lines.append("- table_by_itr_nb_nd_md.csv")

    with open(os.path.join(args.outdir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
