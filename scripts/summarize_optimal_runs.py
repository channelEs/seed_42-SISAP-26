#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def parse_run_identity(filename_stem):
    # Expected: <dataset>__<config>
    if "__" not in filename_stem:
        return "unknown_dataset", filename_stem
    dataset, config = filename_stem.split("__", 1)
    return dataset, config


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rank_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            -safe_float(r.get("Recall@30", "nan")),
            safe_float(r.get("Avg_Time_Per_Query_ms", "nan")),
            safe_float(r.get("Total_Time_s", "nan")),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Summarize optimal-config benchmark runs.")
    parser.add_argument("--runs-dir", required=True, help="Path to results/optimal_runs")
    parser.add_argument("--out-dir", required=True, help="Path to output summary directory")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    csv_dir = runs_dir / "csv"
    out_dir = Path(args.out_dir)

    run_files = sorted(csv_dir.glob("*.csv"))
    if not run_files:
        raise SystemExit(f"No run CSVs found in {csv_dir}")

    combined = []
    for run_file in run_files:
        dataset, config = parse_run_identity(run_file.stem)
        with run_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_out = dict(row)
                row_out["dataset"] = dataset
                row_out["config_name"] = config
                combined.append(row_out)

    if not combined:
        raise SystemExit("Run CSV files were found, but no data rows were parsed.")

    # Combined raw view
    combined_fields = [
        "dataset",
        "config_name",
        "k",
        "itr",
        "nb",
        "nd",
        "heap_factor",
        "mqt",
        "msb",
        "md",
        "Recall@30",
        "Avg_Time_Per_Query_ms",
        "Search_Time_s",
        "Total_Time_s",
        "Clustering_Time_s",
        "Indexing_Time_s",
        "Avg_Blocks_Entered",
        "Avg_Docs_Examined",
    ]
    combined_rows = []
    for row in combined:
        combined_rows.append({k: row.get(k, "") for k in combined_fields})
    write_csv(out_dir / "combined_runs.csv", combined_rows, combined_fields)

    # Per-dataset ranking
    by_dataset = defaultdict(list)
    for row in combined_rows:
        by_dataset[row["dataset"]].append(row)

    ranking_dataset_rows = []
    for dataset, rows in sorted(by_dataset.items()):
        ranked = rank_rows(rows)
        for idx, row in enumerate(ranked, start=1):
            enriched = dict(row)
            enriched["rank"] = idx
            ranking_dataset_rows.append(enriched)

    ranking_dataset_fields = ["dataset", "rank"] + [f for f in combined_fields if f != "dataset"]
    write_csv(out_dir / "ranking_per_dataset.csv", ranking_dataset_rows, ranking_dataset_fields)

    # Overall ranking by config (mean recall desc, mean query time asc)
    by_config = defaultdict(list)
    for row in combined_rows:
        by_config[row["config_name"]].append(row)

    overall_rows = []
    for config_name, rows in sorted(by_config.items()):
        recalls = [safe_float(r["Recall@30"]) for r in rows]
        qtimes = [safe_float(r["Avg_Time_Per_Query_ms"]) for r in rows]
        totals = [safe_float(r["Total_Time_s"]) for r in rows]
        overall_rows.append(
            {
                "config_name": config_name,
                "num_datasets": len(rows),
                "mean_recall@30": sum(recalls) / len(recalls),
                "mean_query_time_ms": sum(qtimes) / len(qtimes),
                "mean_total_time_s": sum(totals) / len(totals),
                "min_recall@30": min(recalls),
                "max_recall@30": max(recalls),
            }
        )

    overall_rows.sort(key=lambda r: (-r["mean_recall@30"], r["mean_query_time_ms"], r["mean_total_time_s"]))
    for i, row in enumerate(overall_rows, start=1):
        row["rank"] = i

    overall_fields = [
        "rank",
        "config_name",
        "num_datasets",
        "mean_recall@30",
        "mean_query_time_ms",
        "mean_total_time_s",
        "min_recall@30",
        "max_recall@30",
    ]
    write_csv(out_dir / "ranking_overall.csv", overall_rows, overall_fields)

    # Human-friendly markdown summary
    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Optimal Config Benchmark Summary\n\n")
        f.write(f"Runs parsed: {len(combined_rows)}\n\n")

        f.write("## Ranking Per Dataset\n\n")
        for dataset, rows in sorted(by_dataset.items()):
            ranked = rank_rows(rows)
            f.write(f"### {dataset}\n\n")
            f.write("| Rank | Config | Recall@30 | Query ms | Total s | md | mqt | msb |\n")
            f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
            for idx, row in enumerate(ranked, start=1):
                f.write(
                    f"| {idx} | {row['config_name']} | {safe_float(row['Recall@30']):.6f} | "
                    f"{safe_float(row['Avg_Time_Per_Query_ms']):.3f} | {safe_float(row['Total_Time_s']):.3f} | "
                    f"{row.get('md', '')} | {row.get('mqt', '')} | {row.get('msb', '')} |\n"
                )
            f.write("\n")

        f.write("## Overall Aggregated Ranking (Reference)\n\n")
        f.write("| Rank | Config | Datasets | Mean Recall@30 | Mean Query ms | Mean Total s |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        for row in overall_rows:
            f.write(
                f"| {row['rank']} | {row['config_name']} | {row['num_datasets']} | "
                f"{row['mean_recall@30']:.6f} | {row['mean_query_time_ms']:.3f} | {row['mean_total_time_s']:.3f} |\n"
            )

    print(f"[OK] Wrote: {out_dir / 'combined_runs.csv'}")
    print(f"[OK] Wrote: {out_dir / 'ranking_per_dataset.csv'}")
    print(f"[OK] Wrote: {out_dir / 'ranking_overall.csv'}")
    print(f"[OK] Wrote: {md_path}")


if __name__ == "__main__":
    main()
