#!/usr/bin/env python3
"""Aggregate and plot range-query trials (methodology: docs/range-queries.md).

Input: directories produced by run_range_queries.py (reads raw/*.trials.csv)
and/or trials CSVs written by LosslessBenchmark --range-trials-csv.

Output (in --out, default: the first input directory):
  range-trials.csv       every trial, decimal MB/s
  range-per-dataset.csv  median over trials per (codec, dataset, size)
  range-summary.csv      arithmetic mean over datasets per (codec, size)
  range-throughput.png/.svg  mean throughput vs range size, log-log, decimal MB/s

Rows whose range queries did not all match full decompression are dropped
(and reported). A codec is plotted only if it covers every dataset at a size,
so that every point of the plot averages the same datasets.
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PRETTY = {
    "rle_gef": "RLE-GEF", "u_gef_approximate": "U-GEF (approx.)", "u_gef_optimal": "U-GEF",
    "b_gef_approximate": "B-GEF (approx.)", "b_gef_optimal": "B-GEF",
    "b_star_gef_approximate": "B*-GEF (approx.)", "b_star_gef_optimal": "B*-GEF",
    "neats": "NeaTS", "dac": "DAC", "alp": "ALP", "leco": "LeCo", "buff": "BUFF", "camel": "Camel",
    "gorilla": "Gorilla", "chimp": "Chimp", "chimp128": "Chimp128", "tsxor": "TSXor", "elf": "Elf",
    "lz4": "LZ4", "zstd": "Zstd", "brotli": "Brotli", "xz": "Xz", "snappy": "Snappy",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="run directories and/or *.trials.csv files")
    p.add_argument("--out", help="output directory (default: first input directory)")
    p.add_argument("--codecs", help="comma-separated codec keys to include, in legend order")
    p.add_argument("--title", default="Random range queries")
    p.add_argument("--note", default="", help="extra footnote line (e.g. CPU, commit)")
    return p.parse_args()


def load(inputs):
    """Return trial rows with a 'codec' key: the case name from the run
    directory (raw/<dataset>__<codec>.trials.csv), else the compressor field."""
    rows = []
    for item in map(Path, inputs):
        files = sorted((item / "raw").glob("*.trials.csv")) if item.is_dir() else [item]
        if item.is_dir() and not files:
            files = sorted(item.glob("*.trials.csv"))
        for f in files:
            key = f.name[: -len(".trials.csv")].split("__", 1)[1] if "__" in f.name else None
            for r in csv.DictReader(open(f)):
                r["codec"] = key or r["compressor"].lower()
                rows.append(r)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    rows = load(args.inputs)
    if not rows:
        sys.exit("no range trials found")
    out = Path(args.out) if args.out else next((Path(i) for i in args.inputs if Path(i).is_dir()), Path("."))
    out.mkdir(parents=True, exist_ok=True)
    order = args.codecs.split(",") if args.codecs else sorted({r["codec"] for r in rows})
    rows = [r for r in rows if r["codec"] in order]

    trials, dropped = [], set()
    for r in rows:
        if int(r["mismatched_queries"]) or int(r["verified_queries"]) < int(r["queries"]):
            dropped.add((r["codec"], r["dataset"], r["range_size"], r["mismatched_queries"], r["verified_queries"]))
            if int(r["mismatched_queries"]):
                continue
        trials.append(dict(codec=r["codec"], compressor=r["compressor"], dataset=r["dataset"],
                           range_size=int(r["range_size"]), trial=int(r["trial"]) + 1, queries=int(r["queries"]),
                           elapsed_ns=int(r["elapsed_ns"]), throughput_MB_per_s=float(r["throughput_MB_per_s"]),
                           queries_per_s=float(r["queries_per_s"])))
    for codec, dataset, size, bad, checked in sorted(dropped):
        what = f"{bad} mismatching queries, dropped" if int(bad) else f"only {checked} queries verified, kept"
        print(f"warning: {codec} {dataset} size {size}: {what}", file=sys.stderr)
    if not trials:
        sys.exit("no valid range trials")

    groups = defaultdict(list)
    for t in trials:
        groups[(t["codec"], t["dataset"], t["range_size"])].append(t)
    per_dataset = []
    for (codec, dataset, size), ts in sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1], kv[0][2])):
        per_dataset.append(dict(codec=codec, compressor=ts[0]["compressor"], dataset=dataset, range_size=size,
                                trials=len(ts), queries=ts[0]["queries"],
                                median_MB_per_s=statistics.median(t["throughput_MB_per_s"] for t in ts),
                                median_queries_per_s=statistics.median(t["queries_per_s"] for t in ts)))

    datasets = sorted({p["dataset"] for p in per_dataset})
    summary = []
    for codec in order:
        for size in sorted({p["range_size"] for p in per_dataset if p["codec"] == codec}):
            vals = [p for p in per_dataset if p["codec"] == codec and p["range_size"] == size]
            summary.append(dict(codec=codec, range_size=size, datasets=len(vals),
                                complete=len(vals) == len(datasets),
                                mean_MB_per_s=statistics.mean(v["median_MB_per_s"] for v in vals),
                                mean_queries_per_s=statistics.mean(v["median_queries_per_s"] for v in vals)))

    write_csv(out / "range-trials.csv", trials)
    write_csv(out / "range-per-dataset.csv", per_dataset)
    write_csv(out / "range-summary.csv", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = "oDs^vP*Xh<>"
    plotted = 0
    for i, codec in enumerate(order):
        pts = [s for s in summary if s["codec"] == codec and s["complete"]]
        partial = [s for s in summary if s["codec"] == codec and not s["complete"]]
        if partial:
            print(f"warning: {codec}: {len(partial)} size(s) not covered by all {len(datasets)} datasets; "
                  "not plotted", file=sys.stderr)
        if not pts:
            continue
        ax.plot([s["range_size"] for s in pts], [s["mean_MB_per_s"] for s in pts], label=PRETTY.get(codec, codec),
                marker=markers[i % len(markers)], markersize=4, linewidth=1.8)
        plotted += 1
    ax.set_xscale("log")  # no base= kwarg: keeps old matplotlib (3.0, Python 3.7) working
    ax.set_yscale("log")
    sizes = sorted({s["range_size"] for s in summary})
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{x:,}" for x in sizes], rotation=45, ha="right", fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("Range size (values)")
    ax.set_ylabel("Throughput (MB/s, 10$^6$ bytes/s)")
    ax.grid(True, which="major", alpha=0.25)
    if plotted:
        ax.legend(ncol=2, frameon=False)
    ax.set_title(f"{args.title} · {len(datasets)} dataset{'s' if len(datasets) != 1 else ''}", loc="left", fontsize=12)
    q = sorted({t["queries"] for t in trials})
    nt = sorted({len(ts) for ts in groups.values()})
    foot = (f"{'/'.join(map(str, q))} shared random ranges per size · median of {'/'.join(map(str, nt))} trial(s), "
            "then arithmetic mean across datasets · 8-byte output values")
    if args.note:
        foot += "\n" + args.note
    fig.text(0.5, 0.012, foot, ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.06 if args.note else 0.04, 1, 1])
    fig.savefig(out / "range-throughput.png", dpi=200)
    fig.savefig(out / "range-throughput.svg")
    print(f"wrote {out}/range-{{trials,per-dataset,summary}}.csv and range-throughput.{{png,svg}}")


if __name__ == "__main__":
    main()
