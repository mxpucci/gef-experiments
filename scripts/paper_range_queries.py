#!/usr/bin/env python3
"""Range-query figure of the paper: throughput vs. range size, left panel for ranges of at
most 200 values, right panel for every size, shared legend.

usage: paper_range_queries.py [results.csv] [out_dir] [--codecs rle_gef_f64,...]

Reads the range_<size>_mbs columns of the benchmark CSV. A size is plotted as the
arithmetic mean over the datasets that have a measurement for every range size (the
largest ones; smaller datasets lack the longest ranges), converted from MiB/s to
decimal MB/s. Writes range_queries.pdf.
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter

MIB_TO_MB = 1048576 / 1e6
SMALL = 200

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False, "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Palatino Linotype", "Book Antiqua", "Palatino", "Times New Roman"],
    "mathtext.fontset": "stix", "font.size": 19, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
})

# CSV name, label, colour, marker (colours match the Pareto plots)
STYLES = [
    ("rle_gef_f64", "RLE-GEF", "#e377c2", "P"),
    ("b_star_gef_approximate_f64", "B*-GEF", "#8c564b", "P"),
    ("ALP", "ALP", "#F8BBD0", "D"),
    ("LeCo", "LeCo", "#FBC02D", "p"),
    ("NeaTS", "NeaTS", "blue", "X"),
    ("BUFF", "BUFF", "#00897B", (8, 1, 0)),
    ("DAC", "DAC", "#C62828", "d"),
    ("lz4", "Lz4", "#E040FB", "o"),
    ("fastpfor_delta", "FastPFor", "#00ACC1", "^"),
    ("PEF", "Partitioned EF", "#546E7A", "H"),
]
DEFAULT = "rle_gef_f64,b_star_gef_approximate_f64,ALP,LeCo,NeaTS,BUFF,DAC,fastpfor_delta"


def load(path, codecs):
    rows = list(csv.DictReader(open(path, newline="")))
    sizes = sorted(int(c[len("range_"):-len("_mbs")]) for c in rows[0] if c.startswith("range_") and c.endswith("_mbs"))
    full = lambda r: all(r.get(f"range_{s}_mbs") not in (None, "") for s in sizes)
    datasets = sorted({r["dataset"] for r in rows if r["compressor"] in codecs and full(r)})
    series = {}
    for c in codecs:
        by_ds = {r["dataset"]: r for r in rows if r["compressor"] == c}
        missing = [d for d in datasets if d not in by_ds or not full(by_ds[d])]
        if missing:
            raise SystemExit(f"{c}: no complete range measurements for {missing}")
        series[c] = [(s, MIB_TO_MB * statistics.mean(float(by_ds[d][f"range_{s}_mbs"]) for d in datasets))
                     for s in sizes]
    return series, sizes, datasets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="benchmark_results/results.csv")
    ap.add_argument("out_dir", nargs="?", default="plots")
    ap.add_argument("--codecs", default=DEFAULT)
    args = ap.parse_args()
    keep = args.codecs.split(",")
    styles = [s for s in STYLES if s[0] in keep]
    series, sizes, datasets = load(args.csv, [s[0] for s in styles])
    print(f"range queries: mean over {len(datasets)} datasets ({', '.join(datasets)})")

    fig, (axs, axl) = plt.subplots(1, 2, figsize=(16, 7.2), gridspec_kw={"width_ratios": [1, 2.2]})
    for key, label, color, marker in styles:
        pts = series[key]
        for ax, sel in ((axs, [p for p in pts if p[0] <= SMALL]), (axl, pts)):
            ax.plot([p[0] for p in sel], [p[1] for p in sel], color=color, marker=marker, markersize=9,
                    linewidth=1.3, label=label)
    for ax in (axs, axl):
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.tick_params(which="major", direction="out", length=7, width=2)
        ax.tick_params(which="minor", direction="out", length=4, width=1.5)
        for s in ax.spines.values():
            s.set_linewidth(2.2)
        ax.grid(True, which="major", color="#dddddd")
        ax.set_axisbelow(True)
        ax.set_xlabel("Range size")
    axl.xaxis.set_major_locator(LogLocator(base=10))
    axl.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
    axl.xaxis.set_minor_formatter(NullFormatter())
    axl.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 4, 5, 10]))
    small = [s for s in sizes if s <= SMALL]
    axs.xaxis.set_major_locator(FixedLocator(small))
    axs.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axs.xaxis.set_minor_locator(FixedLocator([]))
    axs.set_xlim(small[0] / 1.25, small[-1] * 1.25)
    axs.set_ylabel("Throughput (MB/s)")
    fig.tight_layout(rect=[0, 0.065, 1, 1], w_pad=2.5)
    right = axl.get_position().x1
    leg = fig.legend(*axl.get_legend_handles_labels(), loc="upper center", ncol=len(styles), frameon=True,
                     fancybox=True, mode="expand", bbox_to_anchor=(0.0, 0.0, right, 0.065), borderaxespad=0,
                     handlelength=2.6 if len(styles) <= 5 else 1.8, handletextpad=0.6)
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_linewidth(3)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "range_queries.pdf", bbox_inches="tight")
    print(f"Saved {out / 'range_queries.pdf'}")


if __name__ == "__main__":
    main()
