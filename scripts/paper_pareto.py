#!/usr/bin/env python3
"""Pareto plots of the paper: compression ratio vs. compression, decompression and
random-access throughput, plus a standalone legend.

usage: paper_pareto.py [results.csv] [out_dir] [--bitpack benchmark_results/bitpack.csv]

Each point is the arithmetic mean over the datasets of one compressor. Throughputs are
converted from the benchmark's MiB/s to decimal MB/s. The dashed grey line is the mean
compression ratio of plain bit packing (every value in ceil(log2(max - min + 1)) bits),
computed from bitpack.csv. Only the compressors in NAME_MAPPING are plotted.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GEF_SHAPE = "P"
MIB_TO_MB = 1048576 / 1e6

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Palatino Linotype", "Book Antiqua", "Palatino", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 19, "legend.fontsize": 8, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
})

# GEF variants: last column of the legend, in this order.
GEF_ORDER = ["RLE-GEF", "B*-GEF"]

STYLES = {
    "ALP":            {"m": "D", "c": "#F8BBD0"},
    "Brotli":         {"m": "h", "c": "#3B7C26"},
    "BUFF":           {"m": (8, 1, 0), "c": "#00897B"},
    "Camel":          {"m": "8", "c": "#FF5722"},
    "Chimp":          {"m": "<", "c": "#8D4E4B"},
    "Chimp128":       {"m": "^", "c": "#6A1B9A"},
    "DAC":            {"m": "d", "c": "#C62828"},
    "DeXOR":          {"m": "D", "c": "#756bb1"},
    "Elf":            {"m": "s", "c": "#795548"},
    "Elf*":           {"m": "P", "c": "#a63603"},
    "FastPFor":       {"m": "^", "c": "#00ACC1"},
    "Fauna":          {"m": "P", "c": "#006d2c"},
    "FLEA":           {"m": "h", "c": "#dd3497"},
    "Gorilla":        {"m": ">", "c": "black"},
    "LeCo":           {"m": "p", "c": "#FBC02D"},
    "Lz4":            {"m": "o", "c": "#E040FB"},
    "MacaqueV":       {"m": "v", "c": "#31a354"},
    "NeaTS":          {"m": "X", "c": "blue"},
    "Partitioned EF": {"m": "H", "c": "#546E7A"},
    "Pcodec":         {"m": "s", "c": "#3182bd"},
    "SElf*":          {"m": "X", "c": "#e6550d"},
    "Snappy":         {"m": "o", "c": "#8D8E2C"},
    "TSXor":          {"m": "v", "c": "#C0CA33"},
    "Xz":             {"m": "*", "c": "#EA4335"},
    "Zstd":           {"m": "X", "c": "#5BC0BE"},
    "RLE-GEF":        {"m": GEF_SHAPE, "c": "#e377c2"},
    "B*-GEF":         {"m": GEF_SHAPE, "c": "#8c564b"},
}

# CSV compressor name -> plot label. GEF is fed doubles (floating-point forks); B*-GEF uses
# the approximate split point. Pcodec is the int64 variant; FastPFor is the delta variant.
NAME_MAPPING = {
    "rle_gef_f64": "RLE-GEF", "b_star_gef_approximate_f64": "B*-GEF",
    "ALP": "ALP", "BUFF": "BUFF", "Camel": "Camel", "Chimp": "Chimp", "Chimp128": "Chimp128",
    "DAC": "DAC", "Elf": "Elf", "Gorilla": "Gorilla", "LeCo": "LeCo", "NeaTS": "NeaTS",
    "TSXor": "TSXor", "PEF": "Partitioned EF",
    "lz4": "Lz4", "zstd": "Zstd", "brotli": "Brotli", "snappy": "Snappy", "xz": "Xz",
    "pcodec_i64": "Pcodec", "fastpfor_delta": "FastPFor", "flea": "FLEA",
    "elfstar": "Elf*", "selfstar": "SElf*", "dexor": "DeXOR", "macaque": "MacaqueV", "fauna": "Fauna",
}

# (column, y label, file name, axis limits (x, y) of the paper's figures)
PLOTS = [
    ("compression_throughput_mbs", "Compression Throughput (MB/s)", "compression_throughput_vs_ratio",
     (14.42375, 77.4175), (0.4001591875778658, 2546.90148220928)),
    ("decompression_throughput_mbs", "Decompression Throughput (MB/s)", "decompression_throughput_vs_ratio",
     (14.42375, 77.4175), (63.278016903452354, 14991.642329088)),
    ("random_access_mbs", "Random Access Speed (MB/s)", "random_access_speed_vs_ratio",
     (14.42375, 77.4175), (0.05169782636340043, 774.3726805173379)),
]


def sort_key(label):
    return (1, GEF_ORDER.index(label), label) if label in GEF_ORDER else (0, 0, label)


def draw_gef_enclosure(ax, data):
    """Ellipse (round on the log-scale y axis) around the GEF points."""
    pts = [data[n] for n in GEF_ORDER if n in data and data[n][1] > 0]
    if not pts:
        return
    xs, ys = [p[0] for p in pts], np.log10([p[1] for p in pts])
    width = max(max(xs) - min(xs), 2.0)
    height = max(max(ys) - min(ys), 0.2)
    t = np.linspace(0, 2 * np.pi, 200)
    ex = (min(xs) + max(xs)) / 2 + width / 2 * 1.8 * np.cos(t)
    ey = 10 ** ((min(ys) + max(ys)) / 2 + height / 2 * 1.3 * np.sin(t))
    ax.fill(ex, ey, color="#ADD8E6", alpha=0.4, zorder=5)
    ax.plot(ex, ey, color="#00008B", linestyle="--", linewidth=3, zorder=5)


def plot_benchmark(ax, data, y_label):
    draw_gef_enclosure(ax, data)
    for label in sorted(data, key=sort_key):
        ratio, value = data[label]
        if value <= 0:
            continue
        s = STYLES[label]
        ax.scatter(ratio, value, marker=s["m"], color=s["c"], s=80, label=label, edgecolors="none", zorder=10)
    ax.set_yscale("log")
    ax.set_xlabel("Compression ratio (%)")
    ax.set_ylabel(y_label)
    ax.grid(True, which="major", ls="-", color="#dddddd")
    ax.text(0.45, 0.45, "Better", transform=ax.transAxes, rotation=-45, ha="center", va="center", size=18,
            color="dimgrey", alpha=0.9, bbox=dict(boxstyle="larrow,pad=0.5", fc="silver", ec="none", alpha=0.4))


def export_legend(labels, out_dir):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    handles = [ax.scatter([], [], marker=STYLES[l]["m"], color=STYLES[l]["c"], s=250, label=l)
               for l in sorted(labels, key=sort_key)]
    ax.legend(handles, [h.get_label() for h in handles], loc="center", ncol=3, frameon=False, fontsize=18,
              labelspacing=1.5, columnspacing=1.5, handletextpad=0.5, borderpad=0.2)
    ax.axis("off")
    fig.savefig(out_dir / "legend_box.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def bitpack_ratio(path):
    bp = pd.read_csv(path)
    bits = [max(1, math.ceil(math.log2(int(r["max"]) - int(r["min"]) + 1))) for _, r in bp.iterrows()]
    return 100 * sum(b / 64 for b in bits) / len(bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="benchmark_results/results.csv")
    ap.add_argument("out_dir", nargs="?", default="plots")
    ap.add_argument("--bitpack", default="benchmark_results/bitpack.csv")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    df = df[df["compressor"].isin(NAME_MAPPING)].copy()
    df["label"] = df["compressor"].map(NAME_MAPPING)
    df["pct"] = 100 * df["compression_ratio"]
    g = df.groupby("label")[["pct"] + [p[0] for p in PLOTS]].mean().reset_index()
    bitpack = bitpack_ratio(args.bitpack) if Path(args.bitpack).exists() else None

    for col, ylabel, name, xl, yl in PLOTS:
        data = {r["label"]: (r["pct"], r[col] * MIB_TO_MB) for _, r in g.iterrows()}
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_benchmark(ax, data, ylabel)
        xs = [x for x, _ in data.values()]
        ys = [y for _, y in data.values() if y > 0]
        # the paper's limits, widened only if a point would fall outside them
        ax.set_xlim(min(xl[0], min(xs) - 1), max(xl[1], max(xs) + 1))
        ax.set_ylim(min(yl[0], min(ys) / 1.25), max(yl[1], max(ys) * 1.6))
        if bitpack is not None:
            ax.axvline(bitpack, color="#b0b0b0", linestyle="--", linewidth=1.4, zorder=3)
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out / (name + '.pdf')}")
    export_legend(g["label"], out)
    print(f"Saved {out / 'legend_box.pdf'}")


if __name__ == "__main__":
    main()
