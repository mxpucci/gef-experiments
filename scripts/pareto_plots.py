#!/usr/bin/env python3
"""
Generate publication-quality scatter plots and a standalone legend 
matching the specific 3-column layout provided in the reference image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# 1. STYLING & CONFIGURATION
# ==========================================

# Shape for GEF variants (Filled Plus)
GEF_SHAPE = 'P' 

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12
})

# Explicit order for GEF variants (To appear in the 3rd column)
GEF_ORDER = [
    r"RLE-GEF",
    r"U-GEF (Approx.)",
    r"U-GEF (Optimal)",
    r"B-GEF (Approx.)",
    r"B-GEF (Optimal)",
    r"B*-GEF (Approx.)",
    r"B*-GEF (Optimal)"
]

STYLES = {
    # --- COLUMN 1 & 2 Candidates (Standard) ---
    "ALP":      {"m": "D", "c": "#F8BBD0", "label": "ALP"},
    "Brotli":   {"m": "h", "c": "#3B7C26", "label": "Brotli"},
    "Camel":    {"m": "8", "c": "#FF5722", "label": "Camel"},
    "Chimp":    {"m": "<", "c": "#8D4E4B", "label": "Chimp"},
    "Chimp128": {"m": "^", "c": "#6A1B9A", "label": "Chimp128"},
    "DAC":      {"m": "d", "c": "#C62828", "label": "DAC"},
    "Elf":      {"m": "s", "c": "#795548", "label": "Elf"},
    "Falcon":   {"m": "H", "c": "#009688", "label": "Falcon"},
    
    "Gorilla":  {"m": ">", "c": "black",   "label": "Gorilla"},
    "LeCo":     {"m": "p", "c": "#FBC02D", "label": "LeCo"},
    "Lz4":      {"m": "o", "c": "#E040FB", "label": "Lz4"},
    "NeaTS":    {"m": "X", "c": "blue",    "label": "NeaTS"},
    "Snappy":   {"m": "o", "c": "#8D8E2C", "label": "Snappy"},
    "TSXor":    {"m": "v", "c": "#C0CA33", "label": "TSXor"},
    "Xz":       {"m": "*", "c": "#EA4335", "label": "Xz"},
    "Zstd":     {"m": "X", "c": "#5BC0BE", "label": "Zstd"},

    # --- COLUMN 3 Candidates (GEF) ---
    r"RLE-GEF":          {"m": GEF_SHAPE, "c": "#e377c2", "label": "RLE-GEF"},
    r"U-GEF (Approx.)":  {"m": GEF_SHAPE, "c": "#2ca02c", "label": "U-GEF (Approx.)"},
    r"U-GEF (Optimal)":  {"m": GEF_SHAPE, "c": "#d62728", "label": "U-GEF (Optimal)"},
    r"B-GEF (Approx.)":  {"m": GEF_SHAPE, "c": "#1f77b4", "label": "B-GEF (Approx.)"},
    r"B-GEF (Optimal)":  {"m": GEF_SHAPE, "c": "#ff7f0e", "label": "B-GEF (Optimal)"},
    r"B*-GEF (Approx.)": {"m": GEF_SHAPE, "c": "#9467bd", "label": "B*-GEF (Approx.)"},
    r"B*-GEF (Optimal)": {"m": GEF_SHAPE, "c": "#8c564b", "label": "B*-GEF (Optimal)"},
    
    # Extra / Unused in image but kept for safety
    "SNeaTS":   {"m": "X", "c": "grey",    "label": "SNeaTS"},
    "LeaTS":    {"m": "p", "c": "#ff9800", "label": "LeaTS"},
}

# Mapping raw CSV names to our Style Keys
NAME_MAPPING = {
    "neats": "NeaTS",
    "dac": "DAC",
    "rle_gef": r"RLE-GEF",
    "u_gef_approximate": r"U-GEF (Approx.)",
    "u_gef_optimal": r"U-GEF (Optimal)",
    "b_gef_approximate": r"B-GEF (Approx.)",
    "b_gef_optimal": r"B-GEF (Optimal)",
    "b_star_gef_approximate": r"B*-GEF (Approx.)",
    "b_star_gef_optimal": r"B*-GEF (Optimal)",
    "gorilla": "Gorilla",
    "chimp": "Chimp",
    "chimp128": "Chimp128",
    "tsxor": "TSXor",
    "elf": "Elf",
    "camel": "Camel",
    "falcon": "Falcon",
    "lz4": "Lz4",
    "zstd": "Zstd",
    "brotli": "Brotli",
    "snappy": "Snappy",
    "xz": "Xz",
    "leco": "LeCo",
    "alp": "ALP"
}

# ==========================================
# 2. PLOTTING HELPERS
# ==========================================

def get_sort_key(algo_name: str) -> Tuple[int, int, str]:
    """
    Sorts items so Standard algos come first (Columns 1 & 2),
    and GEF algos come last (Column 3).
    """
    if algo_name in GEF_ORDER:
        # Priority 1: GEF items (At the end)
        return (1, GEF_ORDER.index(algo_name), algo_name)
    else:
        # Priority 0: Standard items (Alphabetical)
        return (0, 0, algo_name)

def export_legend_only(data: Dict[str, Tuple[float, float]], output_dir: Path):
    """
    Generates a clean 3-column legend PDF matching the reference image.
    """
    # Create a figure wide enough to hold the legend
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    # 1. Gather Styles
    # Sort keys: Alphabetical first, then GEF specific order
    sorted_keys = sorted(data.keys(), key=get_sort_key)
    
    handles = []
    labels = []
    
    # 2. Generate Invisible Plot Points for Handles
    for algo_name in sorted_keys:
        style = STYLES.get(algo_name)
        if not style:
            # Fallback for unknown CSV entries
            style = {"m": "o", "c": "gray", "label": algo_name}
            
        h = ax.scatter([], [], 
                       marker=style["m"], 
                       color=style["c"], 
                       s=100, # Marker size in legend
                       label=style["label"])
        handles.append(h)
        labels.append(style["label"])
        
    # 3. Create the Legend
    # ncol=3 is the magic number. With ~23 items, matplotlib splits them:
    # Col 1: 8 items, Col 2: 8 items, Col 3: 7 items (GEF)
    legend = ax.legend(
        handles, labels,
        loc="center",
        ncol=3,            
        frameon=False,      # No box around the whole thing
        fontsize=13,        # Readable size
        labelspacing=1.2,   # Vertical padding
        columnspacing=2.5,  # Horizontal padding between columns
        handletextpad=0.5,  # Padding between marker and text
        borderpad=1.0
    )
    
    # 4. Hide the axes (we only want the legend)
    ax.axis('off')
    
    # 5. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "legend.pdf"
    
    # bbox_inches='tight' crops the figure to just the content (the legend)
    fig.savefig(dest, bbox_inches='tight')
    print(f"Saved Legend: {dest}")
    plt.close(fig)

def plot_benchmark(ax, data, y_label, log_scale_y=True):
    sorted_keys = sorted(data.keys(), key=get_sort_key)
    for algo_name in sorted_keys:
        style = STYLES.get(algo_name)
        if not style: continue
        
        ratio, value = data[algo_name]
        if log_scale_y and value <= 0: continue
        
        ax.scatter(ratio, value, marker=style["m"], color=style["c"], 
                   s=80, label=style["label"], edgecolors="none", zorder=10)

    if log_scale_y: ax.set_yscale("log")
    ax.set_xlabel("Compression ratio (%)")
    ax.set_ylabel(y_label)
    ax.grid(True, which="major", ls="-", color='#dddddd')
    
    # "Better" Arrow
    ax.text(0.45, 0.45, "Better", transform=ax.transAxes, rotation=-45,
            ha="center", va="center", size=18, color="dimgrey", alpha=0.9,
            bbox=dict(boxstyle="larrow,pad=0.5", fc="silver", ec="none", alpha=0.4))

# ==========================================
# 3. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str)
    parser.add_argument("output_dir", nargs="?", default="plots")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    # Read and Process CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['label'] = df['compressor'].map(lambda x: NAME_MAPPING.get(x, x))
    df['compression_ratio_pct'] = df['compression_ratio'] * 100.0

    grouped = df.groupby('label').mean(numeric_only=True).reset_index()

    # Prepare Data
    comp_data = {row['label']: (row['compression_ratio_pct'], row['compression_throughput_mbs']) 
                 for _, row in grouped.iterrows()}
    
    # 1. Export Legend (The priority)
    if comp_data:
        export_legend_only(comp_data, output_dir)

    # 2. Export Standard Plots (Optional context)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_benchmark(ax, comp_data, "Compression Throughput (MB/s)")
    fig.savefig(output_dir / "compression_plot.pdf", bbox_inches='tight')
    print(f"Saved Plot: {output_dir / 'compression_plot.pdf'}")

if __name__ == "__main__":
    main()