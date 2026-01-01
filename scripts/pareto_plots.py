#!/usr/bin/env python3
"""
Generate publication-quality scatter plots (Pareto plots) from the CSV output of
lossless_benchmark.cpp. Compares compression ratio (%) against decompression speed
and random-access speed for all compressors, averaging results across all datasets
found in the CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# 1. STYLING
# ==========================================

# Common shape for all GEF variants
GEF_SHAPE = 'P' # 'P' is a filled plus sign

# Configure matplotlib to use LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{xspace}
        \newcommand{\RLEGEF}{\ensuremath{\textup{RLE-GEF}}\xspace}
        \newcommand{\UGEF}{\ensuremath{\textup{U-GEF}}\xspace}
        \newcommand{\BGEF}{\ensuremath{\textup{B-GEF}}\xspace}
        \newcommand{\BSTARGEF}{\ensuremath{\textup{B}^*\textup{-GEF}}\xspace}
    """
})

# Explicit order for GEF variants (Last in legend)
GEF_ORDER = [
    r"\RLEGEF",
    r"\UGEF (Approximate)",
    r"\UGEF (Optimal)",
    r"\BGEF (Approximate)",
    r"\BGEF (Optimal)",
    r"\BSTARGEF (Approximate)",
    r"\BSTARGEF (Optimal)"
]

STYLES = {
    # COMPETITORS (Standard)
    "Xz":       {"m": "*", "c": "#EA4335", "label": "Xz"},
    "Brotli":   {"m": "h", "c": "#3B7C26", "label": "Brotli"},
    "Zstd":     {"m": "X", "c": "#5BC0BE", "label": "Zstd"},
    "Lz4":      {"m": "o", "c": "#E040FB", "label": "Lz4"},
    "Snappy":   {"m": "o", "c": "#8D8E2C", "label": "Snappy"},
    
    # COMPETITORS (Specialized)
    "DAC":      {"m": "d", "c": "#C62828", "label": "DAC"},
    "ALP":      {"m": "D", "c": "#F8BBD0", "label": "ALP"},
    "Chimp":    {"m": "<", "c": "#8D4E4B", "label": "Chimp"},
    "Chimp128": {"m": "^", "c": "#6A1B9A", "label": "Chimp128"},
    "Gorilla":  {"m": ">", "c": "black",   "label": "Gorilla"},
    "TSXor":    {"m": "v", "c": "#C0CA33", "label": "TSXor"},
    "LeCo":     {"m": "p", "c": "#FBC02D", "label": "LeCo"},
    
    # Additional specialized from C++ benchmark
    "Elf":      {"m": "s", "c": "#795548", "label": "Elf"},
    "Camel":    {"m": "8", "c": "#FF5722", "label": "Camel"},
    "Falcon":   {"m": "H", "c": "#009688", "label": "Falcon"},

    # Ours
    "NeaTS":    {"m": "X", "c": "blue",    "label": "NeaTS"},
    "SNeaTS":   {"m": "X", "c": "grey",    "label": "SNeaTS"},
    "LeaTS":    {"m": "p", "c": "#ff9800", "label": "LeaTS"},
    
    # GEF VARIANTS
    # Using LaTeX commands as requested
    r"\RLEGEF":                {"m": GEF_SHAPE, "c": "#e377c2", "label": r"\RLEGEF"},
    r"\UGEF (Approximate)":    {"m": GEF_SHAPE, "c": "#2ca02c", "label": r"\UGEF (Approx.)"},
    r"\UGEF (Optimal)":        {"m": GEF_SHAPE, "c": "#d62728", "label": r"\UGEF (Optimal)"},
    r"\BGEF (Approximate)":    {"m": GEF_SHAPE, "c": "#1f77b4", "label": r"\BGEF (Approx.)"},
    r"\BGEF (Optimal)":        {"m": GEF_SHAPE, "c": "#ff7f0e", "label": r"\BGEF (Optimal)"},
    r"\BSTARGEF (Approximate)":{"m": GEF_SHAPE, "c": "#9467bd", "label": r"\BSTARGEF (Approx.)"},
    r"\BSTARGEF (Optimal)":    {"m": GEF_SHAPE, "c": "#8c564b", "label": r"\BSTARGEF (Optimal)"},
}

# Map raw CSV compressor names to Style keys
NAME_MAPPING = {
    "neats": "NeaTS",
    "dac": "DAC",
    "rle_gef": r"\RLEGEF",
    "u_gef_approximate": r"\UGEF (Approximate)",
    "u_gef_optimal": r"\UGEF (Optimal)",
    "b_gef_approximate": r"\BGEF (Approximate)",
    "b_gef_optimal": r"\BGEF (Optimal)",
    "b_star_gef_approximate": r"\BSTARGEF (Approximate)",
    "b_star_gef_optimal": r"\BSTARGEF (Optimal)",
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
    "xz": "Xz"
}

# ==========================================
# 2. PLOTTING HELPERS
# ==========================================

def draw_better_arrow(ax):
    """
    Draws a blocky 'Better' arrow pointing Top-Left.
    """
    # Coordinates (0-1 relative to axes)
    x_pos = 0.45
    y_pos = 0.45 
    
    # Rotation -45 ensures the arrow points Top-Left
    rotation = -45 

    ax.text(x_pos, y_pos, "Better", 
            transform=ax.transAxes, 
            rotation=rotation,
            ha="center", va="center",
            size=13, color="dimgrey", alpha=0.9,
            family='serif',
            bbox=dict(boxstyle="larrow,pad=0.3", 
                      fc="silver", 
                      ec="none",   
                      alpha=0.4))

def get_sort_key(algo_name: str) -> Tuple[int, int, str]:
    """
    Returns a sort key for legend ordering:
    1. Non-GEF (0) vs GEF (1)
    2. If GEF, index in GEF_ORDER
    3. Alphabetical fallback
    """
    if algo_name in GEF_ORDER:
        return (1, GEF_ORDER.index(algo_name), algo_name)
    else:
        # Non-GEF: primary group 0, secondary index 0, then by name
        return (0, 0, algo_name)

def create_benchmark_plot(
    data: Dict[str, Tuple[float, float]],
    title: str,
    y_label: str,
    log_scale_y: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort keys based on our custom sort order
    sorted_keys = sorted(data.keys(), key=get_sort_key)

    for algo_name in sorted_keys:
        style = STYLES.get(algo_name)
        if style is None:
            # Fallback style if unknown
            style = {"m": "o", "c": "gray", "label": algo_name}
        
        ratio, value = data[algo_name]
        
        if log_scale_y and value <= 0:
            continue
        
        # Increase marker size slightly for visibility
        size = 90 if style.get('m') == GEF_SHAPE else 80
        
        ax.scatter(
            ratio,
            value,
            marker=style["m"],
            color=style["c"],
            s=size,
            label=style["label"],
            edgecolors="none",
            zorder=10,
        )

    if log_scale_y:
        ax.set_yscale("log")

    # Typography and Grid
    ax.set_xlabel(r"Compression ratio (\%)", fontsize=14, family='serif')
    ax.set_ylabel(y_label, fontsize=14, family='serif')
    
    ax.grid(True, which="major", ls="-", color='#dddddd', linewidth=1.0, zorder=0)
    ax.grid(True, which="minor", ls=":", color='#eeeeee', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Thicken borders for publication quality
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    draw_better_arrow(ax)

    # Legend Configuration
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Sort legend handles by labels using the same sort key logic
        # Note: We need to map labels back to keys if possible, or just trust the order
        # Since we added them in order, they should be in order.
        # However, scatter() auto-adds to legend.
        
        # We can re-sort handles/labels based on our desired order
        # Labels in legend might be slightly different than keys (e.g. "B*-GEF (Approx.)" vs full key)
        # But STYLES[key]["label"] matches what is in `labels`.
        
        # Create a lookup map for label -> sort key
        label_to_key = {}
        for k, v in STYLES.items():
            label_to_key[v["label"]] = k
            
        def legend_sort_key(label):
            # Find the original key for this label
            key = label_to_key.get(label, label)
            return get_sort_key(key)
            
        hl = sorted(zip(handles, labels), key=lambda x: legend_sort_key(x[1]))
        handles2, labels2 = zip(*hl)
        
        ax.legend(
            handles2,
            labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18), # Below the chart
            fancybox=False,
            frameon=True,
            edgecolor='lightgrey',
            shadow=False,
            ncol=5, # Wide layout
            fontsize=10,
            columnspacing=1.0
        )

    plt.title(title, fontsize=16, pad=20, family='serif')
    plt.tight_layout()
    return fig

def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{stem}.pdf"
    fig.savefig(dest, bbox_inches="tight")
    print(f"Saved: {dest}")
    plt.close(fig)

# ==========================================
# 3. MAIN LOGIC
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Plot compression trade-off scatter charts from CSV results.")
    parser.add_argument("csv_file", type=str, help="Path to the output CSV file from lossless_benchmark.cpp")
    parser.add_argument("output_dir", nargs="?", default="plots", help="Directory to save the plots")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Check required columns
    required_cols = ['compressor', 'compression_ratio', 'compression_throughput_mbs', 
                     'decompression_throughput_mbs', 'random_access_mbs']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in CSV: {missing_cols}")
        sys.exit(1)

    # Map compressor names to nice labels
    df['label'] = df['compressor'].map(lambda x: NAME_MAPPING.get(x, x))
    
    # Convert compression ratio to percentage
    # The C++ benchmark outputs ratio as (compressed_size / uncompressed_size).
    # We want percentage (e.g., 50.0 for 50%).
    df['compression_ratio_pct'] = df['compression_ratio'] * 100.0

    # Group by label (compressor) and calculate mean across all datasets
    grouped = df.groupby('label').agg({
        'compression_ratio_pct': 'mean',
        'compression_throughput_mbs': 'mean',
        'decompression_throughput_mbs': 'mean',
        'random_access_mbs': 'mean'
    }).reset_index()

    # Prepare data dictionaries for plotting
    # Format: {"Label": (ratio_pct, metric_value)}
    
    # 1. Compression Throughput
    comp_data = {}
    for _, row in grouped.iterrows():
        comp_data[row['label']] = (row['compression_ratio_pct'], row['compression_throughput_mbs'])

    if comp_data:
        fig = create_benchmark_plot(
            comp_data,
            "Compression Ratio vs Compression Throughput",
            "Compression throughput (MB/s)",
            log_scale_y=True
        )
        save_figure(fig, output_dir, "compression_throughput_vs_ratio")
    else:
        print("No data for compression throughput plot.")

    # 2. Decompression Throughput
    decomp_data = {}
    for _, row in grouped.iterrows():
        decomp_data[row['label']] = (row['compression_ratio_pct'], row['decompression_throughput_mbs'])

    if decomp_data:
        fig = create_benchmark_plot(
            decomp_data,
            "Compression Ratio vs Decompression Throughput",
            "Decompression throughput (MB/s)",
            log_scale_y=True
        )
        save_figure(fig, output_dir, "decompression_throughput_vs_ratio")
    else:
        print("No data for decompression throughput plot.")

    # 3. Random Access Throughput
    ra_data = {}
    for _, row in grouped.iterrows():
        ra_data[row['label']] = (row['compression_ratio_pct'], row['random_access_mbs'])
    
    if ra_data:
        fig = create_benchmark_plot(
            ra_data,
            "Compression Ratio vs Random Access Speed",
            "Random access speed (MB/s)",
            log_scale_y=True
        )
        save_figure(fig, output_dir, "random_access_speed_vs_ratio")
    else:
        print("No data for random access plot.")

if __name__ == "__main__":
    main()
