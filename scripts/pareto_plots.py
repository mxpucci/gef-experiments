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
    # 1. Force Type 42 (TrueType) to avoid "Type 3" errors
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    
    # 2. Disable external LaTeX (Safe Mode)
    "text.usetex": False,
    
    # 3. Font Priority List
    # "Linux Libertine O": The actual VLDB font (if you have the .ttf installed)
    # "Palatino Linotype": The Windows/Office standard for Palatino
    # "Book Antiqua": A very common Palatino clone
    # "Palatino": The Apple/Unix standard
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Palatino Linotype", "Book Antiqua", "Palatino", "Times New Roman"],
    
    # 4. Math Font
    # 'stix' is the best match for Palatino-style math. 
    # 'cm' (Computer Modern) is too thin.
    "mathtext.fontset": "stix",
    
    # 5. Sizes (VLDB uses 9pt body, so 8-10pt figures fit best)
    "font.size": 9,
    "legend.fontsize": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})

# Explicit order for GEF variants (To appear in the 3rd column)
GEF_ORDER = [
    r"RLE-GEF",
    r"U-GEF (Approximate)",
    r"U-GEF (Optimal)",
    r"B-GEF (Approximate)",
    r"B-GEF (Optimal)",
    r"B*-GEF (Approximate)",
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
    r"U-GEF (Approximate)":  {"m": GEF_SHAPE, "c": "#2ca02c", "label": "U-GEF (Approximate)"},
    r"U-GEF (Optimal)":  {"m": GEF_SHAPE, "c": "#d62728", "label": "U-GEF (Optimal)"},
    r"B-GEF (Approximate)":  {"m": GEF_SHAPE, "c": "#1f77b4", "label": "B-GEF (Approximate)"},
    r"B-GEF (Optimal)":  {"m": GEF_SHAPE, "c": "#ff7f0e", "label": "B-GEF (Optimal)"},
    r"B*-GEF (Approximate)": {"m": GEF_SHAPE, "c": "#9467bd", "label": "B*-GEF (Approximate)"},
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
    "u_gef_approximate": r"U-GEF (Approximate)",
    "u_gef_optimal": r"U-GEF (Optimal)",
    "b_gef_approximate": r"B-GEF (Approximate)",
    "b_gef_optimal": r"B-GEF (Optimal)",
    "b_star_gef_approximate": r"B*-GEF (Approximate)",
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

def draw_gef_enclosure(ax, data, log_scale_y=True):
    """
    Draws a circle/ellipse enclosing all GEF variants found in data.
    The shape is calculated to appear as an ellipse even on log-scale plots.
    """
    # Collect GEF points
    gef_points = []
    for name in GEF_ORDER:
        if name in data:
            ratio, value = data[name]
            # Filter out non-positive values if log scale
            if log_scale_y and value <= 0:
                continue
            gef_points.append((ratio, value))
            
    if not gef_points:
        return

    xs = [p[0] for p in gef_points]
    ys = [p[1] for p in gef_points]

    if not xs: return

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Margins (Adjusted to be tighter but still slightly wide)
    margin_x = 0.8  
    margin_y = 0.3  
    
    # X axis (always linear)
    width_x = max_x - min_x
    # Ensure minimum width (at least a small range if points are too close)
    if width_x < 2.0: width_x = 2.0 
    
    center_x = (min_x + max_x) / 2
    rad_x = (width_x / 2) * (1 + margin_x)
    
    # Y axis
    if log_scale_y:
        # Log space calculations for visual symmetry
        # Use simple mean of log values
        log_ys = np.log10(ys)
        min_ly, max_ly = min(log_ys), max(log_ys)
        height_ly = max_ly - min_ly
        if height_ly < 0.2: height_ly = 0.2 # Fallback height in log scale
        
        center_ly = (min_ly + max_ly) / 2
        rad_ly = (height_ly / 2) * (1 + margin_y)
        
        # Generate Ellipse in Log Space
        t = np.linspace(0, 2*np.pi, 200)
        ell_x = center_x + rad_x * np.cos(t)
        ell_ly = center_ly + rad_ly * np.sin(t)
        ell_y = 10**ell_ly
    else:
        # Linear space
        height_y = max_y - min_y
        if height_y == 0: height_y = max_y * 0.1
        
        center_y = (min_y + max_y) / 2
        rad_y = (height_y / 2) * (1 + margin_y)
        
        t = np.linspace(0, 2*np.pi, 200)
        ell_x = center_x + rad_x * np.cos(t)
        ell_y = center_y + rad_y * np.sin(t)

    # Draw
    # Fill with higher alpha (increased visibility)
    ax.fill(ell_x, ell_y, color='#ADD8E6', alpha=0.4, zorder=5)
    # Border with full opacity and thicker line
    ax.plot(ell_x, ell_y, color='#00008B', linestyle='--', linewidth=3, zorder=5)

def export_legend_only(data: Dict[str, Tuple[float, float]], output_dir: Path):
    """
    Generates a clean 3-column legend PDF matching the reference image.
    """
    # Create a figure wide enough to hold the legend
    # Same figsize as other plots (10, 6)
    fig = plt.figure(figsize=(10, 6))
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
                       s=250, # Adjusted marker size in legend
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
        fontsize=18,        # Adjusted font size
        labelspacing=1.5,   # Increased vertical padding
        columnspacing=1.5,  # Adjusted horizontal padding
        handletextpad=0.5,  # Padding between marker and text
        borderpad=0.2       # Slight border padding
    )
    
    # 4. Hide the axes (we only want the legend)
    ax.axis('off')
    
    # 5. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "legend_box.pdf"
    
    # bbox_inches='tight' crops the figure to just the content (the legend)
    # pad_inches=0.05 adds a slight margin around the legend
    fig.savefig(dest, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved Legend: {dest}")
    plt.close(fig)

def plot_benchmark(ax, data, y_label, log_scale_y=True, lower_y_is_better=False):
    # Draw Enclosure first (behind points)
    draw_gef_enclosure(ax, data, log_scale_y)

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
    # rotation=-45 points Top-Left (High Y, Low X)
    # rotation=45 points Bottom-Left (Low Y, Low X)
    rotation = 45 if lower_y_is_better else -45
    
    ax.text(0.45, 0.45, "Better", transform=ax.transAxes, rotation=rotation,
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
    
    decomp_data = {row['label']: (row['compression_ratio_pct'], row['decompression_throughput_mbs']) 
                 for _, row in grouped.iterrows() if 'decompression_throughput_mbs' in row}

    ra_data = {row['label']: (row['compression_ratio_pct'], row['random_access_mbs']) 
                 for _, row in grouped.iterrows() if 'random_access_mbs' in row}
    
    # 1. Export Legend (The priority)
    if comp_data:
        export_legend_only(comp_data, output_dir)

    # 2. Export Standard Plots (Optional context)
    
    # Compression Throughput
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_benchmark(ax, comp_data, "Compression Throughput (MB/s)")
    fig.savefig(output_dir / "compression_throughput_vs_ratio.pdf", bbox_inches='tight')
    print(f"Saved Plot: {output_dir / 'compression_throughput_vs_ratio.pdf'}")
    plt.close(fig)

    # Decompression Throughput
    if decomp_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_benchmark(ax, decomp_data, "Decompression Throughput (MB/s)")
        fig.savefig(output_dir / "decompression_throughput_vs_ratio.pdf", bbox_inches='tight')
        print(f"Saved Plot: {output_dir / 'decompression_throughput_vs_ratio.pdf'}")
        plt.close(fig)

    # Random Access Speed
    if ra_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_benchmark(ax, ra_data, "Random Access Speed (MB/s)")
        fig.savefig(output_dir / "random_access_speed_vs_ratio.pdf", bbox_inches='tight')
        print(f"Saved Plot: {output_dir / 'random_access_speed_vs_ratio.pdf'}")
        plt.close(fig)

if __name__ == "__main__":
    main()