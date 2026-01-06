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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd
import numpy as np

# ==========================================
# 1. STYLING
# ==========================================

# Common shape for all GEF variants
GEF_SHAPE = 'P' # 'P' is a filled plus sign

# Configure matplotlib to use LaTeX
plt.rcParams.update({
    "text.usetex": False,
    # "font.family": "serif",
    # "text.latex.preamble": r"""
    #     \usepackage{amsmath}
    #     \usepackage{xspace}
    #     \newcommand{\RLEGEF}{\ensuremath{\textup{RLE-GEF}}\xspace}
    #     \newcommand{\UGEF}{\ensuremath{\textup{U-GEF}}\xspace}
    #     \newcommand{\BGEF}{\ensuremath{\textup{B-GEF}}\xspace}
    #     \newcommand{\BSTARGEF}{\ensuremath{\textup{B}^*\textup{-GEF}}\xspace}
    # """
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
            size=18, color="dimgrey", alpha=0.9,
            family='serif',
            bbox=dict(boxstyle="larrow,pad=0.5", 
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

def get_sorted_handles_labels(handles, labels):
    """
    Sorts legend handles and labels based on the custom sort key.
    """
    # Create a lookup map for label -> sort key
    label_to_key = {}
    for k, v in STYLES.items():
        label_to_key[v["label"]] = k
        
    def legend_sort_key(label):
        # Find the original key for this label
        key = label_to_key.get(label, label)
        return get_sort_key(key)
        
    hl = sorted(zip(handles, labels), key=lambda x: legend_sort_key(x[1]))
    return zip(*hl)

def plot_on_axis(ax, data: Dict[str, Tuple[float, float]], y_label: str, title: str, log_scale_y: bool = True):
    """
    Plots the benchmark data on a single axis.
    """
    # Sort keys based on our custom sort order
    sorted_keys = sorted(data.keys(), key=get_sort_key)

    gef_xs = []
    gef_ys = []

    for algo_name in sorted_keys:
        style = STYLES.get(algo_name)
        if style is None:
            # Fallback style if unknown
            style = {"m": "o", "c": "gray", "label": algo_name}
        
        ratio, value = data[algo_name]

        if algo_name in GEF_ORDER:
            gef_xs.append(ratio)
            gef_ys.append(value)
        
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

    # Draw grouping ellipse for GEF
    if gef_xs:
        from matplotlib.patches import Ellipse
        min_x, max_x = min(gef_xs), max(gef_xs)
        width_x = max_x - min_x
        if width_x == 0: width_x = 2.0
        center_x = (min_x + max_x) / 2
        width = width_x * 1.5

        min_y, max_y = min(gef_ys), max(gef_ys)
        if log_scale_y and min_y > 0:
            log_min, log_max = np.log10(min_y), np.log10(max_y)
            log_h = log_max - log_min
            if log_h == 0: log_h = 0.1
            log_c = (log_min + log_max) / 2
            log_half_h = (log_h * 1.5) / 2
            top = 10**(log_c + log_half_h)
            bottom = 10**(log_c - log_half_h)
            center_y = (top + bottom) / 2
            height = top - bottom
        else:
            height_y = max_y - min_y
            if height_y == 0: height_y = min_y * 0.1
            center_y = (min_y + max_y) / 2
            height = height_y * 1.5

        ell = Ellipse((center_x, center_y), width, height,
                      facecolor='#E3F2FD', alpha=0.5,
                      edgecolor='#2196F3', linestyle='--', linewidth=2.0, zorder=0)
        ax.add_patch(ell)

    if log_scale_y:
        ax.set_yscale("log")

    # Typography and Grid
    ax.set_xlabel(r"Compression ratio (%)", fontsize=14, family='serif')
    ax.set_ylabel(y_label, fontsize=14, family='serif')
    # ax.set_title(title, fontsize=16, pad=20, family='serif')
    
    ax.grid(True, which="major", ls="-", color='#dddddd', linewidth=1.0, zorder=0)
    ax.grid(True, which="minor", ls=":", color='#eeeeee', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Thicken borders for publication quality
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    draw_better_arrow(ax)

def create_benchmark_plot(
    data: Dict[str, Tuple[float, float]],
    title: str,
    y_label: str,
    log_scale_y: bool = False,
    show_legend: bool = False  # <--- NEW ARGUMENT
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_on_axis(ax, data, y_label, title, log_scale_y)

    # Only add legend if requested
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            handles2, labels2 = get_sorted_handles_labels(handles, labels)
            ax.legend(
                handles2, labels2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                fancybox=False, frameon=True, edgecolor='lightgrey',
                ncol=5, fontsize=10, columnspacing=1.0
            )

    plt.tight_layout()
    return fig

def export_legend_only(data: Dict[str, Tuple[float, float]], output_dir: Path):
    """
    Generates a rectangular standalone PDF legend to fit in a subplot slot.
    """
    # 1. Use a large figure size to ensure the legend fits during calculation
    fig = plt.figure(figsize=(10, 10)) 
    ax = fig.add_subplot(111)
    
    # Plot invisible dummy points
    sorted_keys = sorted(data.keys(), key=get_sort_key)
    for algo_name in sorted_keys:
        style = STYLES.get(algo_name)
        if not style: continue
        ax.scatter([], [], marker=style["m"], color=style["c"], s=80, label=style["label"])
        
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = get_sorted_handles_labels(handles, labels)
    
    # Create the legend centered in the figure
    legend = ax.legend(
        handles2, labels2,
        loc="center",
        ncol=3,            
        frameon=False,     
        fontsize=12,
        labelspacing=1.2,
        columnspacing=1.5
    )
    
    # Hide axes components manually instead of axis('off') to prevent
    # transform glitches, though axis('off') is usually fine.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Force a draw so we can calculate the bounding boxes
    fig.canvas.draw()
    
    gef_bboxes = []
    renderer = fig.canvas.get_renderer()
    
    # Support both old and new matplotlib versions for legend handles
    legend_handles_list = getattr(legend, 'legendHandles', getattr(legend, 'legend_handles', []))

    # 1. Collect bounding boxes for all GEF items (Text and Markers)
    found_count = 0
    for text, handle in zip(legend.get_texts(), legend_handles_list):
        # Strip string to ensure clean match
        if text.get_text().strip() in GEF_ORDER:
            found_count += 1
            try:
                # get_window_extent returns bbox in pixels (display coords)
                gef_bboxes.append(text.get_window_extent(renderer))
                gef_bboxes.append(handle.get_window_extent(renderer))
            except Exception:
                pass
    
    if gef_bboxes:
        # 2. Union them to get one big box in Display Coordinates
        bbox_display = mtransforms.Bbox.union(gef_bboxes)
        
        # 3. Add padding in Display Coordinates (pixels)
        # 15 pixels padding
        bbox_display_padded = bbox_display.padded(15)

        # 4. Transform Display Coordinates -> Axes Coordinates
        # This ensures the box scales/moves with the axes when saved.
        bbox_axes = ax.transAxes.inverted().transform_bbox(bbox_display_padded)
        
        # 5. SANITY CHECK: Ensure width/height are positive and non-zero
        width = max(bbox_axes.width, 0.01)
        height = max(bbox_axes.height, 0.01)

        # 6. Create the rectangular patch using Axes Coordinates
        # Use Rectangle instead of FancyBboxPatch to avoid path errors during savefig
        rect = Rectangle(
            (bbox_axes.x0, bbox_axes.y0),
            width, 
            height,
            facecolor='#E3F2FD',
            edgecolor='#2196F3',
            alpha=0.5,
            linewidth=2.0,
            linestyle='--',
            transform=ax.transAxes, # Anchor to axes
            zorder=-1 # Draw behind text
        )
                              
        ax.add_patch(rect)
    else:
        print("Warning: No GEF items found in legend. Rectangle will not be drawn.")
    
    dest = output_dir / "legend_box.pdf"
    # bbox_inches='tight' will crop the large (10x10) figure down to just the legend content
    fig.savefig(dest, bbox_inches='tight')
    print(f"Saved Legend Box: {dest}")
    plt.close(fig)

def create_combined_plot(
    comp_data: Dict[str, Tuple[float, float]],
    decomp_data: Dict[str, Tuple[float, float]],
    ra_data: Dict[str, Tuple[float, float]]
) -> plt.Figure:
    """
    Creates a combined figure with 3 subplots in a row:
    Compression, Decompression, Random Access.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # 1. Compression
    plot_on_axis(
        axes[0], 
        comp_data, 
        "Compression throughput (MB/s)", 
        "Compression Throughput", 
        log_scale_y=True
    )
    
    # 2. Decompression
    plot_on_axis(
        axes[1], 
        decomp_data, 
        "Decompression throughput (MB/s)", 
        "Decompression Throughput", 
        log_scale_y=True
    )
    
    # 3. Random Access
    plot_on_axis(
        axes[2], 
        ra_data, 
        "Random access speed (MB/s)", 
        "Random Access Speed", 
        log_scale_y=True
    )
    
    # Use handles/labels from one of the axes
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        handles2, labels2 = get_sorted_handles_labels(handles, labels)
        
        # Combined legend at the bottom of the figure
        fig.legend(
            handles2,
            labels2,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            fancybox=False,
            frameon=True,
            edgecolor='lightgrey',
            shadow=False,
            ncol=9, # Wider layout for combined plot
            fontsize=12,
            columnspacing=1.0
        )

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22) # Reserve more space at bottom for legend
    
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
    df['compression_ratio_pct'] = df['compression_ratio'] * 100.0

    # Group by label (compressor) and calculate mean across all datasets
    grouped = df.groupby('label').agg({
        'compression_ratio_pct': 'mean',
        'compression_throughput_mbs': 'mean',
        'decompression_throughput_mbs': 'mean',
        'random_access_mbs': 'mean'
    }).reset_index()

    # Prepare data dictionaries for plotting
    comp_data = {}
    for _, row in grouped.iterrows():
        comp_data[row['label']] = (row['compression_ratio_pct'], row['compression_throughput_mbs'])

    decomp_data = {}
    for _, row in grouped.iterrows():
        decomp_data[row['label']] = (row['compression_ratio_pct'], row['decompression_throughput_mbs'])

    ra_data = {}
    for _, row in grouped.iterrows():
        ra_data[row['label']] = (row['compression_ratio_pct'], row['random_access_mbs'])
    
    # Generate Individual Plots
    if comp_data:
        fig = create_benchmark_plot(
            comp_data,
            "Compression Ratio vs Compression Throughput",
            "Compression throughput (MB/s)",
            log_scale_y=True,
            show_legend=False
        )
        save_figure(fig, output_dir, "compression_throughput_vs_ratio")

    if decomp_data:
        fig = create_benchmark_plot(
            decomp_data,
            "Compression Ratio vs Decompression Throughput",
            "Decompression throughput (MB/s)",
            log_scale_y=True,
            show_legend=False
        )
        save_figure(fig, output_dir, "decompression_throughput_vs_ratio")

    if ra_data:
        fig = create_benchmark_plot(
            ra_data,
            "Compression Ratio vs Random Access Speed",
            "Random access speed (MB/s)",
            log_scale_y=True,
            show_legend=False
        )
        save_figure(fig, output_dir, "random_access_speed_vs_ratio")

    # Generate the Shared Legend
    if comp_data:
        export_legend_only(comp_data, output_dir)

    # Generate Combined Plot
    if comp_data and decomp_data and ra_data:
        fig = create_combined_plot(comp_data, decomp_data, ra_data)
        save_figure(fig, output_dir, "combined_benchmark_plot")
    else:
        print("Skipping combined plot due to missing data.")

if __name__ == "__main__":
    main()