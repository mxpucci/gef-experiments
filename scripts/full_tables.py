#!/usr/bin/env python3


import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================

# 1. Group Definitions (Lowercase for robust matching)
GENERAL_PURPOSE_KEYS = {"xz", "brotli", "zstd", "lz4", "snappy"}

# 2. GEF Variants Order
GEF_ORDER = [
    "rle_gef",
    "u_gef_approximate",
    "u_gef_optimal",
    "b_gef_approximate",
    "b_gef_optimal",
    "b_star_gef_approximate",
    "b_star_gef_optimal"
]

# 3. LaTeX Names (Full Notation for Ratio/Comp)
NAME_MAPPING_FULL = {
    # GEF Variants
    "rle_gef": r"\RLEGEF",
    "u_gef_approximate": r"$\hat{\UGEF}$",
    "u_gef_optimal": r"$\UGEF^*$",
    "b_gef_approximate": r"$\hat{\BGEF}$",
    "b_gef_optimal": r"$\BGEF^*$",
    "b_star_gef_approximate": r"$\hat{\BSTARGEF}$",
    "b_star_gef_optimal": r"$\BSTARGEF^*$",
    
    # Standard Compressors
    "neats": "NeaTS", "dac": "DAC", "gorilla": "Gorilla",
    "chimp": "Chimp", "chimp128": "Chimp128", "lz4": "Lz4",
    "zstd": "Zstd", "brotli": "Brotli", "snappy": "Snappy",
    "xz": "Xz", "leco": "LeCo", "alp": "ALP", "elf": "ELF",
    "tsxor": "TSXor"
}

# 4. LaTeX Names (Simplified for Decomp/RA)
NAME_MAPPING_SIMPLE = {
    "u_gef_optimal": r"\UGEF",
    "b_gef_optimal": r"\BGEF",
    "b_star_gef_optimal": r"\BSTARGEF",
    "rle_gef": r"\RLEGEF"
}

# 5. Default Dataset Order
DEFAULT_DATASET_ORDER = [
    "IT", "US", "ECG", "WD", "AP", "UK", "GE", "LON", 
    "LAT", "DP", "CT", "DU", "BT", "BW", "BM", "BP"
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_latex_name(raw_name: str, simplify: bool = False) -> str:
    if simplify and raw_name in NAME_MAPPING_SIMPLE:
        return NAME_MAPPING_SIMPLE[raw_name]
    if raw_name in NAME_MAPPING_FULL: return NAME_MAPPING_FULL[raw_name]
    if raw_name.lower() in NAME_MAPPING_FULL: return NAME_MAPPING_FULL[raw_name.lower()]
    return raw_name.capitalize()

def format_value(val: float, conversion_factor: float = 1.0) -> str:
    if pd.isna(val): return "-"
    return f"{val / conversion_factor:.2f}"

def get_vldb_formatting(val: float, all_values: List[float], is_min_best: bool, conversion_factor: float = 1.0) -> str:
    if pd.isna(val): return "-"
    formatted_str = format_value(val, conversion_factor)
    valid_values = [v for v in all_values if not pd.isna(v)]
    sorted_vals = sorted(valid_values, reverse=not is_min_best)
    try:
        rank = sorted_vals.index(val)
    except ValueError:
        return formatted_str
    if rank == 0: return f"\\textbf{{{formatted_str}}}"
    elif rank == 1: return f"\\underline{{{formatted_str}}}"
    elif rank == 2: return f"\\textit{{{formatted_str}}}"
    else: return formatted_str

def generate_single_table(
    df: pd.DataFrame,
    metric_col: str,
    col_groups: List[Tuple[str, List[str]]],
    datasets: List[str],
    caption: str,
    label: str,
    is_min_best: bool = False,
    conversion_factor: float = 1.0,
    simplify_headers: bool = False,
    footer_note: str = None  # NEW ARGUMENT FOR FOOTER
) -> str:
    
    # 1. Prepare Data Cache
    all_compressors = [c for _, cols in col_groups for c in cols]
    
    data_map = {}
    for ds in datasets:
        ds_clean = ds.replace(".bin", "").strip() 
        lookup_name = ds if ds in df['dataset'].values else (f"{ds}.bin" if f"{ds}.bin" in df['dataset'].values else ds)
        
        data_map[ds_clean] = {}
        for comp in all_compressors:
            subset = df[(df['dataset'] == lookup_name) & (df['compressor'] == comp)]
            if not subset.empty:
                data_map[ds_clean][comp] = subset[metric_col].mean()
            else:
                data_map[ds_clean][comp] = np.nan

    # 2. Build LaTeX Header
    latex = []
    latex.append(r"\begin{table}[htbp]") 
    
    legend = r" \textbf{Bold}: Best, \underline{Underlined}: Second, \textit{Italics}: Third."
    latex.append(f"\\caption{{{caption}{legend}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(r"\centering")
    latex.append(r"\scriptsize") 
    latex.append(r"\setlength{\tabcolsep}{3pt}") 
    
    # 3. Column Definition
    col_def_parts = ["@{}l"] 
    for _, cols in col_groups:
        if not cols: continue
        col_def_parts.append(f"*{{{len(cols)}}}{{c}}")
    col_def_parts.append("@{}")
    col_def = " ".join(col_def_parts)
    
    latex.append(f"\\begin{{tabular}}{{{col_def}}}")
    latex.append(r"\toprule")
    
    # 4. Super Headers
    super_headers = [r""] 
    cmidrules = []
    current_col_idx = 2 
    
    for title, cols in col_groups:
        if not cols: continue
        count = len(cols)
        super_headers.append(f"\\multicolumn{{{count}}}{{c}}{{\\textbf{{{title}}}}}")
        cmidrules.append(f"\\cmidrule(lr){{{current_col_idx}-{current_col_idx + count - 1}}}")
        current_col_idx += count

    latex.append(" & ".join(super_headers) + r" \\")
    latex.append("\n".join(cmidrules)) 
    
    # 5. Headers
    headers = [r"\textbf{Dataset}"]
    for _, cols in col_groups:
        for comp in cols:
            d_name = get_latex_name(comp, simplify=simplify_headers)
            headers.append(f"\\rotatebox{{45}}{{{d_name}}}")
            
    latex.append(" & ".join(headers) + r" \\")
    latex.append(r"\midrule")
    
    # 6. Data Rows
    for idx, ds in enumerate(datasets):
        if idx > 0 and idx % 5 == 0:
            latex.append(r"\addlinespace")

        ds_clean = ds.replace(".bin", "").strip()
        ds_display = ds_clean.replace("_", r"\_")
        
        if ds_clean not in data_map: continue 

        row_cells = [ds_display]
        
        row_values_all = [
            data_map[ds_clean][c] for c in all_compressors 
            if not pd.isna(data_map[ds_clean][c])
        ]
        
        for _, cols in col_groups:
            for comp in cols:
                val = data_map[ds_clean][comp]
                fmt = get_vldb_formatting(val, row_values_all, is_min_best, conversion_factor)
                row_cells.append(fmt)
            
        latex.append(" & ".join(row_cells) + r" \\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    
    # 7. ADD FOOTER NOTE (If provided)
    if footer_note:
        latex.append(r"\\") # spacing
        latex.append(r"\footnotesize")
        latex.append(footer_note)

    latex.append(r"\end{table}")
    
    return "\n".join(latex)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate VLDB style LaTeX tables.")
    parser.add_argument("csv_file", type=str, help="Path to input CSV file")
    parser.add_argument("output_dir", nargs="?", default="tables", help="Output directory")
    parser.add_argument("--compressors", type=str, help="Comma-separated list of compressors (excluding GEF).")
    parser.add_argument("--datasets", type=str, help="Comma-separated list of datasets to define row order.")
    
    args = parser.parse_args()
    csv_path, output_dir = Path(args.csv_file), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(f"Error: CSV file not found: {csv_path}")

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['dataset'] = df['dataset'].str.strip()
    df['compressor'] = df['compressor'].str.strip()

    # 1. IDENTIFY COMPRESSORS
    all_found = set(df['compressor'].unique())
    gef_list_all = [c for c in GEF_ORDER if c in all_found]
    gef_list_optimal = []
    if "rle_gef" in all_found: gef_list_optimal.append("rle_gef")
    
    def get_best_variant(base):
        opt = f"{base}_optimal"
        app = f"{base}_approximate"
        if opt in all_found: return opt
        if app in all_found: return app
        return None

    for base_key in ["u_gef", "b_gef", "b_star_gef"]:
        best = get_best_variant(base_key)
        if best: gef_list_optimal.append(best)
        
    if args.compressors:
        target_others = [c.strip() for c in args.compressors.split(',')]
        others_list = [c for c in target_others if c in all_found]
    else:
        others_list = [c for c in all_found if c not in GEF_ORDER and c != "b_star_gef_optimal"]

    general_list = []
    special_list = []
    for c in others_list:
        if c.lower() in GENERAL_PURPOSE_KEYS:
            general_list.append(c)
        else:
            special_list.append(c)
    general_list.sort()
    special_list.sort()
    
    # 2. DEFINE GROUPS
    column_groups_all = [
        ("General-purpose compressors", general_list),
        ("Special-purpose compressors", special_list),
        ("GEF variants", gef_list_all)
    ]
    
    column_groups_opt = [
        ("General-purpose compressors", general_list),
        ("Special-purpose compressors", special_list),
        ("GEF variants", gef_list_optimal)
    ]
    
    # 3. ORGANIZE DATASETS
    available_datasets = set(df['dataset'].unique())
    final_dataset_order = []
    target_order = [d.strip() for d in args.datasets.split(',')] if args.datasets else DEFAULT_DATASET_ORDER
    
    for ds in target_order:
        if ds in available_datasets: final_dataset_order.append(ds)
        elif f"{ds}.bin" in available_datasets: final_dataset_order.append(f"{ds}.bin")
            
    remaining = sorted([d for d in available_datasets if d not in final_dataset_order])
    final_dataset_order.extend(remaining)

    # 4. CAPTION NOTE -> MOVED TO FOOTER
    # This text will be appended at the bottom of the table environment
    split_point_note = r"With $\hat C$ and $C^*$, we denote the GEF variant $C$ that uses either its approximated or optimal split point, respectively."

    # 5. GENERATE TABLES
    
    # Table 1: Ratio (With Footer Note)
    t1 = generate_single_table(df, 'compression_ratio', column_groups_all, final_dataset_order,
        "Compression Ratio (\\%)", "tab:ratio", is_min_best=True, simplify_headers=False, footer_note=split_point_note)
    with open(output_dir / "table_compression_ratio.tex", "w") as f: f.write(t1)

    # Table 2: Comp Speed (With Footer Note)
    t2 = generate_single_table(df, 'compression_throughput_mbs', column_groups_all, final_dataset_order,
        "Compression Throughput (MB/s)", "tab:comp_speed", is_min_best=False, simplify_headers=False, footer_note=split_point_note)
    with open(output_dir / "table_compression_throughput.tex", "w") as f: f.write(t2)

    # Table 3: Decomp Speed (Optimal Only, Simple Headers, No Note)
    t3 = generate_single_table(df, 'decompression_throughput_mbs', column_groups_opt, final_dataset_order,
        "Decompression Throughput (GB/s)", "tab:decomp_speed", is_min_best=False, conversion_factor=1024.0, simplify_headers=True)
    with open(output_dir / "table_decompression_throughput.tex", "w") as f: f.write(t3)

    # Table 4: Random Access (Optimal Only, Simple Headers, No Note)
    t4 = generate_single_table(df, 'random_access_mbs', column_groups_opt, final_dataset_order,
        "Random Access Throughput (MB/s)", "tab:ra_speed", is_min_best=False, simplify_headers=True)
    with open(output_dir / "table_random_access.tex", "w") as f: f.write(t4)

    print(f"Tables generated in {output_dir}/")

if __name__ == "__main__":
    main()