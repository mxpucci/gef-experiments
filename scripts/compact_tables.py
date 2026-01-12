#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Raw names of GEF variants in the CSV
GEF_VARIANTS = [
    "rle_gef",
    "u_gef_approximate",
    "u_gef_optimal",
    "b_gef_approximate",
    "b_gef_optimal",
    "b_star_gef_approximate",
    "b_star_gef_optimal"
]

# Compressors with Random Access Support
RA_COMPRESSORS = {
    "alp", "leco", "neats", "sneats", "leats", "dac",
    *GEF_VARIANTS 
}

# Mapping raw names to LaTeX headers
LATEX_MAPPING = {
    "rle_gef": r"\RLEGEF",
    "u_gef_approximate": r"\UGEF (Approx.)",
    "u_gef_optimal": r"\UGEF (Opt.)",
    "b_gef_approximate": r"\BGEF (Approx.)",
    "b_gef_optimal": r"\BGEF (Opt.)",
    "b_star_gef_approximate": r"\BSTARGEF"
}

# Fallback mappings
NAME_MAPPING = {
    "neats": "NeaTS", "dac": "DAC", "gorilla": "Gorilla",
    "chimp": "Chimp", "chimp128": "Chimp128", "tsxor": "TSXor",
    "elf": "ELF", "camel": "Camel", "falcon": "Falcon",
    "lz4": "Lz4", "zstd": "Zstd", "brotli": "Brotli",
    "snappy": "Snappy", "xz": "Xz", "leco": "LeCo", "alp": "ALP"
}

GEF_COLUMN_ORDER = [
    "rle_gef", "u_gef_approximate", "u_gef_optimal",
    "b_gef_approximate", "b_gef_optimal", "b_star_gef_approximate"
]

DATASET_ORDER = [
    "AP.bin", "BM.bin", "BP.bin", "BT.bin", "CH.bin", "CM.bin", "CN.bin", "CS.bin",
    "DM.bin", "DN.bin", "DS.bin", "FL.bin", "HI.bin", "HN.bin", "HS.bin",
    "LL.bin", "LP.bin", "ML.bin", "MP.bin", "PL.bin", "PN.bin", "PS.bin",
    "RD.bin", "RL.bin", "RN.bin", "RS.bin", "SL.bin", "SP.bin", "ST.bin", "SV.bin",
    "TM.bin", "TP.bin", "TT.bin", "US.bin" 
]

# Formatting Config
# Using scriptsize + 0pt padding ensures we fit maximum data without scaling the font
TABLE_FONT_SIZE = r"\scriptsize"
TAB_COL_SEP = "0pt" 
ARRAY_STRETCH = "0.9" 

# ==========================================
# 2. HELPERS
# ==========================================

def get_latex_name(raw_name: str, use_short_gef_names: bool = False) -> str:
    if use_short_gef_names:
        if raw_name == "u_gef_optimal": return r"\UGEF"
        if raw_name == "b_gef_optimal": return r"\BGEF"
            
    if raw_name in LATEX_MAPPING:
        return LATEX_MAPPING[raw_name]
    
    # Check NAME_MAPPING (try exact, then lowercase)
    if raw_name in NAME_MAPPING:
        return NAME_MAPPING[raw_name]
    
    lower_name = raw_name.lower()
    if lower_name in NAME_MAPPING:
        return NAME_MAPPING[lower_name]
        
    return raw_name.capitalize()

def format_smart(val: float, is_ratio: bool = False, conversion_factor: float = 1.0) -> str:
    if pd.isna(val):
        return "-"
    
    scaled_val = val / conversion_factor
    
    if is_ratio:
        return f"{scaled_val * 100:.2f}"
    else:
        # Integers for large numbers (MB/s) save space and reduce clutter
        if scaled_val >= 10:
            return f"{scaled_val:.0f}"
        else:
            return f"{scaled_val:.2f}"

def get_rank_formatting(
    val: float, 
    all_values_for_ranking: List[float], 
    local_best: float, 
    global_best: float, 
    is_ratio: bool,
    ranking_style: str = "split", 
    is_min_best: bool = False,
    conversion_factor: float = 1.0
) -> str:
    if pd.isna(val):
        return "-"
    
    formatted = format_smart(val, is_ratio, conversion_factor)
    
    is_global = np.isclose(val, global_best, rtol=1e-05)
    is_local = np.isclose(val, local_best, rtol=1e-05)

    if ranking_style == "split":
        if is_global:
            return f"\\underline{{\\textbf{{{formatted}}}}}"
        elif is_local:
            return f"\\textbf{{{formatted}}}"
        else:
            return formatted
            
    elif ranking_style == "global_vldb":
        valid_vals = sorted([v for v in all_values_for_ranking if not pd.isna(v)], reverse=not is_min_best)
        try:
            rank = valid_vals.index(val)
        except ValueError:
            return formatted
            
        if rank == 0:
            return f"\\textbf{{{formatted}}}"
        elif rank == 1:
            return f"\\underline{{{formatted}}}"
        elif rank == 2:
            return f"\\textit{{{formatted}}}"
        else:
            return formatted
            
    return formatted

def generate_caption(base_caption: str, ranking_style: str) -> str:
    """Appends the formatting explanation to the caption."""
    if ranking_style == "split":
        legend = r" \textbf{Bold}: Local Best, \underline{\textbf{Underlined}}: Global Best."
    elif ranking_style == "global_vldb":
        legend = r" \textbf{Bold}: Best, \underline{Underlined}: Second, \textit{Italics}: Third."
    else:
        legend = ""
    return base_caption + legend

def generate_latex_table(
    df: pd.DataFrame, 
    metric_col: str, 
    no_ra_compressors: List[str], 
    ra_compressors: List[str],
    datasets: List[str],
    base_caption: str,
    label: str,
    is_min_best: bool = False,
    is_ratio: bool = False,
    ranking_style: str = "split",
    use_short_gef_names: bool = False,
    merge_gef_variants: bool = False,
    conversion_factor: float = 1.0,
    width_command: str = r"\columnwidth", # Allow switching between columnwidth and textwidth
    skip_caption_legend: bool = False
) -> str:
    
    # 1. Prepare Columns
    ra_display_cols = []
    processed = set()
    
    target_list = ra_compressors
    for c in target_list:
        if c in processed: continue
        
        if merge_gef_variants:
            if c == "u_gef_approximate" and "u_gef_optimal" in target_list:
                ra_display_cols.append((r"\UGEF", ["u_gef_approximate", "u_gef_optimal"]))
                processed.update(["u_gef_approximate", "u_gef_optimal"])
            elif c == "u_gef_optimal" and "u_gef_approximate" in processed: continue
            elif c == "b_gef_approximate" and "b_gef_optimal" in target_list:
                ra_display_cols.append((r"\BGEF", ["b_gef_approximate", "b_gef_optimal"]))
                processed.update(["b_gef_approximate", "b_gef_optimal"])
            elif c == "b_gef_optimal" and "b_gef_approximate" in processed: continue
            else:
                ra_display_cols.append((get_latex_name(c, use_short_gef_names), [c]))
                processed.add(c)
        else:
            ra_display_cols.append((get_latex_name(c, use_short_gef_names), [c]))
            
    # 2. Setup Header
    if skip_caption_legend:
        full_caption = base_caption
    else:
        full_caption = generate_caption(base_caption, ranking_style)
    
    latex = []
    if width_command == r"\textwidth":
        latex.append(r"\begin{table*}[htbp]")
    else:
        latex.append(r"\begin{table}[htbp]")

    latex.append(f"\\caption{{{full_caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(r"\centering")
    latex.append(TABLE_FONT_SIZE)
    latex.append(f"\\renewcommand{{\\arraystretch}}{{{ARRAY_STRETCH}}}")
    latex.append(f"\\setlength{{\\tabcolsep}}{{{TAB_COL_SEP}}}")
    
    # UNIFORMITY FIX: tabular* with \extracolsep{\fill}
    # This forces the table to take exactly `width_command` space.
    # The columns will spread out automatically.
    
    total_cols = 1 + len(no_ra_compressors) + len(ra_display_cols)
    # l column (dataset) is flush left. The rest are spread.
    
    # We need a separator column between No-RA and RA groups
    # Column definition: Dataset (l) | No-RA (c...) | SEPARATOR | RA (c...)
    # Actually, we can just use a vertical bar in the column definition if we want a line,
    # or just rely on the headers.
    # The user asked for a "separator" and "header".
    # Let's add a multicolumn header row.
    
    col_def = f"l @{{\\extracolsep{{\\fill}}}} " + \
              f"*{{{len(no_ra_compressors)}}}{{c}} " + \
              f"@{{\\quad|\\quad}} " + \
              f"*{{{len(ra_display_cols)}}}{{c}} @{{}}"
    
    latex.append(f"\\begin{{tabular*}}{{{width_command}}}{{{col_def}}}")
    latex.append(r"\toprule")
    
    # Super Header Row
    if len(no_ra_compressors) > 0 and len(ra_display_cols) > 0:
        latex.append(
            f" & \\multicolumn{{{len(no_ra_compressors)}}}{{c@{{\\quad|\\quad}}}}{{\\textbf{{Without Random Access}}}}" + 
            f" & \\multicolumn{{{len(ra_display_cols)}}}{{c}}{{\\textbf{{With Random Access}}}} \\\\"
        )
    
    # Header Row
    row_headers = [r"\textbf{Dataset}"]
    
    for c in no_ra_compressors:
        name = get_latex_name(c, use_short_gef_names)
        row_headers.append(f"\\adjustbox{{angle=45,lap=\\width-1em}}{{{name}}}")
        
    for h, _ in ra_display_cols:
        row_headers.append(f"\\adjustbox{{angle=45,lap=\\width-1em}}{{{h}}}")
        
    latex.append(" & ".join(row_headers) + r" \\")
    latex.append(r"\midrule")
    
    # 3. Data Rows
    for idx, ds in enumerate(datasets):
        # if idx > 0 and idx % 5 == 0:
        #    latex.append(r"\addlinespace[2pt]")

        row_str = [ds.replace(".bin", "").replace("_", r"\_")]
        
        no_ra_vals_map = {} 
        ra_vals_map = {} 
        
        for c in no_ra_compressors:
            val = df[(df['dataset'] == ds) & (df['compressor'] == c)][metric_col].mean()
            no_ra_vals_map[c] = val
            
        for _, sub_cols in ra_display_cols:
            for c in sub_cols:
                val = df[(df['dataset'] == ds) & (df['compressor'] == c)][metric_col].mean()
                ra_vals_map[c] = val

        valid_no_ra = [v for v in no_ra_vals_map.values() if not pd.isna(v)]
        valid_ra = [v for v in ra_vals_map.values() if not pd.isna(v)]
        valid_all = valid_no_ra + valid_ra
        
        if not valid_all:
             # Just fill with empty strings/dashes respecting the new separator logic
             # The separator is in the col_def, so we just need enough &
             # But wait, python join doesn't care about col_def separators.
             # We just need N columns of data.
             latex.append(" & ".join(row_str + ["-"] * (len(no_ra_compressors) + len(ra_display_cols))) + r" \\")
             continue

        if is_min_best:
            best_no_ra = min(valid_no_ra) if valid_no_ra else np.nan
            best_ra = min(valid_ra) if valid_ra else np.nan
            best_global = min(valid_all)
        else:
            best_no_ra = max(valid_no_ra) if valid_no_ra else np.nan
            best_ra = max(valid_ra) if valid_ra else np.nan
            best_global = max(valid_all)
            
        for c in no_ra_compressors:
            val = no_ra_vals_map[c]
            fmt = get_rank_formatting(val, valid_all, best_no_ra, best_global, is_ratio, ranking_style, is_min_best, conversion_factor)
            row_str.append(fmt)
            
        for _, sub_cols in ra_display_cols:
            parts = []
            for raw_col in sub_cols:
                val = ra_vals_map[raw_col]
                fmt = get_rank_formatting(val, valid_all, best_ra, best_global, is_ratio, ranking_style, is_min_best, conversion_factor)
                parts.append(fmt)
            row_str.append(" / ".join(parts))

        latex.append(" & ".join(row_str) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular*}") # Note the *
    if width_command == r"\textwidth":
        latex.append(r"\end{table*}")
    else:
        latex.append(r"\end{table}")
    
    return "\n".join(latex)

def generate_combined_table(
    df: pd.DataFrame,
    no_ra_compressors: List[str],
    ra_compressors_all: List[str],
    ra_compressors_optimal: List[str],
    datasets: List[str],
    base_caption: str,
    label: str,
    threshold: float,
    include_approx: bool = False
) -> str:
    
    full_caption = base_caption + r" \textbf{Bold}: Best, \underline{Underlined}: Second, \textit{Italics}: Third."
    ra_for_ratio_comp = ra_compressors_all if include_approx else ra_compressors_optimal
    
    sections = [
        (r"Comp. ratio (\%)", "compression_ratio", True, True, "global_vldb", ra_for_ratio_comp, include_approx, 1.0), 
        (r"Comp. (MB/s)", "compression_throughput_mbs", False, False, "global_vldb", ra_for_ratio_comp, include_approx, 1.0), 
        (r"Decomp. (GB/s)", "decompression_throughput_mbs", False, False, "global_vldb", ra_compressors_optimal, False, 1024.0), 
        (r"Random Access (MB/s)", "random_access_mbs", False, False, "global_vldb", ra_compressors_optimal, False, 1.0) 
    ]
    
    ra_display_cols_master = []
    processed = set()
    layout_source = ra_compressors_all if include_approx else ra_compressors_optimal
    
    for c in layout_source:
        if c in processed: continue
        if include_approx:
            if c == "u_gef_approximate" and "u_gef_optimal" in layout_source:
                ra_display_cols_master.append((r"\UGEF", ["u_gef_approximate", "u_gef_optimal"]))
                processed.update(["u_gef_approximate", "u_gef_optimal"])
            elif c == "u_gef_optimal" and "u_gef_approximate" in processed: continue
            elif c == "b_gef_approximate" and "b_gef_optimal" in layout_source:
                ra_display_cols_master.append((r"\BGEF", ["b_gef_approximate", "b_gef_optimal"]))
                processed.update(["b_gef_approximate", "b_gef_optimal"])
            elif c == "b_gef_optimal" and "b_gef_approximate" in processed: continue
            else:
                ra_display_cols_master.append((get_latex_name(c, False), [c]))
                processed.add(c)
        else:
            ra_display_cols_master.append((get_latex_name(c, True), [c]))
            processed.add(c)

    latex = []
    latex.append(r"\begin{table*}[htbp]")
    latex.append(f"\\caption{{{full_caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(r"\centering")
    latex.append(TABLE_FONT_SIZE)
    latex.append(f"\\renewcommand{{\\arraystretch}}{{{ARRAY_STRETCH}}}")
    latex.append(f"\\setlength{{\\tabcolsep}}{{{TAB_COL_SEP}}}")
    
    # UNIFORMITY FIX: tabular* with \textwidth
    total_cols = 1 + len(no_ra_compressors) + len(ra_display_cols_master)
    col_def = f"l @{{\\extracolsep{{\\fill}}}} *{{{total_cols-1}}}{{c}} @{{}}"
    
    latex.append(f"\\begin{{tabular*}}{{\\textwidth}}{{{col_def}}}")
    latex.append(r"\toprule")
    
    row_headers = [r"\textbf{Dataset}"]
    for c in no_ra_compressors:
        row_headers.append(f"\\adjustbox{{angle=45,lap=\\width-1em}}{{{get_latex_name(c, False)}}}")
    for h, _ in ra_display_cols_master:
        row_headers.append(f"\\adjustbox{{angle=45,lap=\\width-1em}}{{{h}}}")
        
    latex.append(" & ".join(row_headers) + r" \\")
    latex.append(r"\midrule")
    
    for title, metric, is_min, is_ratio, rank_style, ra_src_list, merge_flag, conversion_factor in sections:
        latex.append(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{title}}}}} \\\\")
        latex.append(r"\midrule")
        
        for idx, ds in enumerate(datasets):
            if idx > 0 and idx % 5 == 0: latex.append(r"\addlinespace[2pt]")
            
            row_str = [ds.replace(".bin", "").replace("_", r"\_")]
            
            no_ra_vals = {}
            ra_section_vals = {}
            
            for c in no_ra_compressors:
                val = df[(df['dataset'] == ds) & (df['compressor'] == c)][metric].mean()
                no_ra_vals[c] = val
                
            for c in ra_src_list:
                val = df[(df['dataset'] == ds) & (df['compressor'] == c)][metric].mean()
                ra_section_vals[c] = val

            all_ranking = list(no_ra_vals.values()) + list(ra_section_vals.values())
            valid_all = [v for v in all_ranking if not pd.isna(v)]
            
            if not valid_all:
                latex.append(" & ".join(row_str + ["-"] * (total_cols-1)) + r" \\")
                continue

            valid_no_ra = [v for v in no_ra_vals.values() if not pd.isna(v)]
            valid_ra = [v for v in ra_section_vals.values() if not pd.isna(v)]

            if is_min:
                best_g = min(valid_all)
                best_nr = min(valid_no_ra) if valid_no_ra else np.nan
                best_r = min(valid_ra) if valid_ra else np.nan
            else:
                best_g = max(valid_all)
                best_nr = max(valid_no_ra) if valid_no_ra else np.nan
                best_r = max(valid_ra) if valid_ra else np.nan
                
            for c in no_ra_compressors:
                fmt = get_rank_formatting(no_ra_vals[c], valid_all, best_nr, best_g, is_ratio, rank_style, is_min, conversion_factor)
                row_str.append(fmt)
                
            for h, cols in ra_display_cols_master:
                parts = []
                active_sub_cols = [c for c in cols if c in ra_src_list]
                
                if not active_sub_cols:
                    parts.append("-")
                else:
                    target_cols = cols if merge_flag else active_sub_cols
                    for c in target_cols:
                        if c in active_sub_cols:
                            val = ra_section_vals.get(c, np.nan)
                            fmt = get_rank_formatting(val, valid_all, best_r, best_g, is_ratio, rank_style, is_min, conversion_factor)
                            parts.append(fmt)
                        else:
                            parts.append("-")
                            
                row_str.append(" / ".join(parts))
                
            latex.append(" & ".join(row_str) + r" \\")

        latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular*}")
    latex.append(r"\end{table*}")
    
    return "\n".join(latex)

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for benchmark results.")
    parser.add_argument("csv_file", type=str, help="Path to input CSV file")
    parser.add_argument("threshold", type=float, help="Compression Ratio threshold.")
    parser.add_argument("output_dir", nargs="?", default="tables", help="Output directory")
    parser.add_argument("--include-approx", action="store_true", help="Include approximate values.")
    parser.add_argument("--compressors", type=str, help="Comma-separated list of compressors to compare against GEF variants.")
    parser.add_argument("--datasets", type=str, help="Comma-separated list of datasets to define order (and priority).")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir)
    threshold = args.threshold
    include_approx = args.include_approx
    
    # Parse target compressors if provided
    target_compressors = None
    if args.compressors:
        target_compressors = set(c.strip().lower() for c in args.compressors.split(','))

    # Parse dataset order if provided
    custom_dataset_order = None
    if args.datasets:
        custom_dataset_order = [d.strip() for d in args.datasets.split(',')]

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['dataset'] = df['dataset'].str.strip()
    df['compressor'] = df['compressor'].str.strip()

    # Filter Logic
    all_compressors = df['compressor'].unique()
    avg_ratios = df.groupby('compressor')['compression_ratio'].mean()
    
    no_ra_list = []
    ra_list = []
    
    selected_others_no_ra = []
    selected_others_ra = []
    
    for comp in all_compressors:
        if comp == "b_star_gef_optimal": continue
        is_gef = comp in GEF_VARIANTS
        
        # Filtering logic
        if target_compressors is not None:
            # If compressors are explicitly specified:
            # 1. Always include GEF variants (as per "compare GEF variants against...")
            # 2. Include others ONLY if they are in the target list
            if not is_gef:
                if comp.lower() not in target_compressors:
                    continue
        else:
            # Default behavior: Filter by threshold if not GEF
            if not is_gef and avg_ratios.get(comp, 1.0) > threshold: continue
        
        supports_ra = comp.lower() in RA_COMPRESSORS
        if is_gef: pass
        else:
            if supports_ra: selected_others_ra.append(comp)
            else: selected_others_no_ra.append(comp)
                
    selected_others_no_ra.sort()
    selected_others_ra.sort()
    
    no_ra_list = selected_others_no_ra
    ra_list = list(selected_others_ra)
    for gef in GEF_COLUMN_ORDER:
        if gef in all_compressors: ra_list.append(gef)
    
    ra_list_optimal_only = []
    for c in ra_list:
        if c == "u_gef_approximate" or c == "b_gef_approximate": continue
        ra_list_optimal_only.append(c)

    # Dataset Order
    available_datasets = set(df['dataset'].unique())
    final_dataset_order = []
    
    order_source = custom_dataset_order if custom_dataset_order else DATASET_ORDER
    
    for ds in order_source:
        if ds in available_datasets:
            final_dataset_order.append(ds)
        elif f"{ds}.bin" in available_datasets:
            final_dataset_order.append(f"{ds}.bin")
            
    remaining = sorted(list(available_datasets - set(final_dataset_order)))
    final_dataset_order.extend(remaining)
    
    # Generate
    output_dir.mkdir(parents=True, exist_ok=True)
    ra_for_ratio = ra_list if include_approx else ra_list_optimal_only

    # Individual Tables
    t1 = generate_latex_table(
        df, 'compression_ratio', no_ra_list, ra_for_ratio, final_dataset_order,
        r"Compression ratios across datasets. We highlight in \textbf{bold} the best result in each family, and \underline{underline} the best result overall.", 
        "tab:ratio",
        is_min_best=True, is_ratio=True, ranking_style="split",
        merge_gef_variants=include_approx, use_short_gef_names=True,
        skip_caption_legend=True
    )
    with open(output_dir / "table_compression_ratio.tex", "w") as f: f.write(t1)
        
    t2 = generate_latex_table(
        df, 'compression_throughput_mbs', no_ra_list, ra_for_ratio, final_dataset_order,
        "Comp. Speed (MB/s)", "tab:comp_speed",
        is_min_best=False, is_ratio=False, ranking_style="global_vldb",
        merge_gef_variants=include_approx, use_short_gef_names=True
    )
    with open(output_dir / "table_compression_throughput.tex", "w") as f: f.write(t2)

    t3 = generate_latex_table(
        df, 'decompression_throughput_mbs', no_ra_list, ra_list_optimal_only, final_dataset_order,
        "Decomp. Speed (GB/s)", "tab:decomp_speed",
        is_min_best=False, is_ratio=False, ranking_style="global_vldb",
        use_short_gef_names=True, merge_gef_variants=False, conversion_factor=1024.0
    )
    with open(output_dir / "table_decompression_throughput.tex", "w") as f: f.write(t3)
        
    t4 = generate_latex_table(
        df, 'random_access_mbs', [], ra_list_optimal_only, final_dataset_order,
        "Random Access (MB/s)", "tab:ra_speed",
        is_min_best=False, is_ratio=False, ranking_style="global_vldb",
        use_short_gef_names=True, merge_gef_variants=False
    )
    with open(output_dir / "table_random_access.tex", "w") as f: f.write(t4)

    # Combined
    combined = generate_combined_table(
        df, no_ra_list, ra_list, ra_list_optimal_only, final_dataset_order,
        "Combined Benchmarks", "tab:combined", threshold, include_approx=include_approx
    )
    with open(output_dir / "table_combined.tex", "w") as f: f.write(combined)

    print(f"Optimized tables generated in {output_dir}/")

    # Calculate Max Improvement for GEF variants
    gef_targets = ["b_star_gef_approximate", "rle_gef", "u_gef_optimal", "b_gef_optimal"]
    
    # Identify non-GEF RA compressors for comparison (exclude ALL GEF variants)
    non_gef_ra_compressors = [c for c in ra_list if c not in GEF_VARIANTS]
    # For "No RA" comparison (which effectively means Best State-of-the-Art), we use ALL non-GEF compressors
    # (including RA ones if they happen to be better, excluding GEF variants)
    all_non_gef_compressors = [c for c in all_compressors if c not in GEF_VARIANTS]

    # Detect actual compressor names for ALP and LeCo (case-insensitive)
    alp_name = next((c for c in all_compressors if c.lower() == 'alp'), None)
    leco_name = next((c for c in all_compressors if c.lower() == 'leco'), None)

    for target_comp in gef_targets:
        if target_comp not in all_compressors:
            continue
            
        latex_name = get_latex_name(target_comp, use_short_gef_names=True)
        print(f"\nCalculating improvements for {target_comp} ({latex_name})...")
        max_imp_no_ra = -float('inf')
        max_imp_ra = -float('inf')
        max_imp_alp = -float('inf')
        max_imp_leco = -float('inf')
        
        best_ds_no_ra = None
        best_ds_ra = None
        best_ds_alp = None
        best_ds_leco = None

        for ds in final_dataset_order:
            ds_df = df[df['dataset'] == ds]
            
            target_vals = ds_df[ds_df['compressor'] == target_comp]['compression_ratio']
            if target_vals.empty: continue
            val_target = target_vals.mean()
            if pd.isna(val_target): continue

            # Best Non-GEF (All) - effectively replacing "No RA" comparison with "Best SOTA"
            no_ra_vals = ds_df[ds_df['compressor'].isin(all_non_gef_compressors)]['compression_ratio']
            valid_no_ra = [v for v in no_ra_vals if not pd.isna(v)]
            if valid_no_ra:
                best_no_ra = min(valid_no_ra)
                # Improvement %: (Reference - Target) / Reference * 100
                imp = (best_no_ra - val_target) / best_no_ra * 100.0
                if imp > max_imp_no_ra:
                    max_imp_no_ra = imp
                    best_ds_no_ra = ds

            # Best RA (Excluding all GEF variants)
            ra_vals = ds_df[ds_df['compressor'].isin(non_gef_ra_compressors)]['compression_ratio']
            valid_ra = [v for v in ra_vals if not pd.isna(v)]
            if valid_ra:
                best_ra = min(valid_ra)
                imp = (best_ra - val_target) / best_ra * 100.0
                if imp > max_imp_ra:
                    max_imp_ra = imp
                    best_ds_ra = ds

            # Improvement vs ALP
            if alp_name:
                alp_vals = ds_df[ds_df['compressor'] == alp_name]['compression_ratio']
                if not alp_vals.empty:
                    val_alp = alp_vals.mean()
                    if not pd.isna(val_alp):
                        imp = (val_alp - val_target) / val_alp * 100.0
                        if imp > max_imp_alp:
                            max_imp_alp = imp
                            best_ds_alp = ds

            # Improvement vs LeCo
            if leco_name:
                leco_vals = ds_df[ds_df['compressor'] == leco_name]['compression_ratio']
                if not leco_vals.empty:
                    val_leco = leco_vals.mean()
                    if not pd.isna(val_leco):
                        imp = (val_leco - val_target) / val_leco * 100.0
                        if imp > max_imp_leco:
                            max_imp_leco = imp
                            best_ds_leco = ds

        if best_ds_no_ra:
            print(f"Max improvement vs Best SOTA: {max_imp_no_ra:.2f}% (Dataset: {best_ds_no_ra})")
        else:
            print("Max improvement vs Best SOTA: N/A")
            
        if best_ds_ra:
            print(f"Max improvement vs RA:        {max_imp_ra:.2f}% (Dataset: {best_ds_ra})")
        else:
            print("Max improvement vs RA:        N/A")

        if best_ds_alp:
            print(f"Max improvement vs ALP:       {max_imp_alp:.2f}% (Dataset: {best_ds_alp})")
        else:
            print("Max improvement vs ALP:       N/A")

        if best_ds_leco:
            print(f"Max improvement vs LeCo:      {max_imp_leco:.2f}% (Dataset: {best_ds_leco})")
        else:
            print("Max improvement vs LeCo:      N/A")

if __name__ == "__main__":
    main()