#!/usr/bin/env python3
"""Generate the paper's per-dataset LaTeX table from a LosslessBenchmark CSV.

usage:
  paper_table.py [results.csv] [compressors] [-o table.tex]
                 [--metrics ratio,comp,ra] [--label tab:compression_all]

Without arguments it reads benchmark_results/results.csv and writes the paper's table
(PAPER_COMPRESSORS) to stdout.

Compressors are given by their table label; '|' separates the group without
native random access (left) from the group with it (right). Labels are mapped
to CSV compressor names through LABELS below (extend it as needed); a label
that is not in LABELS is looked up verbatim in the CSV.

Bold marks the best value of each row within each group, underline the best
overall (lower is better for the ratio, higher for throughputs). The Average row is the arithmetic mean over the datasets.
Throughputs are converted from the CSV's MiB/s to decimal MB/s.
"""
import argparse, csv, sys
from collections import defaultdict

LABELS = {
    "Pcodec": "pcodec_i64", "Pcodec (f64)": "pcodec",
    "Lz4": "lz4", "Zstd": "zstd", "Xz": "xz", "Brotli": "brotli", "Snappy": "snappy",
    "Gorilla": "Gorilla", "Chimp": "Chimp", "Chimp128": "Chimp128", "TSXor": "TSXor",
    "Elf": "Elf", "Elf*": "elfstar", "SElf*": "selfstar", "DeXOR": "dexor",
    "FLEA": "flea", "Camel": "Camel", "MacaqueV": "macaque", "Fauna": "fauna",
    "FastPFor": "fastpfor_delta", "FastPFor (FOR)": "fastpfor",
    "DAC": "DAC", "Partitioned EF": "PEF", "BUFF": "BUFF", "LeCo": "LeCo",
    "NeaTS": "NeaTS", "ALP": "ALP",
    # GEF fed with doubles (floating-point forks); B*-GEF with the approximate split point
    r"\RLEGEF": "rle_gef_f64",
    r"\BSTARGEF": "b_star_gef_approximate_f64",
    # GEF on the scaled integers
    r"\RLEGEF (int)": "rle_gef",
    r"\BSTARGEF (int)": "b_star_gef_approximate",
    r"\BSTARGEF (opt, int)": "b_star_gef_optimal",
}

PAPER_COMPRESSORS = ("Pcodec,FastPFor,Lz4,Zstd,Xz,Brotli,Gorilla,FLEA,Chimp128,SElf* | "
                     "DAC,Partitioned EF,BUFF,LeCo,NeaTS,ALP,\\RLEGEF,\\BSTARGEF")

MIB_TO_MB = 1048576 / 1e6  # the harness's *_mbs columns are MiB/s
METRICS = {
    "ratio": ("compression_ratio", "Compression ratio (\\%)", False, lambda v: 100 * v),
    "comp": ("compression_throughput_mbs", "Compression throughput (MB/s)", True, lambda v: v * MIB_TO_MB),
    "decomp": ("decompression_throughput_mbs", "Decompression throughput (MB/s)", True, lambda v: v * MIB_TO_MB),
    "ra": ("random_access_mbs", "Random access throughput (MB/s)", True, lambda v: v * MIB_TO_MB),
}
CAPTION_NAMES = {"ratio": "compression ratio (\\%)", "comp": "compression throughput (MB/s)",
                 "decomp": "decompression throughput (MB/s)", "ra": "random access throughput (MB/s)"}


def fmt(metric, v):
    if metric == "ratio":
        return f"{v:.2f}"
    if v < 1:
        return f"{v:.2f}"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="benchmark_results/results.csv")
    ap.add_argument("compressors", nargs="?", default=PAPER_COMPRESSORS,
                    help="comma-separated labels; '|' splits the two groups")
    ap.add_argument("-o", "--output")
    ap.add_argument("--metrics", default="ratio,comp,ra", help="any of: " + ",".join(METRICS))
    ap.add_argument("--label", default="tab:compression_all")
    args = ap.parse_args()

    groups = [[c.strip() for c in g.split(",") if c.strip()] for g in args.compressors.split("|")]
    if len(groups) != 2:
        sys.exit("expected exactly one '|' separating the two groups")
    labels = groups[0] + groups[1]
    metrics = [m.strip() for m in args.metrics.split(",")]
    for m in metrics:
        if m not in METRICS:
            sys.exit(f"unknown metric {m}")

    data = defaultdict(dict)  # (csv name) -> dataset -> row
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            data[r["compressor"]][r["dataset"]] = r
    names = [LABELS.get(l, l) for l in labels]
    for l, n in zip(labels, names):
        if n not in data:
            sys.exit(f"compressor '{l}' (CSV name '{n}') not found in {args.csv}")
    datasets = sorted(set.intersection(*(set(data[n]) for n in names)))
    for n in names:
        missing = set().union(*(set(data[x]) for x in names)) - set(data[n])
        if missing:
            print(f"warning: {n} lacks {sorted(missing)}; those datasets are dropped", file=sys.stderr)

    n1, n2 = len(groups[0]), len(groups[1])
    out = []
    w = out.append
    w(r"\begin{table*}[t]")
    parts = [CAPTION_NAMES[m] for m in metrics]
    listed = parts[0] if len(parts) == 1 else (
        f"{parts[0]} and {parts[1]}" if len(parts) == 2 else ", ".join(parts[:-1]) + ", and " + parts[-1])
    w(r"\caption{" + listed[0].upper() + listed[1:]
      + r". \textbf{Bold}: best within each group; \underline{underlined}: best overall. All compressors are"
      r" queried through our library, which provides random access to the methods that do not support it"
      r" natively by blocking the input into $1{,}000$-value chunks.}")
    w(r"\label{" + args.label + "}")
    w(r"\centering")
    w(r"\scriptsize")
    w(r"\setlength{\tabcolsep}{2.2pt}")
    w(r"\renewcommand{\arraystretch}{1.02}")
    w(r"\adjustbox{max width=\textwidth}{%")
    w(r"\begin{tabular}{@{}c l | *{%d}{c} | *{%d}{c} @{}}" % (n1, n2))
    w(r"\toprule")
    w(r" & & \multicolumn{%d}{c|}{\textbf{Without native random access}} & \multicolumn{%d}{c}{\textbf{With native random access}} \\"
      % (n1, n2))
    w(r" & \textbf{Dataset} & " + " & ".join(r"\rotatebox{55}{%s}" % l for l in labels) + r" \\")
    for m in metrics:
        col, title, higher, scale = METRICS[m]
        w(r"\midrule")
        rows = []
        for d in datasets:
            vals = [float(data[n][d][col]) for n in names]
            rows.append((d, [scale(v) if scale else v for v in vals]))
        avg = [sum(r[1][i] for r in rows) / len(rows) for i in range(len(names))]
        rows.append((r"\textbf{Average}", avg))
        for k, (d, vals) in enumerate(rows):
            cells = [fmt(m, v) for v in vals]
            best = (lambda idx: max(idx, key=lambda i: vals[i])) if higher else (lambda idx: min(idx, key=lambda i: vals[i]))
            for idx in (range(n1), range(n1, n1 + n2)):  # bold: best within each group
                i = best(idx); cells[i] = r"\textbf{%s}" % cells[i]
            i = best(range(n1 + n2)); cells[i] = r"\underline{%s}" % cells[i]  # underlined: best overall
            lead = (r"\multirow{%d}{*}{\rotatebox{90}{\textbf{%s}}}" % (len(rows), title)) if k == 0 else ""
            w(f"{lead} & {d} & " + " & ".join(cells) + r" \\")
    w(r"\bottomrule")
    w(r"\end{tabular}}")
    w(r"\end{table*}")
    text = "\n".join(out) + "\n"
    if args.output:
        open(args.output, "w").write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
