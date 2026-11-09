#!/usr/bin/env python3
"""Range-query campaign driver (methodology: docs/range-queries.md).

Runs LosslessBenchmark once per (dataset, codec) so that every case is an
isolated, resumable process, then aggregates the per-trial output with
plot_range_queries.py (per-dataset CSV, summary CSV, plot in decimal MB/s).

Example (NeaTS paper, Fig. 4 setup, 11 largest datasets, pinned to CPU 4):
  scripts/run_range_queries.py --binary build/LosslessBenchmark \\
      --dataset-dir datasets --largest 11 \\
      --codecs neats,dac,rle_gef,b_star_gef_optimal,alp \\
      --trials 3 --taskset 4 --out results/range-queries
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

DEFAULT_SIZES = [10 * 2 ** k for k in range(17)]  # 10 .. 655,360 (NeaTS Fig. 4)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", required=True, help="LosslessBenchmark / LosslessBenchmarkFull executable")
    src = p.add_argument_group("datasets (paths and/or a directory)")
    src.add_argument("datasets", nargs="*", help="dataset .bin files")
    src.add_argument("--dataset-dir", help="directory of .bin datasets")
    src.add_argument("--names", help="comma-separated dataset stems to take from --dataset-dir")
    src.add_argument("--largest", type=int, help="take the N largest .bin files of --dataset-dir")
    p.add_argument("--codecs", required=True, help="comma-separated codec names, as accepted by -c")
    p.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)), help="range sizes (default: 10*2^k, k=0..16)")
    p.add_argument("--queries", type=int, default=10000, help="queries per size (default: 10000)")
    p.add_argument("--seed", type=int, default=1234, help="seed of the shared starts (default: 1234)")
    p.add_argument("--trials", type=int, default=3, help="timed trials per size (default: 3)")
    p.add_argument("--batch-bytes", type=int, help="output arena per timed batch (binary default if omitted)")
    p.add_argument("--no-verify", action="store_true", help="skip the untimed per-query check (not recommended)")
    p.add_argument("--extra-args", default="", help="extra arguments passed verbatim to the binary (e.g. '-b 1000')")
    p.add_argument("--taskset", help="pin every case with `taskset -c CPU` (e.g. 4)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--rerun", action="store_true", help="ignore completed cases and run them again")
    p.add_argument("--stop-on-error", action="store_true", help="stop at the first failing case")
    p.add_argument("--no-plot", action="store_true", help="only run cases, do not aggregate/plot")
    return p.parse_args()


def select_datasets(args):
    paths = [Path(d) for d in args.datasets]
    if args.dataset_dir:
        pool = sorted(Path(args.dataset_dir).glob("*.bin"))
        if args.names:
            wanted = args.names.split(",")
            by_stem = {p.stem: p for p in pool}
            missing = [w for w in wanted if w not in by_stem]
            if missing:
                sys.exit(f"datasets not found in {args.dataset_dir}: {', '.join(missing)}")
            pool = [by_stem[w] for w in wanted]
        if args.largest:
            pool = sorted(pool, key=lambda p: p.stat().st_size, reverse=True)[: args.largest]
        paths += pool
    if not paths:
        sys.exit("no datasets selected")
    for p in paths:
        if not p.is_file():
            sys.exit(f"missing dataset: {p}")
    return paths


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def case_ok(csv_path, trials_path, sizes, trials):
    """A case is complete when both CSVs exist, the main CSV has one row and
    the trials CSV has one record per (size < n, trial)."""
    try:
        rows = list(csv.DictReader(open(csv_path)))
        trial_rows = list(csv.DictReader(open(trials_path)))
    except OSError:
        return False
    if len(rows) != 1:
        return False
    n = int(rows[0]["num_values"])
    expected = sum(1 for s in sizes if 0 < s < n) * trials
    return len(trial_rows) == expected


def main():
    args = parse_args()
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        sys.exit(f"missing binary: {binary}")
    datasets = select_datasets(args)
    codecs = [c for c in args.codecs.split(",") if c]
    sizes = [int(s) for s in args.sizes.split(",") if s]
    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    metadata = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "datasets": [str(p) for p in datasets],
        "codecs": codecs,
        "range_sizes": sizes,
        "queries_per_size": args.queries,
        "seed": args.seed,
        "trials": args.trials,
        "batch_bytes": args.batch_bytes,
        "verify": not args.no_verify,
        "extra_args": args.extra_args,
        "taskset": args.taskset,
        "aggregation": "median over trials per (codec, dataset, size); arithmetic mean over datasets",
        "units": "decimal MB/s = queries * size * 8 bytes / 1e6 / seconds (requested output values only)",
    }
    try:
        repo = Path(__file__).resolve().parent.parent
        metadata["git_commit"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    env = dict(os.environ, OMP_NUM_THREADS="1", LC_ALL="C")
    failures = []
    for dataset in datasets:
        for codec in codecs:
            tag = f"{dataset.stem}__{codec}"
            csv_path, trials_path, log_path = raw / f"{tag}.csv", raw / f"{tag}.trials.csv", raw / f"{tag}.log"
            if not args.rerun and case_ok(csv_path, trials_path, sizes, args.trials):
                print(f"[skip] {tag} (complete)")
                continue
            cmd = [str(binary), "-c", codec, "-r", ",".join(map(str, sizes)),
                   "--range-queries", str(args.queries), "--range-seed", str(args.seed),
                   "--range-trials", str(args.trials), "--range-trials-csv", str(trials_path),
                   "-o", str(csv_path)]
            if args.batch_bytes:
                cmd += ["--range-batch-bytes", str(args.batch_bytes)]
            if args.no_verify:
                cmd.append("--no-range-verify")
            cmd += args.extra_args.split()
            cmd.append(str(dataset))
            if args.taskset:
                cmd = ["taskset", "-c", args.taskset] + cmd
            print(f"[run ] {tag}", flush=True)
            with open(log_path, "w") as log:
                log.write(" ".join(cmd) + "\n")
                log.flush()
                rc = subprocess.call(cmd, stdout=log, stderr=log, env=env)
            problem = None
            if rc != 0:
                problem = f"exit {rc}"
            elif not case_ok(csv_path, trials_path, sizes, args.trials):
                problem = "incomplete output"
            else:
                status = next(csv.DictReader(open(csv_path)))["validation_status"]
                if status in ("failed", "error", "bitwise_failed"):
                    problem = f"validation_status={status}"
            if problem:
                failures.append(f"{tag}: {problem} (see {log_path})")
                print(f"[FAIL] {tag}: {problem}", flush=True)
                if args.stop_on_error:
                    break
        if failures and args.stop_on_error:
            break

    (out / "failures.txt").write_text("".join(f + "\n" for f in failures))
    if not args.no_plot:
        plot = Path(__file__).resolve().parent / "plot_range_queries.py"
        subprocess.check_call([sys.executable, str(plot), str(out), "--codecs", ",".join(codecs)])
    if failures:
        print(f"{len(failures)} failing case(s); see {out / 'failures.txt'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
