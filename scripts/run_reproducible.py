#!/usr/bin/env python3
"""Reproducible, accountable benchmark runner for LosslessBenchmark(Full).

Why this exists: on multi-socket machines a single benchmark process per dataset
gives throughput numbers that depend on where the kernel happens to place its
memory (which NUMA node, how many huge pages), and on what else is running.
Run-to-run differences of 2-4x were observed on large datasets for the same
binary. This runner removes the uncontrolled factors it can and records the rest.

What it does (all codecs are treated identically):
  * One process per (dataset, compressor, repetition); repetitions interleave the
    compressors and rotate their order, so drift affects all codecs equally.
  * Pins each process to one CPU and binds its memory to that CPU's NUMA node
    (MPOL_BIND via set_mempolicy, inherited by the benchmark; no numactl needed).
    The node is chosen to have enough free memory; with binding the kernel cannot
    silently spill pages to a remote node.
  * Optionally disables transparent huge pages for the benchmark process
    (--thp off), removing dependence on memory fragmentation.
  * Serializes runs with an flock so concurrent benchmark sessions cannot overlap.
  * Records provenance: binary/dataset sha256, git revisions (+ dirty diff hash),
    compiler and flags from CMakeCache, CPU/kernel/THP settings, per-run CPU, node,
    node free memory, load average, wall time and exit status.
  * Reports per (dataset, compressor) the median of every numeric metric, plus
    min/max and coefficient of variation; metrics with CV above --cv-warn are
    flagged. results.csv (medians, same columns as the benchmark's CSV) is what
    generate_tables.sh / generate_plots.sh consume.

Linux only (x86_64 / aarch64 syscalls). Python >= 3.6, standard library only.

Example:
  scripts/run_reproducible.py --bin build/LosslessBenchmarkFull \\
      --datasets /path/to/datasets --compressors leco,b_star_gef_optimal --reps 5
"""
import argparse
import csv
import ctypes
import fcntl
import hashlib
import io
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time

SYS_SET_MEMPOLICY = {'x86_64': 238, 'aarch64': 237}
MPOL_BIND = 2
PR_SET_THP_DISABLE = 41
NODE_DIR = '/sys/devices/system/node'


# ---------------------------------------------------------------- topology ---

def parse_cpulist(text):
    cpus = []
    for part in text.strip().split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def numa_nodes():
    """{node: [cpus]}; a single pseudo-node when the system exposes no NUMA info."""
    nodes = {}
    if os.path.isdir(NODE_DIR):
        for name in os.listdir(NODE_DIR):
            if name.startswith('node') and name[4:].isdigit():
                with open(os.path.join(NODE_DIR, name, 'cpulist')) as f:
                    cpus = parse_cpulist(f.read())
                if cpus:
                    nodes[int(name[4:])] = cpus
    return nodes or {0: sorted(os.sched_getaffinity(0))}


def node_free_bytes(node):
    path = os.path.join(NODE_DIR, 'node%d' % node, 'meminfo')
    try:
        with open(path) as f:
            for line in f:
                if 'MemFree:' in line:
                    return int(line.split()[-2]) * 1024
    except OSError:
        pass
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    return 0


def cpu_node(cpu, nodes):
    for node, cpus in nodes.items():
        if cpu in cpus:
            return node
    raise SystemExit('CPU %d is not online' % cpu)


def node_busy_fraction(cpus, interval=1.0):
    """Mean busy fraction of the given CPUs over a short sample (all processes)."""
    busy = []
    def snap():
        out = {}
        with open('/proc/stat') as f:
            for line in f:
                parts = line.split()
                if parts[0].startswith('cpu') and parts[0][3:].isdigit() and int(parts[0][3:]) in cpus:
                    vals = list(map(int, parts[1:]))
                    out[int(parts[0][3:])] = (sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0))
        return out
    a = snap(); time.sleep(interval); b = snap()
    for c in cpus:
        if c in a and c in b and b[c][0] > a[c][0]:
            total = b[c][0] - a[c][0]
            busy.append(1.0 - (b[c][1] - a[c][1]) / total)
    return sum(busy) / len(busy) if busy else 0.0


def sibling_threads(cpu):
    path = '/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list' % cpu
    try:
        with open(path) as f:
            return parse_cpulist(f.read())
    except OSError:
        return [cpu]


# ---------------------------------------------------------- process setup ---

def make_preexec(cpu, node, bind, thp_off):
    libc = ctypes.CDLL(None, use_errno=True)
    nr = SYS_SET_MEMPOLICY.get(platform.machine())
    words = node // 64 + 1
    mask = (ctypes.c_ulong * words)()
    mask[node // 64] = 1 << (node % 64)

    def preexec():
        os.sched_setaffinity(0, {cpu})
        if bind:
            if nr is None or libc.syscall(nr, MPOL_BIND, mask, words * 64 + 1) != 0:
                os.write(2, b'run_reproducible: set_mempolicy(MPOL_BIND) failed\n')
                os._exit(97)
        if thp_off and libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
            os.write(2, b'run_reproducible: prctl(PR_SET_THP_DISABLE) failed\n')
            os._exit(98)
    return preexec


# ------------------------------------------------------------- provenance ---

def sha256_file(path, cache=None):
    st = os.stat(path)
    key = '%s:%d:%d' % (os.path.abspath(path), st.st_size, int(st.st_mtime))
    if cache is not None and key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    if cache is not None:
        cache[key] = h.hexdigest()
    return h.hexdigest()


def sh(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(path):
    if not os.path.isdir(path) or sh(['git', 'rev-parse', '--git-dir'], cwd=path) is None:
        return None
    diff = subprocess.run(['git', 'diff', 'HEAD'], cwd=path, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL).stdout
    untracked = sh(['git', 'ls-files', '--others', '--exclude-standard'], cwd=path) or ''
    return {'path': os.path.abspath(path),
            'commit': sh(['git', 'rev-parse', 'HEAD'], cwd=path),
            'branch': sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=path),
            'dirty': bool(diff) or bool(untracked),
            'diff_sha256': hashlib.sha256(diff).hexdigest() if diff else None,
            'untracked_files': len(untracked.splitlines())}


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cmake_cache(bin_path):
    cache = os.path.join(os.path.dirname(os.path.abspath(bin_path)), 'CMakeCache.txt')
    keep = ('CMAKE_CXX_COMPILER', 'CMAKE_BUILD_TYPE', 'CMAKE_CXX_FLAGS', 'ARCH_FLAGS', 'COMMON_OPT_FLAGS',
            'NEATS_WITH_LECO', 'BUILD_PORTABLE')
    out = {}
    if os.path.exists(cache):
        with open(cache) as f:
            for line in f:
                name = line.split(':', 1)[0]
                if name.startswith(keep) and '=' in line:
                    out[name] = line.split('=', 1)[1].strip()
        flags = os.path.join(os.path.dirname(cache), 'CMakeFiles',
                             os.path.basename(bin_path) + '.dir', 'flags.make')
        if os.path.exists(flags):
            with open(flags) as f:
                out['flags.make'] = [l.strip() for l in f if l.startswith(('CXX_FLAGS', 'CXX_DEFINES'))]
        detected = os.path.join(os.path.dirname(cache), 'CMakeFiles')
        for root, _, files in os.walk(detected):
            if 'CMakeCXXCompiler.cmake' in files:
                with open(os.path.join(root, 'CMakeCXXCompiler.cmake')) as f:
                    for line in f:
                        for key in ('CMAKE_CXX_COMPILER ', 'CMAKE_CXX_COMPILER_ID ', 'CMAKE_CXX_COMPILER_VERSION '):
                            if line.startswith('set(' + key):
                                out[key.strip()] = line.split(None, 1)[1].rstrip(')\n').strip('"')
                break
        compiler = out.get('CMAKE_CXX_COMPILER')
        if compiler and os.path.exists(compiler):
            version = sh([compiler, '--version'])
            out['compiler_version'] = version.splitlines()[0] if version else None
    return out


def machine_info():
    cpu_model = None
    with open('/proc/cpuinfo') as f:
        for line in f:
            if line.startswith('model name'):
                cpu_model = line.split(':', 1)[1].strip()
                break
    return {'hostname': socket.gethostname(), 'kernel': platform.release(), 'cpu_model': cpu_model,
            'online_cpus': os.cpu_count(),
            'thp_enabled': read('/sys/kernel/mm/transparent_hugepage/enabled'),
            'thp_defrag': read('/sys/kernel/mm/transparent_hugepage/defrag'),
            'governor_cpu0': read('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
            'no_turbo': read('/sys/devices/system/cpu/intel_pstate/no_turbo')}


def loadavg():
    return read('/proc/loadavg')


def cpu_busy(cpus, interval=0.5):
    """Fraction of time the given CPUs were busy over a short sample."""
    def snap():
        out = {}
        with open('/proc/stat') as f:
            for line in f:
                parts = line.split()
                if parts[0].startswith('cpu') and parts[0][3:].isdigit() and int(parts[0][3:]) in cpus:
                    vals = list(map(int, parts[1:]))
                    out[int(parts[0][3:])] = (sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0))
        return out
    a = snap(); time.sleep(interval); b = snap()
    busy = []
    for c in cpus:
        if c in a and c in b and b[c][0] > a[c][0]:
            total = b[c][0] - a[c][0]
            busy.append(1.0 - (b[c][1] - a[c][1]) / total)
    return max(busy) if busy else 0.0


# ---------------------------------------------------------------- results ---

def to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(rows, cv_warn):
    """Median row (same columns) + per-metric spread for one (dataset, compressor)."""
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    median_row, spread, flagged = {}, {}, []
    for key in columns:
        values = [row.get(key) for row in rows]
        numbers = [to_number(v) for v in values]
        if all(n is not None for n in numbers) and numbers:
            med = statistics.median(numbers)
            median_row[key] = ('%.4f' % med) if any('.' in str(v) for v in values) else str(int(round(med)))
            if len(numbers) > 1 and any(n != numbers[0] for n in numbers):
                mean = statistics.mean(numbers)
                cv = statistics.stdev(numbers) / mean if mean else 0.0
                spread[key] = {'min': min(numbers), 'max': max(numbers), 'cv': cv}
                if cv > cv_warn and key.endswith(('_mbs', '_ns')):
                    flagged.append(key)
        else:
            distinct = sorted(set(str(v) for v in values))
            median_row[key] = distinct[0] if len(distinct) == 1 else '|'.join(distinct)
    return columns, median_row, spread, flagged


# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bin', default='build/LosslessBenchmarkFull', help='benchmark executable')
    ap.add_argument('--datasets', required=True, nargs='+', help='.bin files and/or directories of .bin files')
    ap.add_argument('--compressors', required=True, help='comma-separated benchmark compressor names')
    ap.add_argument('--reps', type=int, default=5, help='processes per (dataset, compressor); default 5')
    ap.add_argument('--out', default=None, help='output directory (default benchmark_results/run-<timestamp>)')
    ap.add_argument('--cpu', type=int, default=None,
                    help='CPU to pin to (default: an idle CPU on the node with the most free memory)')
    ap.add_argument('--no-membind', action='store_true', help='do not bind memory to the CPU\'s NUMA node')
    ap.add_argument('--thp', choices=['system', 'off'], default='system',
                    help='transparent huge pages for the benchmark process (default: system setting)')
    ap.add_argument('--mem-factor', type=float, default=4.0,
                    help='required free node memory = factor x dataset file size (default 4)')
    ap.add_argument('--lock', default='/tmp/gef-bench.lock', help='flock file shared by all benchmark sessions')
    ap.add_argument('--cv-warn', type=float, default=0.05, help='flag throughput/latency metrics with CV above this')
    ap.add_argument('--max-node-busy', type=float, default=0.15,
                    help='before each run, wait until the pinned NUMA node is at most this busy (fraction of '
                         'its CPUs, other processes; default 0.15). The measured value is recorded per run.')
    ap.add_argument('--busy-timeout', type=float, default=3600,
                    help='give up waiting for a quiet node after this many seconds and run anyway, flagged')
    ap.add_argument('--seed', type=int, default=1, help='seed for the per-repetition compressor order')
    ap.add_argument('--extra-args', default='', help='extra arguments passed to the benchmark (e.g. "-r 10,100")')
    a = ap.parse_args()

    if not sys.platform.startswith('linux'):
        raise SystemExit('run_reproducible.py requires Linux (sched_setaffinity / set_mempolicy).')
    bin_path = os.path.abspath(a.bin)
    if not os.access(bin_path, os.X_OK):
        raise SystemExit('benchmark executable not found: %s' % bin_path)
    datasets = []
    for item in a.datasets:
        if os.path.isdir(item):
            datasets += sorted(os.path.join(item, f) for f in os.listdir(item) if f.endswith('.bin'))
        else:
            datasets.append(item)
    datasets = sorted(set(map(os.path.abspath, datasets)), key=os.path.getsize)
    if not datasets:
        raise SystemExit('no .bin datasets found')
    compressors = [c for c in a.compressors.split(',') if c]

    repo = (sh(['git', 'rev-parse', '--show-toplevel'], cwd=os.path.dirname(bin_path))
            or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.abspath(a.out or os.path.join(repo, 'benchmark_results', time.strftime('run-%Y%m%dT%H%M%S')))
    os.makedirs(out, exist_ok=True)

    nodes = numa_nodes()
    need = int(a.mem_factor * max(os.path.getsize(d) for d in datasets))
    if a.cpu is None:
        ranked = sorted(nodes, key=node_free_bytes, reverse=True)
        node = ranked[0]
        candidates = [c for c in nodes[node] if c in os.sched_getaffinity(0)]
        # Prefer a CPU whose hyperthread siblings are idle too.
        cpu = min(candidates[1:] or candidates, key=lambda c: cpu_busy(sibling_threads(c), 0.2))
    else:
        cpu = a.cpu
        node = cpu_node(cpu, nodes)
    bind = not a.no_membind and len(nodes) > 1
    if bind and node_free_bytes(node) < need:
        raise SystemExit('node %d has %.1f GB free, but the largest dataset needs ~%.1f GB (--mem-factor); '
                         'pick another --cpu, free memory, or pass --no-membind (not recommended).'
                         % (node, node_free_bytes(node) / 1e9, need / 1e9))

    cache_path = os.path.join(out, '.sha256-cache.json')
    sha_cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    meta = {
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'command': sys.argv,
        'binary': {'path': bin_path, 'sha256': sha256_file(bin_path)},
        'build': cmake_cache(bin_path),
        'repo': git_state(repo),
        'submodules': {name: git_state(os.path.join(repo, 'lib', name))
                       for name in sorted(os.listdir(os.path.join(repo, 'lib')))
                       if os.path.isdir(os.path.join(repo, 'lib', name))} if os.path.isdir(os.path.join(repo, 'lib')) else {},
        'machine': machine_info(),
        'placement': {'cpu': cpu, 'node': node, 'membind': bind, 'thp': a.thp,
                      'node_free_bytes_at_start': node_free_bytes(node), 'nodes': {str(k): v for k, v in nodes.items()}},
        'reps': a.reps, 'compressors': compressors, 'lock': a.lock, 'extra_args': a.extra_args,
        'datasets': [{'path': d, 'bytes': os.path.getsize(d), 'sha256': sha256_file(d, sha_cache)} for d in datasets],
    }
    json.dump(sha_cache, open(cache_path, 'w'))
    with open(os.path.join(out, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('output: %s\nplacement: cpu %d, node %d, membind=%s, thp=%s' % (out, cpu, node, bind, a.thp), flush=True)
    if meta['repo'] and meta['repo']['dirty']:
        print('warning: repository has uncommitted changes (diff hash recorded in metadata.json)', flush=True)

    runs_file = open(os.path.join(out, 'runs.jsonl'), 'a')
    rng = random.Random(a.seed)
    collected = {}
    failures = 0
    noisy = 0
    preexec = make_preexec(cpu, node, bind, a.thp == 'off')
    env = dict(os.environ, OMP_NUM_THREADS='1', LC_ALL='C')
    for rep in range(a.reps):
        for dataset in datasets:
            order = compressors[:]
            rng.shuffle(order)
            for comp in order:
                name = os.path.splitext(os.path.basename(dataset))[0]
                csv_path = os.path.join(out, 'raw', '%s.%s.rep%d.csv' % (name, comp, rep))
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                cmd = [bin_path, '-o', csv_path, '-c', comp] + a.extra_args.split() + [dataset]
                lock_fd = os.open(a.lock, os.O_RDWR | os.O_CREAT, 0o666)
                waited = time.time()
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                # Other users' jobs on the same node share its L3 and memory
                # bandwidth: wait for the node to be quiet, and record it.
                busy_wait = time.time()
                node_busy = node_busy_fraction(nodes[node])
                while node_busy > a.max_node_busy and time.time() - busy_wait < a.busy_timeout:
                    time.sleep(10)
                    node_busy = node_busy_fraction(nodes[node])
                record = {'rep': rep, 'dataset': name, 'compressor': comp, 'cpu': cpu, 'node': node,
                          'lock_wait_s': round(busy_wait - waited, 3),
                          'busy_wait_s': round(time.time() - busy_wait, 3),
                          'node_busy_before': round(node_busy, 3),
                          'node_busy_exceeded': node_busy > a.max_node_busy,
                          'node_free_bytes': node_free_bytes(node), 'loadavg_before': loadavg(),
                          'sibling_busy_before': round(cpu_busy(sibling_threads(cpu), 0.2), 3)}
                started = time.time()
                try:
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          preexec_fn=preexec, env=env)
                finally:
                    record['wall_s'] = round(time.time() - started, 3)
                    record['loadavg_after'] = loadavg()
                    os.close(lock_fd)
                record['returncode'] = proc.returncode
                if record['node_busy_exceeded']:
                    noisy += 1
                if proc.returncode == 0 and os.path.exists(csv_path):
                    with open(csv_path) as f:
                        rows = list(csv.DictReader(f))
                    for row in rows:
                        record['result'] = row
                        collected.setdefault((name, row.get('compressor', comp)), []).append(row)
                else:
                    failures += 1
                    record['stderr_tail'] = proc.stderr.decode(errors='replace')[-2000:]
                runs_file.write(json.dumps(record) + '\n')
                runs_file.flush()
                ra = record.get('result', {}).get('random_access_ns', '-')
                print('rep %d  %-12s %-24s rc=%d  ra_ns=%s  wall=%.1fs  load=%s'
                      % (rep, name, comp, proc.returncode, ra, record['wall_s'], record['loadavg_before'].split()[0]),
                      flush=True)

    # Aggregate: medians in benchmark CSV format + spread report.
    all_columns, medians, report = [], [], {}
    for (name, comp), rows in sorted(collected.items()):
        columns, med, spread, flagged = summarize(rows, a.cv_warn)
        for c in columns:
            if c not in all_columns:
                all_columns.append(c)
        med['repetitions'] = str(len(rows))
        medians.append(med)
        report['%s|%s' % (name, comp)] = {'n': len(rows), 'spread': spread, 'unstable_metrics': flagged}
    all_columns.append('repetitions')
    with open(os.path.join(out, 'results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_columns, restval='')
        w.writeheader()
        w.writerows(medians)
    meta['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    meta['failures'] = failures
    meta['runs_on_busy_node'] = noisy
    with open(os.path.join(out, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(report, f, indent=1)

    unstable = {k: v['unstable_metrics'] for k, v in report.items() if v['unstable_metrics']}
    print('\nresults.csv: median of %d repetitions per (dataset, compressor) -> %s' % (a.reps, out))
    if unstable:
        print('UNSTABLE (CV > %.0f%%):' % (100 * a.cv_warn))
        for key, metrics in sorted(unstable.items()):
            print('  %-40s %s' % (key, ', '.join(metrics)))
    if failures:
        print('%d benchmark process(es) failed; see runs.jsonl' % failures)
    if noisy:
        print('WARNING: %d run(s) started while the node was busier than --max-node-busy '
              '(node_busy_exceeded in runs.jsonl); treat their numbers with caution' % noisy)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
