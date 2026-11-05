"""Compare frozen competitor rows with the two verified retained GEF runs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
GEF_REVISION = '07d4e4aec2d6dc12aea27d2b753b5785b10a92f1'
GEF_FINGERPRINT = '8e4d13390d90db7186dbf5b8104a160699aa92ddf94fefbcdb05af0eb204db2a'
GEF_INPUTS = ['gef-07d4e4a-r9.csv', 'gef-07d4e4a-r10.csv']
spec = importlib.util.spec_from_file_location('styles', ROOT.parents[2] / 'scripts/pareto_plots.py')
styles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(styles)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
c = pd.read_csv(ROOT / 'competitors-snapshot.csv')
gef_runs = []
for filename in GEF_INPUTS:
    run = pd.read_csv(ROOT / filename)
    assert set(run.source_fingerprint) == {GEF_FINGERPRINT}
    assert set(run.validation_status) == {'passed'}
    assert set(run.partition_size) == {32000}
    assert set(run.queries) == {1000000} and set(run.seed) == {42}
    run = run[run.repetition == 'median'].copy()
    assert len(run) == 32 and run.dataset.nunique() == 16
    assert not run.duplicated(['dataset', 'codec']).any()
    run['measurement_run'] = filename
    # Average rates themselves, preserving the verified cross-run convention.
    run['random_access_mvalues_s'] = 1000 / run.random_access_ns
    gef_runs.append(run)
g = pd.concat(gef_runs, ignore_index=True)
identity = ['num_values', 'original_num_values', 'decimals', 'partition_size',
            'queries', 'seed', 'input_fingerprint', 'compressed_bytes',
            'theoretical_bytes', 'allocated_bytes', 'checksum']
grouped = g.groupby(['dataset', 'codec'])
assert (grouped.size() == 2).all()
assert (grouped[identity].nunique() == 1).all().all()
means = ['compression_mib_s', 'decompression_mib_s', 'random_access_ns',
         'random_access_mvalues_s']
g = grouped.agg({**{column: 'first' for column in identity},
                 **{column: 'mean' for column in means}}).reset_index()
g['validation_status'] = 'passed'
# The old input is retained as history and checks matching values and sizes.
old = pd.read_csv(ROOT / 'gef-942.csv')
old = old[old.repetition == 'median'].set_index(['dataset', 'codec'])
for key, row in g.set_index(['dataset', 'codec']).iterrows():
    assert all(row[column] == old.loc[key, column] for column in identity), key
assert not c.duplicated(['dataset', 'requested_compressor']).any()
c = c[c.job_status == 'completed'].copy()
inventory = set(c.requested_compressor)
assert len(inventory) == 25
datasets = sorted(d for d, rows in c.groupby('dataset') if set(rows.requested_compressor) == inventory)
c = c[c.dataset.isin(datasets)].copy()
gef_available_datasets = sorted(g.dataset.unique())
g = g[g.dataset.isin(datasets)].copy()
assert set(g.partition_size) == {32000}
assert len(g) == 2 * len(datasets)
for d in datasets:
    assert set(c[c.dataset == d].num_values) == set(g[g.dataset == d].num_values)
    assert set(g[g.dataset == d].num_values) == set(g[g.dataset == d].original_num_values)
assert (c.uncompressed_bits == c.num_values * 64).all()
c['label'] = c.requested_compressor.map(styles.NAME_MAPPING)
c['ratio_pct'] = c.compressed_bits / c.uncompressed_bits * 100
c['qualification'] = np.where(c.input_conversion == 'float32', 'float32 conversion',
    np.where(c.validation_status == 'precision_bounded', 'bounded precision', 'none'))
c.loc[c.audit_status.isin(['sample_bitwise_failed', 'full_bitwise_failed']), 'qualification'] = 'bitwise audit failed'
c['compression_mib_s'] = c.compression_throughput_mbs
c['decompression_mib_s'] = c.decompression_throughput_mbs
c['harness'] = 'LosslessBenchmark family; single measurement; ' + c.measurement_run
g['label'] = g.codec.map({'RLE_GEF': 'RLE-GEF', 'B_STAR_GEF_APPROXIMATE': 'B*-GEF (Approximate)'})
g['ratio_pct'] = g.compressed_bytes / (g.num_values * 8) * 100
g['qualification'] = 'none'
g['harness'] = 'GefTuning; mean of two run medians (three repetitions/run); warmed access'
g['input_conversion'] = 'none'
g['input_changed_values'] = 0
g['owned_bytes'] = g.allocated_bytes
c['random_access_mvalues_s'] = 1000 / c.random_access_ns
columns = ['dataset', 'label', 'num_values', 'ratio_pct', 'compression_mib_s',
           'decompression_mib_s', 'random_access_ns', 'qualification', 'validation_status',
           'input_conversion', 'input_changed_values', 'owned_bytes', 'harness',
           'random_access_mvalues_s']
# Falcon excluded by user; retain historical raw rows only.
c = c[c.requested_compressor != 'falcon'].copy()
data = pd.concat([c[columns], g[columns]], ignore_index=True)
assert data.label.notna().all()
assert (data.groupby('label').size() == len(datasets)).all()
metrics = ['ratio_pct', 'compression_mib_s', 'decompression_mib_s', 'random_access_ns', 'random_access_mvalues_s']
assert np.isfinite(data[metrics]).all().all() and (data[metrics] > 0).all().all()
data.to_csv(ROOT / 'matched-per-dataset.csv', index=False)

def aggregate(method):
    rows = []
    for label, group in data.groupby('label'):
        qualification = ('bitwise audit failed' if 'bitwise audit failed' in set(group.qualification) else group.qualification.iloc[0])
        row = dict(label=label, datasets=len(group), qualification=qualification)
        for metric in metrics:
            row[metric] = group[metric].mean() if method == 'arithmetic' else np.exp(np.log(group[metric]).mean())
        row['corpus_ratio_pct'] = np.average(group.ratio_pct, weights=group.num_values)
        rows.append(row)
    return pd.DataFrame(rows)

def frontier(frame, metric):
    eligible = frame[frame.qualification == 'none']
    return eligible[[not (((eligible.ratio_pct <= r.ratio_pct) & (eligible[metric] >= r[metric])) &
        ((eligible.ratio_pct < r.ratio_pct) | (eligible[metric] > r[metric]))).any() for _, r in eligible.iterrows()]].sort_values('ratio_pct')

panels = [('compression_mib_s', 'Compression (MiB/s)'),
          ('decompression_mib_s', 'Decompression (MiB/s)'),
          ('random_access_mvalues_s', 'Random access (million values/s)')]

def plot(frame, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.4))
    handles = []
    for i, (ax, (metric, ylabel)) in enumerate(zip(axes, panels)):
        f = frontier(frame, metric)
        ax.plot(f.ratio_pct, f[metric], '--', color='#999999', lw=1, zorder=1)
        for _, row in frame.iterrows():
            st = styles.STYLES[row.label]
            qualified = row.qualification != 'none'
            gef = 'GEF' in row.label
            label = row.label.replace(' (Approximate)', ' approx.')
            if row.label == 'BUFF': label += ' (plain)'
            if row.qualification == 'bitwise audit failed': label += ' ‡'
            elif qualified: label += ' †'
            if gef: label += ' [07d4e4a]'
            h = ax.scatter(row.ratio_pct, row[metric], marker=st['m'],
                s=135 if gef else 65, facecolors='none' if qualified else st['c'],
                edgecolors=st['c'] if qualified else '#333333', linewidths=1 if qualified else .5,
                label=label, zorder=4 if gef else 3)
            if i == 0: handles.append(h)
        ax.set_yscale('log')
        ax.set_xlabel('Compressed / original size (%) — lower is better')
        ax.set_ylabel(ylabel + ' — higher is better')
        ax.grid(True, which='major', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=15, y=.98)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .07), ncol=7, fontsize=9, frameon=False)
    fig.text(.5, .012, 'GEF 07d4e4a: two independent full runs, three repetitions/run; mean of per-run median metrics.\n'
             'Preliminary: different timing harnesses. ALP/Gorilla/Camel rerun 2026-11-05; original Java Camel, warmed JVM.\n'
             '† BUFF (plain): bounded precision; Fauna/MacaqueV: float32. ‡ Original Camel: not lossless on some inputs. Hollow points excluded; Falcon omitted.',
             ha='center', fontsize=8.5)
    fig.subplots_adjust(left=.055, right=.99, top=.90, bottom=.30, wspace=.29)
    return fig

frontiers = {}
for method in ['arithmetic', 'geometric']:
    table = aggregate(method)
    for metric, _ in panels:
        names = set(frontier(table, metric).label)
        table['frontier_' + metric] = table.label.isin(names)
        frontiers[method + '_' + metric] = sorted(names)
    table.to_csv(ROOT / ('summary-' + method + '.csv'), index=False)
    fig = plot(table, f'Optimized GEF (07d4e4a) vs competitors • {len(datasets)} common full datasets • {method} means')
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(ROOT / ('pareto-' + method + '.' + ext), dpi=170)
    plt.close(fig)
with PdfPages(ROOT / 'pareto-per-dataset.pdf') as pdf:
    for d in datasets:
        fig = plot(data[data.dataset == d].sort_values('label'), f'{d} • optimized GEF 07d4e4a (uniform K=32,000) vs competitors')
        pdf.savefig(fig)
        plt.close(fig)
manifest = dict(datasets=datasets, n_datasets=len(datasets), competitor_rows=len(c), gef_rows=len(g),
    gef_revision=GEF_REVISION, gef_source_fingerprint=GEF_FINGERPRINT,
    gef_runs=GEF_INPUTS, gef_available_datasets=gef_available_datasets,
    gef_excluded_datasets=sorted(set(gef_available_datasets) - set(datasets)),
    gef_aggregation='Arithmetic mean of the two per-run median metrics for each dataset; access rates averaged independently of latency.',
    frontier=frontiers,
    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'competitors-snapshot.csv', ROOT/'gef-942.csv', ROOT/'alp-corrected-full16.csv', ROOT/'gorilla-camel-original-full16.csv', *(ROOT / name for name in GEF_INPUTS), Path(__file__)]})
(ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
