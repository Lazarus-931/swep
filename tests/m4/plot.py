#!/usr/bin/env python3
import csv, os, sys
try:
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('Agg')
except ImportError:
    print("pip3 install matplotlib"); sys.exit(1)

rows = list(csv.DictReader(open('results.csv')))
os.makedirs('plots', exist_ok=True)

def get(test):
    return [(r['param'], float(r['value'])) for r in rows if r['test'] == test]

def line(data, xlabel, ylabel, title, fname, peak=None, xlog=False):
    x = [float(d[0]) for d in data]
    y = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'o-', linewidth=2, markersize=7, color='#2196F3')
    if peak: ax.axhline(y=peak, color='red', linestyle='--', alpha=0.5, label=f'Peak ({peak})')
    if xlog: ax.set_xscale('log', base=2)
    ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14); ax.grid(True, alpha=0.3)
    if peak: ax.legend(fontsize=11)
    fig.tight_layout(); fig.savefig(f'plots/{fname}', dpi=150); plt.close()
    print(f'  plots/{fname}')

d = get('bw_saturation')
if d: line(d, 'Threads', 'GB/s', 'Bandwidth Saturation', 'bw_saturation.png', peak=102, xlog=True)

d = get('cache_under_pressure')
if d: line(d, 'Working Set (KB)', 'GB/s', 'Cache Boundaries Under Pressure', 'cache_pressure.png', xlog=True)

d = get('core_scaling')
if d: line(d, 'Threadgroups (1 per core)', 'GB/s', 'Per-Core Bandwidth Scaling', 'core_scaling.png', peak=102)

for ws in ['L1(16K)', 'DRAM(32M)']:
    d = get(f'lat_under_load_{ws}')
    if d: line(d, 'Bandwidth Threads', 'Latency (ns)', f'Latency Under Load — {ws}', f'lat_load_{ws.replace("(","").replace(")","")}.png')

d = get('atomic_contention')
if d: line(d, 'Counter Count', 'Gops/s', 'Atomic Throughput vs Contention', 'atomics.png')

rw = {r['param']: float(r['value']) for r in rows if r['test'] == 'rw_asymmetry'}
if rw:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(rw.keys()), list(rw.values()), color=['#4CAF50', '#FF9800', '#2196F3'])
    ax.set_ylabel('GB/s', fontsize=13); ax.set_title('Read / Write / Copy Bandwidth', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y'); fig.tight_layout()
    fig.savefig('plots/rw_asymmetry.png', dpi=150); plt.close()
    print('  plots/rw_asymmetry.png')

print('\nDone.')
