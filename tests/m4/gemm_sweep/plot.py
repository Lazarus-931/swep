#!/usr/bin/env python3
import csv, os, sys
try:
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('Agg')
except ImportError:
    print("pip3 install matplotlib"); sys.exit(1)

rows = list(csv.DictReader(open('gemm_results.csv')))
os.makedirs('plots', exist_ok=True)

kernels = sorted(set(r['kernel'] for r in rows))
colors = {'naive': '#FF9800', 'simd': '#4CAF50', 'tiled_16a': '#2196F3', 'tiled_24a': '#9C27B0'}

# square sizes only
sq = [r for r in rows if r['M'] == r['N'] == r['K']]
if sq:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for k in kernels:
        pts = sorted([(int(r['M']), float(r['gflops'])) for r in sq if r['kernel'] == k])
        if pts:
            ax1.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=k, color=colors.get(k, '#666'), linewidth=2)
            ax2.plot([p[0] for p in pts], [float(r['pct_peak']) for r in sorted([r for r in sq if r['kernel']==k], key=lambda r: int(r['M']))], 'o-', label=k, color=colors.get(k, '#666'), linewidth=2)
    ax1.set_xlabel('Matrix Size (N=M=K)'); ax1.set_ylabel('GFLOPS'); ax1.set_title('GEMM Throughput — Square Matrices')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_xscale('log', base=2)
    ax2.set_xlabel('Matrix Size (N=M=K)'); ax2.set_ylabel('% of Peak'); ax2.set_title('GEMM Efficiency — Square Matrices')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_xscale('log', base=2)
    fig.tight_layout(); fig.savefig('plots/square.png', dpi=150); plt.close()
    print('  plots/square.png')

# all shapes, best kernel comparison
fig, ax = plt.subplots(figsize=(14, 8))
shape_labels = sorted(set(f"{r['M']}x{r['N']}x{r['K']}" for r in rows), key=lambda s: max(int(x) for x in s.split('x')))
for k in kernels:
    gf = []
    labels_used = []
    for sl in shape_labels:
        m, n, kk = sl.split('x')
        r = [r for r in rows if r['kernel'] == k and r['M'] == m and r['N'] == n and r['K'] == kk]
        if r:
            gf.append(float(r[0]['gflops']))
            labels_used.append(sl)
    if gf:
        ax.barh([f"{sl}\n{k}" for sl in labels_used], gf, color=colors.get(k, '#666'), alpha=0.8, height=0.8)
ax.set_xlabel('GFLOPS'); ax.set_title('GEMM Throughput — All Shapes × All Kernels')
ax.grid(True, alpha=0.3, axis='x'); fig.tight_layout()
fig.savefig('plots/all_shapes.png', dpi=150); plt.close()
print('  plots/all_shapes.png')

# arithmetic intensity vs achieved GFLOPS
fig, ax = plt.subplots(figsize=(10, 6))
for k in kernels:
    pts = [(float(r['arith_intensity']), float(r['gflops'])) for r in rows if r['kernel'] == k]
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], label=k, color=colors.get(k, '#666'), s=60, alpha=0.7)
ax.set_xlabel('Arithmetic Intensity (FLOPS/byte)'); ax.set_ylabel('GFLOPS')
ax.set_title('Roofline — Arithmetic Intensity vs Throughput')
ax.axhline(y=5170, color='red', linestyle='--', alpha=0.4, label='Compute peak (5170 GFLOPS)')
ax.axline((0, 0), slope=102, color='blue', linestyle='--', alpha=0.4, label='Memory roof (102 GB/s)')
ax.set_xlim(0, max(float(r['arith_intensity']) for r in rows) * 1.1)
ax.set_ylim(0, max(float(r['gflops']) for r in rows) * 1.2)
ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
fig.savefig('plots/roofline.png', dpi=150); plt.close()
print('  plots/roofline.png')

print('\nDone.')
