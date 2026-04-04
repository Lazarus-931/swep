#!/usr/bin/env python3
import csv, os, sys
try:
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('Agg')
except ImportError:
    print("pip3 install matplotlib"); sys.exit(1)

rows = list(csv.DictReader(open('results.csv')))
os.makedirs('plots', exist_ok=True)

labels = [f"{r['M']}x{r['N']}x{r['K']}" for r in rows]
naive = [float(r['naive_gflops']) for r in rows]
best = [float(r['best_gflops']) for r in rows]
speedup = [float(r['speedup']) for r in rows]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

x = range(len(labels))
w = 0.35
ax1.bar([i - w/2 for i in x], naive, w, label='naive', color='#FF9800')
ax1.bar([i + w/2 for i in x], best, w, label='best', color='#2196F3')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=70, ha='right', fontsize=8)
ax1.set_ylabel('GFLOPS', fontsize=13)
ax1.set_title('GEMM Throughput: best vs naive — M4', fontsize=15)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(x, speedup, color='#4CAF50')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=70, ha='right', fontsize=8)
ax2.set_ylabel('Speedup (x)', fontsize=13)
ax2.set_title('Speedup: best / naive', fontsize=15)
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig('plots/best_vs_naive.png', dpi=150)
plt.close()
print('plots/best_vs_naive.png')
