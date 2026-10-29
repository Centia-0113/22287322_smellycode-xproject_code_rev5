"""Fig. 3 (revised): held-out permutation importance (drop in PR-AUC) per metric cluster and smell for XGBoost.
Bars = primary test scoring; dots = primary validation folds (mean ± SD); diamonds = post hoc granularity-matched
validation folds (mean ± SD), on the same four clusters for comparability. Small multiples with independent x-axes.
Usage: python fig3.py <results.json> <posthoc_results.json> <out.png>"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
R = json.load(open(sys.argv[1])); H = json.load(open(sys.argv[2])); out = sys.argv[3]; m = 'XGB'
CL = list(R['clusters']); SM = [('GC', 'God Class'), ('DC', 'Data Class'), ('LM', 'Long Method'), ('FE', 'Feature Envy')]
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.5,
                     'xtick.color': MUTED, 'ytick.color': INK, 'xtick.major.width': 0.5, 'ytick.major.width': 0, 'xtick.major.size': 2})
fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.85), sharey=True)
labels = ['Size and operands', 'Distinct operators', 'Operator effort', 'Cyclomatic complexity']
ypos = np.arange(len(CL))[::-1]
for ax, (k, name) in zip(axes.ravel(), SM):
    t = [R['perm_test'][f'{m}-{k}'][c] for c in CL]
    vm = np.array([R['perm_val'][f'{m}-{k}'][c]['mean'] for c in CL]); vs = np.array([R['perm_val'][f'{m}-{k}'][c]['sd'] for c in CL])
    hm = np.array([H['perm_val_matched_original_clusters'][f'{m}-{k}'][c]['mean'] for c in CL]); hs = np.array([H['perm_val_matched_original_clusters'][f'{m}-{k}'][c]['sd'] for c in CL])
    ax.barh(ypos, t, height=0.5, color=BLUE, zorder=2, label='Primary: test')
    ax.errorbar(vm, ypos + 0.17, xerr=vs, fmt='o', ms=2.4, color=INK, ecolor=INK, elinewidth=0.6, capsize=1.1, zorder=3, label='Primary: validation folds')
    ax.errorbar(hm, ypos - 0.17, xerr=hs, fmt='D', ms=2.4, color=ORANGE, ecolor=ORANGE, elinewidth=0.6, capsize=1.1, zorder=3,
                markeredgecolor='white', markeredgewidth=0.3, label='Post hoc matched: validation folds')
    ax.axvline(0, color=MUTED, lw=0.5, zorder=1)
    ax.grid(axis='x', color=GRID, lw=0.4, zorder=0); ax.set_axisbelow(True)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.set_title(name, fontsize=7, color=INK, pad=2, loc='left', fontweight='bold')
    ax.tick_params(axis='x', labelsize=5.5, pad=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    lo = min(0, (vm - vs).min(), (hm - hs).min(), min(t)); hi = max((vm + vs).max(), (hm + hs).max(), max(t))
    ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
axes[0, 0].set_yticks(ypos); axes[0, 0].set_yticklabels(labels); axes[1, 0].set_yticklabels(labels)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=5.4, bbox_to_anchor=(0.5, -0.01), handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
fig.text(0.62, 0.075, 'Drop in PR-AUC when the cluster is shuffled (axes differ by smell)', ha='center', fontsize=5.6, color=MUTED)
fig.tight_layout(pad=0.3, h_pad=0.9, w_pad=1.6, rect=(0, 0.1, 1, 1))
fig.savefig(out, dpi=400); print('saved', out)
