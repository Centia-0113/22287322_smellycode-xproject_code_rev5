"""Metric clusters for permutation importance, formed on TRAINING projects only (run after step 2, before step 4).
Average-linkage clustering of the 14 metrics on 1 - |Spearman rho|; cutting at distance 0.05 (|rho| >= 0.95) gives the
four clusters hard-coded as CL in step4_test.py: size/operands, distinct operators, operator effort, cyclomatic complexity."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
df = load(sys.argv[1]); tr = df[df.partition == 'train']
rho = tr[FEAT].rank().corr()
pd.set_option('display.width', 250)
print(rho.round(2).to_string())
D = 1 - rho.abs().values; np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), 'average')
for cut in [0.05, 0.1, 0.15, 0.2, 0.3]:
    lab = fcluster(Z, t=cut, criterion='distance')
    cl = {}
    for f, l in zip(FEAT, lab): cl.setdefault(l, []).append(f)
    print(f'\ncut {cut} (|rho| >= {1-cut:.2f}): {len(cl)} clusters'); [print('  ', v) for v in cl.values()]
