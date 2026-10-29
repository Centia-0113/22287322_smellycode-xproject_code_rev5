"""Post hoc step G: separates the two changes behind the granularity-matched results (Section VI-F).
The frozen PRIMARY models are re-scored, at their primary thresholds and without refitting, on the matched evaluation
populations (class-level test samples for God Class and Data Class, method-level test samples for Long Method and
Feature Envy). Comparing these scores with the primary scores (all test samples) isolates the change of evaluation
population; comparing them with the matched models' scores isolates the effect of training on matched samples.
Like steps C and F, this re-uses the previously examined test projects and is exploratory.
Usage: python population_check.py <work dir>"""
import os, sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
W = sys.argv[1]
df = load(W); lv = pd.read_pickle(W + '/level.pkl').values
tp = pickle.load(open(W + '/test_preds.pkl', 'rb')); te = tp['te_idx']
H = json.load(open(W + '/posthoc_results.json')); POP = H['populations']
out = {'populations': {}, 'scores': {}}
for s in LAB:
    m = lv[te] == POP[s]
    out['populations'][SHORT[s]] = {'primary_n': int(len(te)), 'matched_population': POP[s], 'matched_n': int(m.sum()),
                                    'positives': int(df[s].values[te].sum()), 'positives_in_matched': int(df[s].values[te][m].sum())}
for mod in ['RF', 'LR', 'XGB']:
    out['scores'][mod] = {}
    for s in LAB:
        m = lv[te] == POP[s]; thr = tp['sel'][(mod, s)]['thr']; y = df[s].values[te]; p = tp['probs'][(mod, s)]
        prim = scores(y, p, thr); prim_on_matched = scores(y[m], p[m], thr)
        matched = H['test_rescored'][mod]['per_smell'][SHORT[s]]
        out['scores'][mod][SHORT[s]] = {k: {'F1': round(float(v['F1']), 4), 'PRAUC': round(float(v['PRAUC']), 4), 'P': round(float(v['P']), 4), 'R': round(float(v['R']), 4)}
                                        for k, v in [('primary_model_all_test', prim), ('primary_model_matched_population', prim_on_matched),
                                                     ('matched_model_matched_population', matched)]}
print(json.dumps(out, indent=1))
json.dump(out, open(W + '/population_check.json', 'w'), indent=1)
