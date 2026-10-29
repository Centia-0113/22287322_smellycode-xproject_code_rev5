"""Bootstrap details for the paper (post hoc step F): how often a project-level resample lacked positives for a smell,
how often a model predicted no positives (precision undefined, set to 0), and paired bootstrap differences between models
in macro-F1 and macro PR-AUC (and in per-smell F1), using exactly the resamples of steps 4 and C (same seed and order).
Also counts the distinct code strings behind the removed Long Method and Feature Envy labels.
Usage: python bootstrap_details.py <work dir> <raw CSV>"""
import os, sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
W, RAW = sys.argv[1], sys.argv[2]
df = load(W); MODELS = ['RF', 'LR', 'XGB']; out = {}
def run(Y, P, THR, PRJ_of, projs, pidx_of):
    rng = np.random.default_rng(SEED)
    zero_pos = {s: 0 for s in LAB}; zero_pred = {f'{m}-{SHORT[s]}': 0 for m in MODELS for s in LAB}
    mac = {m: {'F1': [], 'AP': []} for m in MODELS}
    per = {(m, s): [] for m in MODELS for s in LAB}   # per-smell F1 in each resample (for paired per-smell differences)
    for b in range(1000):
        pick = rng.choice(projs, size=len(projs), replace=True)
        f1s = {m: [] for m in MODELS}; aps = {m: [] for m in MODELS}
        for s in LAB:
            ii = np.concatenate([pidx_of(s)[p] for p in pick]); y = Y[s][ii]
            if y.sum() == 0: zero_pos[s] += 1; continue
            for m in MODELS:
                sc = scores(y, P[(m, s)][ii], THR[(m, s)])
                if (P[(m, s)][ii] >= THR[(m, s)]).sum() == 0: zero_pred[f'{m}-{SHORT[s]}'] += 1
                f1s[m].append(sc['F1']); aps[m].append(sc['PRAUC']); per[(m, s)].append(sc['F1'])
        for m in MODELS: mac[m]['F1'].append(np.mean(f1s[m])); mac[m]['AP'].append(np.mean(aps[m]))
    diffs = {}
    for a, b in [('RF', 'XGB'), ('XGB', 'LR'), ('RF', 'LR')]:
        for k in ['F1', 'AP']:
            d = np.array(mac[a][k]) - np.array(mac[b][k])
            diffs[f'{a}-{b} macro{k}'] = {'mean': float(d.mean()), 'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'share_positive': float((d > 0).mean())}
    per_smell = {}
    for a, b in [('RF', 'XGB'), ('XGB', 'LR'), ('RF', 'LR')]:
        for s in LAB:
            d = np.array(per[(a, s)]) - np.array(per[(b, s)])
            per_smell[f'{a}-{b} F1_{SHORT[s]}'] = {'mean': float(d.mean()), 'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'share_positive': float((d > 0).mean())}
    return {'paired_per_smell_F1': per_smell, 'resamples_without_positives': {SHORT[s]: v for s, v in zero_pos.items()}, 'model_smell_resamples_without_predicted_positives': zero_pred, 'paired_differences': diffs}
# primary protocol (step 4)
tp = pickle.load(open(W + '/test_preds.pkl', 'rb')); te = tp['te_idx']
Y = {s: df[s].values[te] for s in LAB}; proj = df.Project.values[te]; projs = np.unique(proj)
pidx = {p: np.where(proj == p)[0] for p in projs}
THR = {k: v['thr'] for k, v in tp['sel'].items()}
out['primary'] = run(Y, tp['probs'], THR, None, projs, lambda s: pidx)
# post hoc (step C)
hp = pickle.load(open(W + '/posthoc_test_preds.pkl', 'rb')); H = json.load(open(W + '/posthoc_results.json'))
POP = H['populations']; te2 = hp['te_idx']
Y2 = {s: df[s].values[te2[POP[s]]] for s in LAB}; PRJ = {pop: df.Project.values[te2[pop]] for pop in te2}
projs2 = np.unique(np.concatenate([PRJ['class'], PRJ['method']]))
pidx2 = {pop: {p: np.where(PRJ[pop] == p)[0] for p in projs2} for pop in PRJ}
THR2 = {(m, s): H['validation'][m]['settings'][SHORT[s]]['thr'] for m in MODELS for s in LAB}
# thresholds in posthoc_results.json are rounded to 3 dp; re-derive exact ones from the tuning output
best = pickle.load(open(W + '/posthoc_cv_best.pkl', 'rb'))
for m in MODELS:
    for s in LAB:
        a, b = best[(m, s, False)], best[(m, s, True)]
        w = (b['F1'], b['PRAUC']) > (a['F1'], a['PRAUC']); THR2[(m, s)] = best[(m, s, w)]['thr']
out['posthoc'] = run(Y2, hp['probs'], THR2, None, projs2, lambda s: pidx2[POP[s]])
# distinct strings behind the removed method-level labels
raw = pd.read_csv(RAW, low_memory=False, usecols=['Project', 'Code', 'Logical Lines'] + LAB)
raw['_n'] = raw['Code'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
raw = raw[~((raw['Logical Lines'] == 0) & (raw['_n'].str.len() > 0))]
nun = raw.groupby('_n')[LAB].nunique().max(axis=1); conf = set(nun[nun > 1].index)
c = raw[raw['_n'].isin(conf)]
for s in ['Long method', 'Feature envy']:
    g = c.groupby('_n')[s].agg(['sum', 'size'])
    out[f'contradictory_{SHORT[s]}'] = {'positive_labels': int(g['sum'].sum()), 'distinct_strings_with_a_positive_copy': int((g['sum'] > 0).sum()),
                                         'median_copies_of_those_strings': float(g.loc[g['sum'] > 0, 'size'].median())}
d = raw[~raw['_n'].isin(conf)]; dup = d.duplicated('_n', keep='first')
kept = d[~dup].set_index('_n')
for s in ['Long method', 'Feature envy']:
    removed = d[dup & (d[s] == 1)]
    out[f'dedup_removed_{SHORT[s]}'] = {'labels': int(len(removed)), 'all_copies_of_kept_positive_strings': bool((kept.loc[removed['_n'].unique(), s] == 1).all())}
print(json.dumps(out, indent=1))
json.dump(out, open(W + '/bootstrap_details.json', 'w'), indent=1)
