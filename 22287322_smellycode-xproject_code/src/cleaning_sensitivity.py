"""Post hoc step D: sensitivity of the results to the cleaning rule, on the TRAINING projects only (grouped 5-fold CV over
the same project folds; test projects untouched). XGBoost with the settings selected under each protocol.
Variants:  raw       - released file as is (duplicates and contradictory labels kept);
           majority  - one copy per code string, contradictory strings kept with their per-smell majority label (ties dropped);
           cleaned   - the adopted rule: contradictory strings removed, one copy of other duplicates kept.
Usage: python cleaning_sensitivity.py <raw CSV> <work dir>"""
import os, sys, json, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from granularity import level
RAW, W = sys.argv[1], sys.argv[2]; t0 = time.time()
man = pd.read_csv(W + '/split_manifest.csv').set_index('Project')
res = json.load(open(W + '/results.json')); pbest = pickle.load(open(W + '/posthoc_cv_best.pkl', 'rb'))
raw = pd.read_csv(RAW, low_memory=False, usecols=['Project', 'Code'] + FEAT + LAB)
raw['_n'] = raw['Code'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
raw = raw[~((raw['Logical Lines'] == 0) & (raw['_n'].str.len() > 0))].drop(columns=['Code']).reset_index(drop=True)
# majority-label variant, built on the whole file (de-duplication is dataset-wide), then restricted to training projects
mean = raw.groupby('_n')[LAB].mean()
tie = (mean == 0.5).any(axis=1)
maj = raw.drop_duplicates('_n', keep='first').set_index('_n')
maj = maj[~tie.reindex(maj.index).values]
maj[LAB] = (mean.reindex(maj.index)[LAB] > 0.5).astype(int).values
maj = maj.reset_index()
clean = load(W)[['Project', '_n'] + FEAT + LAB]
nun = raw.groupby('_n')[LAB].nunique().max(axis=1)
info = {'majority_ties_dropped': int(tie.sum()), 'contradictory_strings': int((nun > 1).sum())}
VAR = {'raw': raw, 'majority': maj, 'cleaned': clean}
POP = {'God class': 'class', 'Data class': 'class', 'Long method': 'method', 'Feature envy': 'method'}
def settings_primary(s):
    st = res['validation']['XGB']['settings'][SHORT[s]]; return st['params'], st['weighted']
def settings_matched(s):
    a, b = pbest[('XGB', s, False)], pbest[('XGB', s, True)]
    w = (b['F1'], b['PRAUC']) > (a['F1'], a['PRAUC']); return pbest[('XGB', s, w)]['params'], w
out = {'info': info}
for vname, d in VAR.items():
    d = d[d['Project'].map(man['partition']) == 'train'].reset_index(drop=True)
    fold = d['Project'].map(man['fold']).values; lv = d['_n'].map(level).values; X = d[FEAT].values
    out[vname] = {'n_train_samples': int(len(d))}
    for proto in ['primary', 'matched']:
        for s in LAB:
            msk = np.ones(len(d), bool) if proto == 'primary' else (lv == POP[s])
            params, w = settings_primary(s) if proto == 'primary' else settings_matched(s)
            y = d[s].values; P, Yv = [], []
            for k in range(1, 6):
                tri = np.where(msk & (fold != k))[0]; evi = np.where(msk & (fold == k))[0]
                est = make('XGB', params, w).fit(X[tri], y[tri]); P.append(est.predict_proba(X[evi])[:, 1]); Yv.append(y[evi])
            p, yy = np.concatenate(P), np.concatenate(Yv); thr, f1 = best_threshold(yy, p)
            out[vname][f'{proto}-{SHORT[s]}'] = {'n': int(len(yy)), 'positives': int(yy.sum()), 'prevalence': float(yy.mean()),
                                                 'PRAUC': float(average_precision_score(yy, p)), 'F1': float(f1)}
            print(vname, proto, SHORT[s], {k: round(v, 4) if isinstance(v, float) else v for k, v in out[vname][f'{proto}-{SHORT[s]}'].items()}, '[%.0fs]' % (time.time() - t0), flush=True)
json.dump(out, open(W + '/cleaning_sensitivity.json', 'w'), indent=1)
print(json.dumps(info), 'done in %.0fs' % (time.time() - t0))
