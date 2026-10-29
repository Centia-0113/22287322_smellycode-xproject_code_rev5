"""Sensitivity check, training projects only (test projects untouched): PR-AUC of the selected configuration under
(a) a random sample-level 80/20 split of the RAW training-project data (duplicates and contradictory labels kept),
(b) the same random split of the CLEANED data, versus (c) grouped cross-validation (pooled out-of-fold PR-AUC)."""
import os, sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import train_test_split
W, RAW = sys.argv[1], sys.argv[2]
df = load(W); res = json.load(open(W + '/results.json')); best = pickle.load(open(W + '/cv_best.pkl', 'rb'))
train_projects = set(df.loc[df.partition == 'train', 'Project'])
raw = pd.read_csv(RAW, low_memory=False, usecols=['Project'] + FEAT + LAB)
raw = raw[raw['Project'].isin(train_projects)]
clean = df[df.partition == 'train']
out = {}
for m in ['RF', 'LR', 'XGB']:
    for s in LAB:
        st = res['validation'][m]['settings'][SHORT[s]]
        row = {}
        for name, d in [('random_raw', raw), ('random_clean', clean)]:
            Xa, ya = d[FEAT].values, d[s].values
            Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.2, random_state=SEED, stratify=ya)
            est = make(m, st['params'], st['weighted']).fit(Xtr, ytr)
            row[name] = float(average_precision_score(yte, est.predict_proba(Xte)[:, 1]))
        b = best[(m, s, st['weighted'])]; row['grouped_cv'] = float(b['PRAUC'])
        row['prevalence_raw'] = float(raw[s].mean()); row['prevalence_clean'] = float(clean[s].mean())
        out[f'{m}-{SHORT[s]}'] = row; print(m, SHORT[s], {k: round(v, 3) for k, v in row.items()}, flush=True)
json.dump(out, open(W + '/contrast.json', 'w'), indent=1)
