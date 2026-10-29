"""Post hoc step B (exploratory): grouped 5-fold CV on the TRAINING projects with granularity-matched populations.
Identical to step 3 (same split, folds, grids, weighting options, out-of-fold threshold rule and near-duplicate
exclusions), except that God Class and Data Class use class-level samples only and Long Method and Feature Envy
method-level samples only. Usage: python posthoc_tune.py <work dir>"""
import os, sys, time, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
W = sys.argv[1]; t0 = time.time()
df = load(W); lv = pd.read_pickle(W + '/level.pkl').values
POP = {'God class': 'class', 'Data class': 'class', 'Long method': 'method', 'Feature envy': 'method'}
tr_mask = (df.partition == 'train').values
X = df[FEAT].values
folds = {}
for pop in ['class', 'method']:
    m = lv == pop; fl = []
    for k in range(1, 6):
        tri = np.where(tr_mask & m & (df.fold.values != k))[0]
        evi = eval_mask(df, np.where(m & (df.fold.values == k))[0], set(df.Project.values[tri]))
        fl.append((k, tri, evi))
    folds[pop] = fl
    print(pop, 'training samples per fold-train:', [len(t) for _, t, _ in fl], '| eval excluded as near-duplicates:',
          [int((m & (df.fold.values == k)).sum() - len(e)) for k, _, e in fl], flush=True)
rows, best = [], {}
for model in ['LR', 'XGB', 'RF']:
    for smell in LAB:
        y = df[smell].values; fl = folds[POP[smell]]
        for weighted in [False, True]:
            cur = None
            for ci, params in enumerate(GRIDS[model]):
                oof_idx, oof_p, oof_k = [], [], []
                for k, tri, evi in fl:
                    est = make(model, params, weighted).fit(X[tri], y[tri])
                    oof_idx.append(evi); oof_p.append(est.predict_proba(X[evi])[:, 1]); oof_k.append(np.full(len(evi), k))
                oi, op, ok = np.concatenate(oof_idx), np.concatenate(oof_p), np.concatenate(oof_k)
                thr, f1 = best_threshold(y[oi], op)
                ap = average_precision_score(y[oi], op)
                rows.append({'model': model, 'smell': smell, 'weighted': weighted, 'config': ci, 'params': json.dumps(params), 'oof_F1': f1, 'oof_PRAUC': ap, 'thr': thr})
                if cur is None or (f1, ap) > (cur['F1'], cur['PRAUC']):
                    cur = {'config': ci, 'params': params, 'F1': f1, 'PRAUC': ap, 'thr': thr, 'oof_idx': oi, 'oof_p': op, 'oof_k': ok}
            best[(model, smell, weighted)] = cur
            print(f"{model} {SHORT[smell]} weighted={weighted}: best {cur['params']} OOF F1={cur['F1']:.3f} PR-AUC={cur['PRAUC']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
pd.DataFrame(rows).to_csv(W + '/posthoc_cv_all_configs.csv', index=False)
pickle.dump(best, open(W + '/posthoc_cv_best.pkl', 'wb'))
print('done in %.0fs' % (time.time() - t0))
