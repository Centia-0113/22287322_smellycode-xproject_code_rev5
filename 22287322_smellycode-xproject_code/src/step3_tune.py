"""Step 3: grouped 5-fold CV on training projects only. Per model, smell and weighting setting: grid search on pooled
out-of-fold (OOF) F1 at the OOF-optimal threshold; then the weighting setting is chosen per model and smell."""
import os, sys, time, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
W = sys.argv[1]; t0 = time.time()
df = load(W)
tr_mask = (df.partition == 'train').values
folds = []
for k in range(1, 6):
    tr_idx = np.where(tr_mask & (df.fold.values != k))[0]
    ev_idx = np.where(df.fold.values == k)[0]
    ev_idx = eval_mask(df, ev_idx, set(df.Project.values[tr_idx]))
    folds.append((k, tr_idx, ev_idx))
print('eval samples excluded as near-duplicates per fold:', [int((df.fold.values == k).sum() - len(e)) for k, _, e in folds])
X = df[FEAT].values
rows, best = [], {}
for model in ['LR', 'XGB', 'RF']:
    for smell in LAB:
        y = df[smell].values
        for weighted in [False, True]:
            cur = None
            for ci, params in enumerate(GRIDS[model]):
                oof_idx, oof_p, oof_k = [], [], []
                for k, tri, evi in folds:
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
pd.DataFrame(rows).to_csv(W + '/cv_all_configs.csv', index=False)
pickle.dump(best, open(W + '/cv_best.pkl', 'wb'))
print('done in %.0fs' % (time.time() - t0))
