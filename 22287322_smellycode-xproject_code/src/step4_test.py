"""Step 4: model selection summary (validation), single scoring of frozen models on the locked test projects,
project-level bootstrap, co-occurrence analysis, and interpretability (held-out permutation importance + rank agreement)."""
import os, sys, json, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import spearmanr
W = sys.argv[1]; t0 = time.time()
df = load(W); best = pickle.load(open(W + '/cv_best.pkl', 'rb'))
X = df[FEAT].values; MODELS = ['RF', 'LR', 'XGB']
R = {}
# ---------- selection: weighting per model and smell (validation only) ----------
sel = {}
for m in MODELS:
    for s in LAB:
        a, b = best[(m, s, False)], best[(m, s, True)]
        w = (b['F1'], b['PRAUC']) > (a['F1'], a['PRAUC'])
        sel[(m, s)] = dict(best[(m, s, w)], weighted=w)
def perfold(entry, y):
    out = []
    for k in range(1, 6):
        mk = entry['oof_k'] == k; idx = entry['oof_idx'][mk]; p = entry['oof_p'][mk]
        out.append(scores(y[idx], p, entry['thr']))
    return out
val = {}
for m in MODELS:
    f1 = np.zeros((5, 4)); ap = np.zeros((5, 4))
    for j, s in enumerate(LAB):
        for k, sc in enumerate(perfold(sel[(m, s)], df[s].values)): f1[k, j], ap[k, j] = sc['F1'], sc['PRAUC']
    val[m] = {'macroF1_mean': f1.mean(1).mean(), 'macroF1_sd': f1.mean(1).std(ddof=1), 'macroAP_mean': ap.mean(1).mean(), 'macroAP_sd': ap.mean(1).std(ddof=1),
              'perSmellF1': dict(zip([SHORT[s] for s in LAB], f1.mean(0).round(3))), 'perSmellAP': dict(zip([SHORT[s] for s in LAB], ap.mean(0).round(3))),
              'settings': {SHORT[s]: {'params': sel[(m, s)]['params'], 'weighted': bool(sel[(m, s)]['weighted']), 'thr': round(sel[(m, s)]['thr'], 3)} for s in LAB}}
best_model = max(MODELS, key=lambda m: val[m]['macroF1_mean'])
R['validation'] = val; R['best_model_by_validation'] = best_model
print(json.dumps(val, indent=1, default=float)); print('best model by validation macro-F1:', best_model)
# Table V (left): effect of class weighting, validation fold means (weighted - unweighted), every model
dw = {}
for m in MODELS:
    for s in LAB:
        fw, fu = perfold(best[(m, s, True)], df[s].values), perfold(best[(m, s, False)], df[s].values)
        dw[f'{m}-{SHORT[s]}'] = {'dF1': float(np.mean([x['F1'] - y['F1'] for x, y in zip(fw, fu)])), 'dAP': float(np.mean([x['PRAUC'] - y['PRAUC'] for x, y in zip(fw, fu)]))}
R['weighting_delta'] = dw
# ---------- final refit on all training projects, single scoring on the locked test projects ----------
tr_idx = np.where(df.partition.values == 'train')[0]
te_all = np.where(df.partition.values == 'test')[0]
te_idx = eval_mask(df, te_all, set(df.Project.values[tr_idx]))
R['test_excluded_near_dups'] = int(len(te_all) - len(te_idx))
final, probs = {}, {}
for m in MODELS:
    for s in LAB:
        e = sel[(m, s)]
        est = make(m, e['params'], e['weighted']).fit(X[tr_idx], df[s].values[tr_idx])
        final[(m, s)] = est; probs[(m, s)] = est.predict_proba(X[te_idx])[:, 1]
print('final models fitted and test scored once  [%.0fs]' % (time.time() - t0))
Yte = df[LAB].values[te_idx]; proj_te = df.Project.values[te_idx]
test = {}
for m in MODELS:
    per = {}
    for j, s in enumerate(LAB): per[SHORT[s]] = scores(Yte[:, j], probs[(m, s)], sel[(m, s)]['thr'])
    yhat = np.column_stack([(probs[(m, s)] >= sel[(m, s)]['thr']).astype(int) for s in LAB])
    tp = (yhat & Yte).sum(); fp = (yhat & (1 - Yte)).sum(); fn = ((1 - yhat) & Yte).sum()
    micro = 2 * tp / (2 * tp + fp + fn)
    test[m] = {'per_smell': per, 'macroF1': np.mean([per[k]['F1'] for k in per]), 'macroAP': np.mean([per[k]['PRAUC'] for k in per]), 'microF1': micro}
# project-level bootstrap (1,000 resamples of test projects)
rng = np.random.default_rng(SEED); projs = np.unique(proj_te); pidx = {p: np.where(proj_te == p)[0] for p in projs}
boot = {m: {'macroF1': [], 'macroAP': [], **{f'F1_{SHORT[s]}': [] for s in LAB}} for m in MODELS}
for b in range(1000):
    pick = rng.choice(projs, size=len(projs), replace=True); ii = np.concatenate([pidx[p] for p in pick])
    for m in MODELS:
        f1s, aps = [], []
        for j, s in enumerate(LAB):
            y = Yte[ii, j]
            if y.sum() == 0: continue
            sc = scores(y, probs[(m, s)][ii], sel[(m, s)]['thr']); f1s.append(sc['F1']); aps.append(sc['PRAUC']); boot[m][f'F1_{SHORT[s]}'].append(sc['F1'])
        boot[m]['macroF1'].append(np.mean(f1s)); boot[m]['macroAP'].append(np.mean(aps))
for m in MODELS:
    test[m]['CI'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot[m].items()}
# per-project F1 spread for every model
for m in MODELS:
    spread = {}
    for j, s in enumerate(LAB):
        f = [scores(Yte[pidx[p], j], probs[(m, s)][pidx[p]], sel[(m, s)]['thr'])['F1'] for p in projs if Yte[pidx[p], j].sum() >= 3]
        spread[SHORT[s]] = {'n_projects': len(f), 'median': float(np.median(f)) if f else None, 'q1': float(np.percentile(f, 25)) if f else None, 'q3': float(np.percentile(f, 75)) if f else None}
    test[m]['per_project_F1'] = spread
R['test'] = test
print(json.dumps(test, indent=1, default=float))
# ---------- co-occurrence (data profile on the cleaned dataset) and stratified recall on test ----------
Yall = df[LAB].values; co = {}
for a in range(4):
    for b in range(a + 1, 4):
        n11 = int((Yall[:, a] & Yall[:, b]).sum()); n1a, n1b, N = Yall[:, a].sum(), Yall[:, b].sum(), len(Yall)
        n10, n01 = n1a - n11, n1b - n11; n00 = N - n11 - n10 - n01
        phi = (n11 * n00 - n10 * n01) / np.sqrt(float(n1a) * (N - n1a) * n1b * (N - n1b))
        co[f'{SHORT[LAB[a]]}-{SHORT[LAB[b]]}'] = {'n11': n11, f'P({SHORT[LAB[b]]}|{SHORT[LAB[a]]})': n11 / n1a, f'P({SHORT[LAB[a]]}|{SHORT[LAB[b]]})': n11 / n1b, 'phi': phi}
R['cooccurrence'] = co; print(json.dumps(co, indent=1, default=float))
partner = {'God class': 'Data class', 'Data class': 'God class', 'Long method': 'Feature envy', 'Feature envy': 'Long method'}
strat = {}
for m in MODELS:
    for j, s in enumerate(LAB):
        k = LAB.index(partner[s]); pos = Yte[:, j] == 1; alone = pos & (Yte[:, k] == 0); both = pos & (Yte[:, k] == 1)
        hit = probs[(m, s)] >= sel[(m, s)]['thr']
        strat[f'{m}-{SHORT[s]}'] = {'n_alone': int(alone.sum()), 'recall_alone': float(hit[alone].mean()) if alone.sum() else None,
                                    'n_co': int(both.sum()), 'recall_co': float(hit[both].mean()) if both.sum() else None}
R['stratified_recall'] = strat; print(json.dumps(strat, indent=1))
# ---------- interpretability ----------
CL = {'Size/operands': ['Logical Lines', 'Distinct Operands', 'Total Operands', 'Vocabulary', 'Length', 'Calculated Length', 'Volume', 'Bugs'],
      'Distinct operators': ['Distinct Operators'], 'Operator effort': ['Total Operators', 'Difficulty', 'Effort', 'Time Required'],
      'Cyclomatic complexity': ['Cyclomatic Complexity']}
cidx = {c: [FEAT.index(f) for f in fs] for c, fs in CL.items()}
def perm_imp(est, Xe, y, reps=5, seed=SEED):
    base = average_precision_score(y, est.predict_proba(Xe)[:, 1]); out = {}
    r = np.random.default_rng(seed)
    for c, cols in cidx.items():
        d = []
        for _ in range(reps):
            Xp = Xe.copy(); perm = r.permutation(len(Xe)); Xp[:, cols] = Xe[perm][:, cols]
            d.append(base - average_precision_score(y, est.predict_proba(Xp)[:, 1]))
        out[c] = float(np.mean(d))
    return out
pi_test = {f'{m}-{SHORT[s]}': perm_imp(final[(m, s)], X[te_idx], df[s].values[te_idx]) for m in MODELS for s in LAB}
print('test permutation importance done  [%.0fs]' % (time.time() - t0))
pi_val = {}
trm = df.partition.values == 'train'
for m in MODELS:
    for s in LAB:
        e = sel[(m, s)]; y = df[s].values; per = []
        for k in range(1, 6):
            tri = np.where(trm & (df.fold.values != k))[0]; evi = eval_mask(df, np.where(df.fold.values == k)[0], set(df.Project.values[tri]))
            est = make(m, e['params'], e['weighted']).fit(X[tri], y[tri]); per.append(perm_imp(est, X[evi], y[evi]))
        pi_val[f'{m}-{SHORT[s]}'] = {c: {'mean': float(np.mean([p[c] for p in per])), 'sd': float(np.std([p[c] for p in per], ddof=1))} for c in CL}
print('validation-fold permutation importance done  [%.0fs]' % (time.time() - t0))
def specific(m, est):
    keep = est.named_steps['screen'].keep_; kept = [FEAT[i] for i in keep]
    if m == 'RF': v = est.named_steps['clf'].feature_importances_
    elif m == 'LR': v = np.abs(est.named_steps['clf'].coef_[0])
    else:
        g = est.named_steps['clf'].m_.get_booster().get_score(importance_type='total_gain'); v = np.array([g.get(f'f{i}', 0.0) for i in range(len(kept))])
    imp = {c: float(sum(v[kept.index(f)] for f in fs if f in kept)) for c, fs in CL.items()}
    return imp, kept
agree, spec = {}, {}
for m in MODELS:
    for s in LAB:
        imp, kept = specific(m, final[(m, s)]); spec[f'{m}-{SHORT[s]}'] = imp
        a = [pi_test[f'{m}-{SHORT[s]}'][c] for c in CL]; b = [imp[c] for c in CL]
        rho = spearmanr(a, b).correlation
        agree[f'{m}-{SHORT[s]}'] = {'spearman': float(rho), 'top_match': bool(np.argmax(a) == np.argmax(b)), 'top_perm': list(CL)[int(np.argmax(a))], 'top_specific': list(CL)[int(np.argmax(b))]}
cross = {}
for s in LAB:
    for m1, m2 in [('RF', 'XGB'), ('RF', 'LR'), ('XGB', 'LR')]:
        a = [pi_test[f'{m1}-{SHORT[s]}'][c] for c in CL]; b = [pi_test[f'{m2}-{SHORT[s]}'][c] for c in CL]
        cross[f'{SHORT[s]} {m1}-{m2}'] = float(spearmanr(a, b).correlation)
R.update({'clusters': CL, 'perm_test': pi_test, 'perm_val': pi_val, 'specific': spec, 'rank_agreement': agree, 'cross_model_perm_rho': cross, 'kept_features_final': kept})
print(json.dumps({'perm_test': pi_test, 'rank_agreement': agree, 'cross': cross}, indent=1))
json.dump(R, open(W + '/results.json', 'w'), indent=1, default=float)
pickle.dump({'te_idx': te_idx, 'probs': probs, 'sel': {k: {kk: v for kk, v in e.items() if kk not in ('oof_idx', 'oof_p', 'oof_k')} for k, e in sel.items()}}, open(W + '/test_preds.pkl', 'wb'))
print('done in %.0fs' % (time.time() - t0))
