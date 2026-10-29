"""Post hoc step C (exploratory): selection on granularity-matched validation results (same rules as step 4), refit on
matched training samples, and RE-SCORING of the previously examined test projects. These test projects were already
used for the primary results, so this is NOT an independent, untouched test. Also: project bootstrap, per-project F1,
co-occurrence and stratified recall, and held-out permutation importance on matched VALIDATION folds.
Usage: python posthoc_test.py <work dir>"""
import os, sys, json, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
W = sys.argv[1]; t0 = time.time()
df = load(W); lv = pd.read_pickle(W + '/level.pkl').values
best = pickle.load(open(W + '/posthoc_cv_best.pkl', 'rb'))
POP = {'God class': 'class', 'Data class': 'class', 'Long method': 'method', 'Feature envy': 'method'}
X = df[FEAT].values; MODELS = ['RF', 'LR', 'XGB']; R = {'populations': POP}
# ---------- selection on validation (same rules as the primary protocol) ----------
sel = {}
for m in MODELS:
    for s in LAB:
        a, b = best[(m, s, False)], best[(m, s, True)]
        w = (b['F1'], b['PRAUC']) > (a['F1'], a['PRAUC'])
        sel[(m, s)] = dict(best[(m, s, w)], weighted=w)
def perfold(entry, y):
    return [scores(y[entry['oof_idx'][entry['oof_k'] == k]], entry['oof_p'][entry['oof_k'] == k], entry['thr']) for k in range(1, 6)]
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
print(json.dumps(val, indent=1, default=float)); print('best model by matched validation macro-F1:', best_model, flush=True)
dw = {}
for m in MODELS:
    for s in LAB:
        fw, fu = perfold(best[(m, s, True)], df[s].values), perfold(best[(m, s, False)], df[s].values)
        dw[f'{m}-{SHORT[s]}'] = {'dF1': float(np.mean([x['F1'] - y['F1'] for x, y in zip(fw, fu)])), 'dAP': float(np.mean([x['PRAUC'] - y['PRAUC'] for x, y in zip(fw, fu)]))}
R['weighting_delta'] = dw
# ---------- refit on matched training samples; re-score the previously examined test projects ----------
tr_idx, te_idx = {}, {}
for pop in ['class', 'method']:
    tr_idx[pop] = np.where((df.partition.values == 'train') & (lv == pop))[0]
    te_all = np.where((df.partition.values == 'test') & (lv == pop))[0]
    te_idx[pop] = eval_mask(df, te_all, set(df.Project.values[tr_idx[pop]]))
R['test_excluded_near_dups'] = {pop: int(((df.partition.values == 'test') & (lv == pop)).sum() - len(te_idx[pop])) for pop in te_idx}
R['test_n'] = {pop: int(len(te_idx[pop])) for pop in te_idx}
final, probs = {}, {}
for m in MODELS:
    for s in LAB:
        e = sel[(m, s)]; p = POP[s]
        est = make(m, e['params'], e['weighted']).fit(X[tr_idx[p]], df[s].values[tr_idx[p]])
        final[(m, s)] = est; probs[(m, s)] = est.predict_proba(X[te_idx[p]])[:, 1]
print('matched models refitted and previously examined test projects re-scored  [%.0fs]' % (time.time() - t0), flush=True)
Y = {s: df[s].values[te_idx[POP[s]]] for s in LAB}; PRJ = {pop: df.Project.values[te_idx[pop]] for pop in te_idx}
test = {}
for m in MODELS:
    per = {SHORT[s]: scores(Y[s], probs[(m, s)], sel[(m, s)]['thr']) for s in LAB}
    tp = fp = fn = 0
    for s in LAB:
        yh = (probs[(m, s)] >= sel[(m, s)]['thr']).astype(int); y = Y[s]
        tp += int((yh & y).sum()); fp += int((yh & (1 - y)).sum()); fn += int(((1 - yh) & y).sum())
    test[m] = {'per_smell': per, 'macroF1': float(np.mean([per[k]['F1'] for k in per])), 'macroAP': float(np.mean([per[k]['PRAUC'] for k in per])),
               'microF1': 2 * tp / (2 * tp + fp + fn), 'prevalence': {SHORT[s]: float(Y[s].mean()) for s in LAB}}
rng = np.random.default_rng(SEED); projs = np.unique(np.concatenate([PRJ['class'], PRJ['method']]))
pidx = {pop: {p: np.where(PRJ[pop] == p)[0] for p in projs} for pop in PRJ}
boot = {m: {'macroF1': [], 'macroAP': [], **{f'F1_{SHORT[s]}': [] for s in LAB}, **{f'AP_{SHORT[s]}': [] for s in LAB}} for m in MODELS}
for b in range(1000):
    pick = rng.choice(projs, size=len(projs), replace=True)
    ii = {pop: np.concatenate([pidx[pop][p] for p in pick]) for pop in PRJ}
    for m in MODELS:
        f1s, aps = [], []
        for s in LAB:
            i = ii[POP[s]]; y = Y[s][i]
            if y.sum() == 0: continue
            sc = scores(y, probs[(m, s)][i], sel[(m, s)]['thr']); f1s.append(sc['F1']); aps.append(sc['PRAUC'])
            boot[m][f'F1_{SHORT[s]}'].append(sc['F1']); boot[m][f'AP_{SHORT[s]}'].append(sc['PRAUC'])
        boot[m]['macroF1'].append(np.mean(f1s)); boot[m]['macroAP'].append(np.mean(aps))
for m in MODELS:
    test[m]['CI'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot[m].items()}
    spread = {}
    for s in LAB:
        pop = POP[s]
        f = [scores(Y[s][pidx[pop][p]], probs[(m, s)][pidx[pop][p]], sel[(m, s)]['thr'])['F1'] for p in projs if Y[s][pidx[pop][p]].sum() >= 3]
        spread[SHORT[s]] = {'n_projects': len(f), 'median': float(np.median(f)) if f else None, 'q1': float(np.percentile(f, 25)) if f else None, 'q3': float(np.percentile(f, 75)) if f else None}
    test[m]['per_project_F1'] = spread
R['test_rescored'] = test
print(json.dumps(test, indent=1, default=float), flush=True)
# ---------- co-occurrence within populations, stratified recall on the re-scored test ----------
co = {}
for a, b, pop in [('God class', 'Data class', 'class'), ('Long method', 'Feature envy', 'method')]:
    msk = lv == pop; A = df[a].values[msk]; B = df[b].values[msk]; N = int(msk.sum())
    n11 = int((A & B).sum()); n1a, n1b = int(A.sum()), int(B.sum()); n10, n01 = n1a - n11, n1b - n11; n00 = N - n11 - n10 - n01
    co[f'{SHORT[a]}-{SHORT[b]}'] = {'N': N, 'n11': n11, 'phi': (n11 * n00 - n10 * n01) / np.sqrt(float(n1a) * (N - n1a) * n1b * (N - n1b))}
R['cooccurrence_within_population'] = co
partner = {'God class': 'Data class', 'Data class': 'God class', 'Long method': 'Feature envy', 'Feature envy': 'Long method'}
strat = {}
for m in MODELS:
    for s in LAB:
        pos = Y[s] == 1; other = Y[partner[s]]; hit = probs[(m, s)] >= sel[(m, s)]['thr']
        alone, both = pos & (other == 0), pos & (other == 1)
        strat[f'{m}-{SHORT[s]}'] = {'n_alone': int(alone.sum()), 'recall_alone': float(hit[alone].mean()) if alone.sum() else None,
                                    'n_co': int(both.sum()), 'recall_co': float(hit[both].mean()) if both.sum() else None}
R['stratified_recall'] = strat
# ---------- interpretability on matched VALIDATION folds ----------
CL0 = {'Size/operands': ['Logical Lines', 'Distinct Operands', 'Total Operands', 'Vocabulary', 'Length', 'Calculated Length', 'Volume', 'Bugs'],
       'Distinct operators': ['Distinct Operators'], 'Operator effort': ['Total Operators', 'Difficulty', 'Effort', 'Time Required'],
       'Cyclomatic complexity': ['Cyclomatic Complexity']}
def name_cluster(fs):
    if fs == ['Logical Lines']: return 'Logical lines'
    if fs == ['Cyclomatic Complexity']: return 'Cyclomatic complexity'
    if 'Logical Lines' in fs: return 'Size/operands'
    if fs == ['Distinct Operators']: return 'Distinct operators'
    if fs == ['Total Operators']: return 'Total operators'
    if fs == ['Difficulty']: return 'Difficulty'
    if 'Total Operators' in fs: return 'Operators'
    if 'Total Operands' in fs: return 'Operands'
    return '+'.join(fs)
CLP = {}
for pop in ['class', 'method']:
    tr = df[(df.partition.values == 'train') & (lv == pop)]
    rho = tr[FEAT].rank().corr(); D = 1 - rho.abs().values; np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), 'average'), t=0.05, criterion='distance')
    cl = {}
    for f, l in zip(FEAT, lab): cl.setdefault(l, []).append(f)
    CLP[pop] = {name_cluster(v): v for v in cl.values()}
    r = rho.abs().values.copy(); np.fill_diagonal(r, np.nan); R.setdefault('min_pairwise_rho_train', {})[pop] = float(np.nanmin(r))
R['clusters_matched'] = CLP; print('matched clusters:', json.dumps(CLP), flush=True)
def perm_imp(est, Xe, y, CL, reps=5, seed=SEED):
    base = average_precision_score(y, est.predict_proba(Xe)[:, 1]); out = {}; r = np.random.default_rng(seed)
    for c, fs in CL.items():
        cols = [FEAT.index(f) for f in fs]; d = []
        for _ in range(reps):
            Xp = Xe.copy(); perm = r.permutation(len(Xe)); Xp[:, cols] = Xe[perm][:, cols]
            d.append(base - average_precision_score(y, est.predict_proba(Xp)[:, 1]))
        out[c] = float(np.mean(d))
    return out
pv, pv0 = {}, {}
trm = df.partition.values == 'train'
for m in MODELS:
    for s in LAB:
        e = sel[(m, s)]; y = df[s].values; pop = POP[s]; msk = lv == pop; per, per0 = [], []
        for k in range(1, 6):
            tri = np.where(trm & msk & (df.fold.values != k))[0]
            evi = eval_mask(df, np.where(msk & (df.fold.values == k))[0], set(df.Project.values[tri]))
            est = make(m, e['params'], e['weighted']).fit(X[tri], y[tri])
            per.append(perm_imp(est, X[evi], y[evi], CLP[pop])); per0.append(perm_imp(est, X[evi], y[evi], CL0))
        pv[f'{m}-{SHORT[s]}'] = {c: {'mean': float(np.mean([p[c] for p in per])), 'sd': float(np.std([p[c] for p in per], ddof=1))} for c in CLP[pop]}
        pv0[f'{m}-{SHORT[s]}'] = {c: {'mean': float(np.mean([p[c] for p in per0])), 'sd': float(np.std([p[c] for p in per0], ddof=1))} for c in CL0}
        print(m, SHORT[s], 'matched-validation permutation importance done  [%.0fs]' % (time.time() - t0), flush=True)
def specific(m, est, CL):
    kept = [FEAT[i] for i in est.named_steps['screen'].keep_]
    if m == 'RF': v = est.named_steps['clf'].feature_importances_
    elif m == 'LR': v = np.abs(est.named_steps['clf'].coef_[0])
    else:
        g = est.named_steps['clf'].m_.get_booster().get_score(importance_type='total_gain'); v = np.array([g.get(f'f{i}', 0.0) for i in range(len(kept))])
    return {c: float(sum(v[kept.index(f)] for f in fs if f in kept)) for c, fs in CL.items()}, kept
agree, spec, kept_all = {}, {}, {}
for m in MODELS:
    for s in LAB:
        CL = CLP[POP[s]]; imp, kept = specific(m, final[(m, s)], CL); spec[f'{m}-{SHORT[s]}'] = imp; kept_all[POP[s]] = kept
        a = [pv[f'{m}-{SHORT[s]}'][c]['mean'] for c in CL]; b = [imp[c] for c in CL]
        agree[f'{m}-{SHORT[s]}'] = {'spearman': float(spearmanr(a, b).correlation), 'top_match': bool(np.argmax(a) == np.argmax(b)),
                                    'top_perm': list(CL)[int(np.argmax(a))], 'top_specific': list(CL)[int(np.argmax(b))]}
cross = {}
for s in LAB:
    CL = CLP[POP[s]]
    for m1, m2 in [('RF', 'XGB'), ('RF', 'LR'), ('XGB', 'LR')]:
        cross[f'{SHORT[s]} {m1}-{m2}'] = float(spearmanr([pv[f'{m1}-{SHORT[s]}'][c]['mean'] for c in CL], [pv[f'{m2}-{SHORT[s]}'][c]['mean'] for c in CL]).correlation)
R.update({'perm_val_matched': pv, 'perm_val_matched_original_clusters': pv0, 'specific': spec, 'rank_agreement': agree,
          'cross_model_perm_rho': cross, 'kept_features': kept_all})
print(json.dumps({'perm_val_matched': pv, 'rank_agreement': agree, 'cross': cross}, indent=1))
json.dump(R, open(W + '/posthoc_results.json', 'w'), indent=1, default=float)
pickle.dump({'te_idx': te_idx, 'probs': probs}, open(W + '/posthoc_test_preds.pkl', 'wb'))
print('done in %.0fs' % (time.time() - t0))
