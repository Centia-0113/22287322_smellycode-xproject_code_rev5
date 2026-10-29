"""Post hoc step H (exploratory): the 873 samples the independent parser (step E) could not classify.
The granularity rule (step A) assigns every sample, so these samples (848 snippets javalang could not parse and 25
field declarations) were KEPT in the granularity-matched analysis at the level the rule gave them. This step reports
where they fall (rule level, partition, labels) and how much they could influence the post hoc results:
  H1  the frozen matched models of step C are re-scored on the previously examined test projects without these
      samples (no refitting, same thresholds);
  H2  steps B and C are repeated with these samples removed from both populations, i.e. from training, validation
      and test, and the results are compared with the original post hoc results.
Like steps C, F and G, this re-uses the previously examined test projects and is exploratory.
Usage:
  python src/unparsed_check.py --prepare <work dir> <H2 work dir>   # H2 work dir: level.pkl marks them 'unverified'
  python src/posthoc_tune.py <H2 work dir>; python src/posthoc_test.py <H2 work dir>
  python src/unparsed_check.py <work dir> [<H2 work dir>]            # writes <work dir>/unparsed_check.json"""
import os, sys, json, shutil, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def unverified_mask(W):
    pk = pd.read_pickle(W + '/granularity_parser.pkl')['parser_level'].values
    return ~np.isin(pk, ['class', 'method']), pk


if len(sys.argv) == 4 and sys.argv[1] == '--prepare':
    W, W2 = sys.argv[2], sys.argv[3]; os.makedirs(W2, exist_ok=True)
    unver, _ = unverified_mask(W)
    lv = pd.read_pickle(W + '/level.pkl'); vals = lv.values.astype(object); vals[unver] = 'unverified'
    pd.Series(vals, index=lv.index, name=lv.name).to_pickle(W2 + '/level.pkl')
    for f in ['clean.pkl', 'split_cols.pkl']:          # inputs of steps B and C; outputs are written to W2 only
        src, dst = os.path.abspath(os.path.join(W, f)), os.path.join(W2, f)
        if os.path.lexists(dst): os.remove(dst)
        try: os.symlink(src, dst)
        except OSError: shutil.copyfile(src, dst)
    print(f'{W2}: level.pkl marks {int(unver.sum())} samples as unverified; run steps B and C on it')
    sys.exit(0)

W = sys.argv[1]; W2 = sys.argv[2] if len(sys.argv) > 2 else None
df = load(W); lv = pd.read_pickle(W + '/level.pkl').values
unver, pk = unverified_mask(W)
H = json.load(open(W + '/posthoc_results.json')); POP = H['populations']
MODELS = ['RF', 'LR', 'XGB']; part = df.partition.values
vc = lambda a: {str(k): int(v) for k, v in pd.Series(a).value_counts().items()}
out = {'n_unverified': int(unver.sum()), 'parser_outcome': vc(pk[unver]), 'rule_level': vc(lv[unver]),
       'partition': {p: int((unver & (part == p)).sum()) for p in ['train', 'test']},
       'rule_level_by_partition': {p: vc(lv[unver & (part == p)]) for p in ['train', 'test']}}
# share of each matched population, and the labels carried by the unverified samples
out['population_share'] = {}
for pop in ['class', 'method']:
    for p in ['train', 'test']:
        m = (lv == pop) & (part == p)
        out['population_share'][f'{pop}_{p}'] = {'n': int(m.sum()), 'unverified': int((m & unver).sum()),
                                                 'share': round(float((m & unver).sum() / m.sum()), 4)}
out['positives'] = {}
for s in LAB:
    y = df[s].values.astype(bool); inpop = lv == POP[s]
    out['positives'][SHORT[s]] = {'all': int(y.sum()), 'unverified': int((y & unver).sum()),
                                  'unverified_in_own_population': int((y & unver & inpop).sum()),
                                  'unverified_test': int((y & unver & (part == 'test')).sum())}
# H1: frozen matched models re-scored with and without the unverified test samples
best = pickle.load(open(W + '/posthoc_cv_best.pkl', 'rb')); hp = pickle.load(open(W + '/posthoc_test_preds.pkl', 'rb'))
sel = {}
for m in MODELS:
    for s in LAB:
        a, b = best[(m, s, False)], best[(m, s, True)]
        sel[(m, s)] = b if (b['F1'], b['PRAUC']) > (a['F1'], a['PRAUC']) else a
out['H1_rescore_without_unverified'] = {}
for m in MODELS:
    res = {}
    for s in LAB:
        idx = hp['te_idx'][POP[s]]; y = df[s].values[idx]; p = hp['probs'][(m, s)]; thr = sel[(m, s)]['thr']
        full = scores(y, p, thr); keep = ~unver[idx]; excl = scores(y[keep], p[keep], thr)
        assert abs(full['F1'] - H['test_rescored'][m]['per_smell'][SHORT[s]]['F1']) < 1e-9, 'does not reproduce step C'
        res[SHORT[s]] = {'n_test': int(len(idx)), 'n_unverified_test': int((~keep).sum()),
                         'F1_with': round(full['F1'], 4), 'F1_without': round(excl['F1'], 4),
                         'PRAUC_with': round(full['PRAUC'], 4), 'PRAUC_without': round(excl['PRAUC'], 4)}
    res['macroF1_with'] = round(float(np.mean([res[SHORT[s]]['F1_with'] for s in LAB])), 4)
    res['macroF1_without'] = round(float(np.mean([res[SHORT[s]]['F1_without'] for s in LAB])), 4)
    out['H1_rescore_without_unverified'][m] = res
d1 = [abs(out['H1_rescore_without_unverified'][m][SHORT[s]]['F1_with'] - out['H1_rescore_without_unverified'][m][SHORT[s]]['F1_without'])
      for m in MODELS for s in LAB]
out['H1_max_abs_F1_change'] = round(float(max(d1)), 4)
# H2: matched protocol repeated without the unverified samples (if that run exists)
if W2 and os.path.exists(W2 + '/posthoc_results.json'):
    H2 = json.load(open(W2 + '/posthoc_results.json'))
    out['H2_rerun_without_unverified'] = {
        'selected_model': {'with': H['best_model_by_validation'], 'without': H2['best_model_by_validation']},
        'validation_macroF1': {m: {'with': round(H['validation'][m]['macroF1_mean'], 4), 'without': round(H2['validation'][m]['macroF1_mean'], 4)} for m in MODELS},
        'test_n': {'with': H['test_n'], 'without': H2['test_n']},
        'test': {m: {'macroF1': {'with': round(H['test_rescored'][m]['macroF1'], 4), 'without': round(H2['test_rescored'][m]['macroF1'], 4)},
                     'F1': {SHORT[s]: {'with': round(H['test_rescored'][m]['per_smell'][SHORT[s]]['F1'], 4),
                                       'without': round(H2['test_rescored'][m]['per_smell'][SHORT[s]]['F1'], 4)} for s in LAB},
                     'PRAUC': {SHORT[s]: {'with': round(H['test_rescored'][m]['per_smell'][SHORT[s]]['PRAUC'], 4),
                                          'without': round(H2['test_rescored'][m]['per_smell'][SHORT[s]]['PRAUC'], 4)} for s in LAB},
                     'macroF1_CI_without': [round(v, 4) for v in H2['test_rescored'][m]['CI']['macroF1']]} for m in MODELS}}
    d2 = [abs(H['test_rescored'][m]['per_smell'][SHORT[s]]['F1'] - H2['test_rescored'][m]['per_smell'][SHORT[s]]['F1']) for m in MODELS for s in LAB]
    out['H2_rerun_without_unverified']['max_abs_F1_change'] = round(float(max(d2)), 4)
    out['H2_rerun_without_unverified']['max_abs_macroF1_change'] = round(float(max(
        abs(H['test_rescored'][m]['macroF1'] - H2['test_rescored'][m]['macroF1']) for m in MODELS)), 4)
print(json.dumps(out, indent=1))
json.dump(out, open(W + '/unparsed_check.json', 'w'), indent=1)
