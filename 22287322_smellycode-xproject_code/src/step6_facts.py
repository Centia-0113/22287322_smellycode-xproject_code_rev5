"""Descriptive facts quoted in the paper that are not produced by steps 1-5 (run after step 5).
Usage: python step6_facts.py <raw SmellyCode++ CSV> <work dir>"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import spearmanr
RAW, W = sys.argv[1], sys.argv[2]
THRIFT = r'org\.apache\.thrift|TProtocol|TBase|TField'
raw = pd.read_csv(RAW, low_memory=False, usecols=['Project', 'Code', 'Logical Lines'] + LAB)
raw['_n'] = raw['Code'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
raw = raw[~((raw['Logical Lines'] == 0) & (raw['_n'].str.len() > 0))]          # the one implausible record (step 1)
# 1) contradictory labels: share of Apache Thrift code, and where the removed labels came from
nun = raw.groupby('_n')[LAB].nunique().max(axis=1); conf = set(nun[nun > 1].index)
c = raw[raw['_n'].isin(conf)]; thr = c['_n'].str.contains(THRIFT, regex=True)
print('contradictory-label rows using the Apache Thrift API: %d of %d (%.1f%%)' % (thr.sum(), len(c), 100 * thr.mean()))
print('labels in contradictory rows:', c[LAB].sum().astype(int).to_dict(), '| of which Thrift:', c.loc[thr, LAB].sum().astype(int).to_dict())
d = raw[~raw['_n'].isin(conf)]; dup = d.duplicated('_n', keep='first')
print('labels in removed duplicate copies:', d.loc[dup, LAB].sum().astype(int).to_dict())
print('raw labels:', raw[LAB].sum().astype(int).to_dict())
# 2) correlation screening keeps the same metrics in every fold; pairwise Spearman correlations
df = load(W); tr = df[df.partition == 'train']
for k in range(1, 6):
    t = df[(df.partition == 'train') & (df.fold != k)]
    print('fold', k, 'metrics kept by CorrScreen:', [FEAT[i] for i in CorrScreen().fit(t[FEAT].values).keep_])
print('final training partition kept:', [FEAT[i] for i in CorrScreen().fit(tr[FEAT].values).keep_])
for name, x in [('cleaned dataset', df), ('training projects', tr)]:
    r = spearmanr(x[FEAT].values).statistic; np.fill_diagonal(r, np.nan)
    print('%s: minimum pairwise Spearman rho %.3f' % (name, np.nanmin(r)))
# 3) granularity: class- versus method-level samples (no granularity column in the file; rule in granularity.py)
from granularity import level
lv = df['_n'].map(level).values; is_cls = lv == 'class'
for s in LAB: print('%s positives that are class-level: %.3f' % (SHORT[s], is_cls[df[s].values == 1].mean()))
lm_med = df.loc[df['Long method'] == 1, 'Logical Lines'].median(); clean_cls = is_cls & (df[LAB].sum(axis=1).values == 0)
print('median Logical Lines of Long Method samples: %.0f; smell-free class-level samples at least that long: %.1f%%' % (lm_med, 100 * (df.loc[clean_cls, 'Logical Lines'] >= lm_med).mean()))
# 4) random-split contrast relative to prevalence (XGBoost, training projects only)
ct = json.load(open(W + '/contrast.json'))
for k in ['LM', 'FE', 'GC', 'DC']:
    r = ct[f'XGB-{k}']
    print('XGB %s: PR-AUC random raw %.3f (%.1fx prevalence %.4f) | random clean %.3f | grouped %.3f (%.1fx prevalence %.4f)' % (
        k, r['random_raw'], r['random_raw'] / r['prevalence_raw'], r['prevalence_raw'], r['random_clean'], r['grouped_cv'], r['grouped_cv'] / r['prevalence_clean'], r['prevalence_clean']))
