"""Step 1: rule-based cleaning (nothing fitted) and project grouping via near-duplicate screening."""
import sys, re, time, json, itertools, collections, numpy as np, pandas as pd
from datasketch import MinHash, MinHashLSH
t0 = time.time()
SRC, OUT = sys.argv[1], sys.argv[2]
LAB = ['God class', 'Data class', 'Long method', 'Feature envy']
FEAT = ['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length',
        'Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
df = pd.read_csv(SRC, low_memory=False)
log = {'raw_rows': len(df), 'raw_projects': int(df['Project'].nunique()), 'raw_labels': df[LAB].sum().astype(int).to_dict()}
# --- rule-based checks ---
num = df[FEAT]
bad_num = num.isna().any(axis=1) | np.isinf(num).any(axis=1) | (num < 0).any(axis=1)
v_ok = np.isclose(df['Vocabulary'], df['Distinct Operators'] + df['Distinct Operands'])
l_ok = np.isclose(df['Length'], df['Total Operators'] + df['Total Operands'])
df['_n'] = df['Code'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
implaus = (df['Logical Lines'] == 0) & (df['_n'].str.len() > 0)
log['missing_or_invalid_values'] = int(bad_num.sum()); log['halstead_identity_violations'] = int((~v_ok | ~l_ok).sum())
log['implausible_zero_logical_lines'] = int(implaus.sum())
df = df[~(bad_num | ~v_ok | ~l_ok | implaus)].copy()
nun = df.groupby('_n')[LAB].nunique().max(axis=1)
conf = set(nun[nun > 1].index)
log['conflicting_label_strings'] = len(conf); log['conflicting_label_rows'] = int(df['_n'].isin(conf).sum())
df = df[~df['_n'].isin(conf)]
log['within_project_duplicate_rows'] = int(df.duplicated(['Project', '_n']).sum())
xp = df.groupby('_n')['Project'].nunique()
log['cross_project_identical_strings'] = int((xp > 1).sum())
n_before = len(df)
df = df.drop_duplicates('_n', keep='first').reset_index(drop=True)
log['duplicate_rows_removed'] = n_before - len(df)
log['clean_rows'] = len(df); log['clean_labels'] = df[LAB].sum().astype(int).to_dict()
print(json.dumps(log, indent=1))
# --- near-duplicate screening across projects (MinHash, 5-token shingles, Jaccard >= 0.8) ---
TOK_MIN, CAP, K, THR, NPERM = 30, 3000, 5, 0.8, 64
toks = df['_n'].str.split(' ')
ntok = toks.str.len()
idx = np.where(ntok >= TOK_MIN)[0]
lsh = MinHashLSH(threshold=THR, num_perm=NPERM)
mh = {}
for i in idx:
    t = toks.iat[i][:CAP]
    sh = {' '.join(t[j:j+K]).encode('utf8') for j in range(len(t) - K + 1)}
    m = MinHash(num_perm=NPERM, seed=42); m.update_batch(list(sh))
    mh[i] = m; lsh.insert(i, m)
print('minhash built for', len(idx), 'non-trivial samples in %.0fs' % (time.time() - t0))
proj = df['Project'].values
pairs = set()
for i in idx:
    for j in lsh.query(mh[i]):
        if j != i and proj[j] != proj[i] and mh[i].jaccard(mh[j]) >= THR:
            pairs.add((min(i, j), max(i, j)))
print('cross-project near-duplicate pairs:', len(pairs))
# per project pair: fraction of the smaller project's non-trivial samples that have a near-duplicate in the other
nt_per_proj = collections.Counter(proj[idx])
hit = collections.defaultdict(set)
for i, j in pairs:
    a, b = sorted((proj[i], proj[j]))
    hit[(a, b, proj[i])].add(i); hit[(a, b, proj[j])].add(j)
rows = []
for (a, b) in {(k[0], k[1]) for k in hit}:
    fa = len(hit.get((a, b, a), ())) / nt_per_proj[a]; fb = len(hit.get((a, b, b), ())) / nt_per_proj[b]
    rows.append((a, b, len(hit.get((a, b, a), ())), len(hit.get((a, b, b), ())), max(fa, fb)))
pp = pd.DataFrame(rows, columns=['a', 'b', 'samples_a', 'samples_b', 'max_frac']).sort_values('max_frac', ascending=False)
print(pp.head(15).to_string(index=False))
LINK = 0.10   # link two projects when >=10% of either project's non-trivial samples have near-duplicates in the other
parent = {p: p for p in df['Project'].unique()}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for r in pp.itertuples():
    if r.max_frac >= LINK: parent[find(r.a)] = find(r.b)
df['group'] = df['Project'].map(find)
log.update({'near_dup_pairs': len(pairs), 'near_dup_project_pairs': len(pp), 'max_pair_overlap': float(pp['max_frac'].max()) if len(pp) else 0.0,
            'max_pair': pp.iloc[0][['a','b']].tolist() if len(pp) else None, 'link_threshold': LINK, 'groups': int(df['group'].nunique()),
            'merged_projects': int(df['Project'].nunique() - df['group'].nunique())})
# residual near-duplicate samples (below the linking threshold) are flagged so they can be kept out of evaluation sets
nd_cluster = {}
for i, j in pairs: nd_cluster.setdefault(i, set()).add(j); nd_cluster.setdefault(j, set()).add(i)
df['nd_partners'] = [sorted(nd_cluster.get(i, ())) for i in range(len(df))]
log['samples_with_cross_project_near_dups'] = int(sum(1 for v in nd_cluster.values() if v))
pp.to_csv(OUT + '/near_dup_project_pairs.csv', index=False)
df.drop(columns=['Code']).to_pickle(OUT + '/clean.pkl')
json.dump(log, open(OUT + '/clean_log.json', 'w'), indent=1)
print(json.dumps({k: log[k] for k in ['near_dup_pairs','near_dup_project_pairs','max_pair_overlap','max_pair','groups','merged_projects','samples_with_cross_project_near_dups']}, indent=1))
print('done in %.0fs' % (time.time() - t0))
