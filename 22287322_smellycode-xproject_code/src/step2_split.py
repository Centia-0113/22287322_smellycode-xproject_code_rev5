"""Step 2: seeded project-level split (StratifiedGroupKFold on four-label combinations), inner folds, assertions, manifest, Table I."""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
W = sys.argv[1]
LAB = ['God class', 'Data class', 'Long method', 'Feature envy']
df = pd.read_pickle(W + '/clean.pkl')
key = df[LAB].astype(int).astype(str).agg(''.join, axis=1).values
groups = df['group'].values
X0 = np.zeros((len(df), 1))
def ok(idx):  # every smell has at least one positive in this index set
    return all(df[LAB].values[idx, k].sum() > 0 for k in range(4))
used_seed = None
for seed in [42, 43, 44, 45, 46, 47, 48, 49, 50]:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(X0, key, groups))
    inner = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X0[tr], key[tr], groups[tr]))
    if ok(te) and ok(tr) and all(ok(tr[v]) and ok(tr[t]) for t, v in inner):
        used_seed = seed; break
    print('seed', seed, 'rejected: a partition or fold lacks positives for some smell')
assert used_seed is not None
fold = np.zeros(len(df), dtype=int)            # 0 = test, 1..5 = validation fold of the training partition
for k, (t, v) in enumerate(inner, start=1): fold[tr[v]] = k
df['partition'] = np.where(np.isin(np.arange(len(df)), te), 'test', 'train'); df['fold'] = fold
# --- disjointness assertions ---
P = lambda m: set(df.loc[m, 'group'])
assert P(df.partition == 'test').isdisjoint(P(df.partition == 'train')), 'train/test share a project group'
for k in range(1, 6):
    assert P(df.fold == k).isdisjoint(P((df.partition == 'train') & (df.fold != k))), f'fold {k} shares a group'
assert set(df.loc[df.partition == 'train', 'fold']) == {1, 2, 3, 4, 5} and (df.loc[df.partition == 'test', 'fold'] == 0).all()
print('assertions passed; seed used:', used_seed)
# --- manifest and Table I ---
man = df.groupby('Project').agg(group=('group', 'first'), partition=('partition', 'first'), fold=('fold', 'first'), samples=('group', 'size')).reset_index()
man.to_csv(W + '/split_manifest.csv', index=False)
def row(name, m):
    s = df.loc[m]
    r = {'Partition': name, 'Projects': s['Project'].nunique(), 'Samples': len(s)}
    for c in LAB: r[c] = f"{int(s[c].sum())} ({100*s[c].mean():.2f})"
    r['multi'] = f"{int((s[LAB].sum(axis=1) > 1).sum())}"
    return r
rows = [row('Full dataset', np.ones(len(df), bool)), row('Training', df.partition == 'train')]
rows += [row(f'Fold {k}', df.fold == k) for k in range(1, 6)]
rows += [row('Test (locked)', df.partition == 'test')]
t1 = pd.DataFrame(rows); print(t1.to_string(index=False)); t1.to_csv(W + '/table1.csv', index=False)
df[['Project', 'group', 'partition', 'fold']].to_pickle(W + '/split_cols.pkl')
json.dump({'seed': used_seed, 'test_projects': sorted(P(df.partition == 'test')), 'n_test_groups': len(P(df.partition == 'test')),
           'n_train_groups': len(P(df.partition == 'train'))}, open(W + '/split_info.json', 'w'), indent=1)
print('test projects:', sorted(P(df.partition == 'test')))
# where are LM / FE positives concentrated?
for c in ['Long method', 'Feature envy']:
    s = df.groupby('Project')[c].sum().sort_values(ascending=False)
    print(c, 'projects with positives:', int((s > 0).sum()), '| top:', s.head(6).to_dict())
