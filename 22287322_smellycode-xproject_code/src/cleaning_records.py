"""Cleaning records (supports step 1): a per-sample record of every row that step 1 removed, a per-string record of the
code strings whose copies carry contradictory labels, and the counts behind Table I (upper panel).
The script repeats the rule-based decisions of step 1 (same normalisation and order) and checks that the rows it keeps
are exactly those in clean.pkl.
Outputs in <work dir>/cleaning_records/:
  removed_samples.csv       one row per removed sample: raw_row (0-based data row of the CSV), project, file, class, step,
                            string_id, kept_copy_raw_row (duplicate copies only), the four labels, Thrift flag
  contradictory_strings.csv one row per contradictory code string: copies, projects, positive copies per smell, Thrift flag
  cleaning_record.json      rows and positive labels after each step, labels removed, Thrift shares (removed and kept labels)
Usage: python cleaning_records.py <raw CSV> <work dir>"""
import os, sys, json, hashlib, numpy as np, pandas as pd
RAW, W = sys.argv[1], sys.argv[2]
LAB = ['God class', 'Data class', 'Long method', 'Feature envy']
SH = {'God class': 'GC', 'Data class': 'DC', 'Long method': 'LM', 'Feature envy': 'FE'}
FEAT = ['Logical Lines', 'Distinct Operators', 'Distinct Operands', 'Total Operators', 'Total Operands', 'Vocabulary', 'Length',
        'Calculated Length', 'Volume', 'Difficulty', 'Effort', 'Time Required', 'Bugs', 'Cyclomatic Complexity']
THRIFT = r'org\.apache\.thrift|TProtocol|TBase|TField'          # same pattern as step6_facts.py
OUT = os.path.join(W, 'cleaning_records'); os.makedirs(OUT, exist_ok=True)

def positives(d):
    return {SH[s]: int(d[s].sum()) for s in LAB}

df = pd.read_csv(RAW, low_memory=False)
df['raw_row'] = np.arange(len(df))
df['_n'] = df['Code'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
df['string_id'] = [hashlib.sha1(s.encode('utf8')).hexdigest()[:12] for s in df['_n']]
df['thrift'] = df['_n'].str.contains(THRIFT, regex=True).astype(int)
steps = [{'step': 'raw file', 'rows_removed': 0, 'rows_left': int(len(df)), 'positives_left': positives(df)}]
removed = []

# 1) value checks, as in step 1
num = df[FEAT]
bad_num = num.isna().any(axis=1) | np.isinf(num).any(axis=1) | (num < 0).any(axis=1)
v_ok = np.isclose(df['Vocabulary'], df['Distinct Operators'] + df['Distinct Operands'])
l_ok = np.isclose(df['Length'], df['Total Operators'] + df['Total Operands'])
zero_ll = (df['Logical Lines'] == 0) & (df['_n'].str.len() > 0)
for name, m in [('invalid or missing value', bad_num), ('Halstead identity violated', ~v_ok | ~l_ok), ('zero logical lines', zero_ll)]:
    r = df[m].copy(); r['step'] = name; r['kept_copy_raw_row'] = np.nan; removed.append(r)
drop = bad_num | ~v_ok | ~l_ok | zero_ll
df = df[~drop].copy()
steps.append({'step': 'value checks', 'rows_removed': int(drop.sum()), 'rows_left': int(len(df)), 'positives_left': positives(df)})

# 2) contradictory labels: identical code whose copies disagree on any smell
nun = df.groupby('_n')[LAB].nunique().max(axis=1)
conf = set(nun[nun > 1].index)
mc = df['_n'].isin(conf)
r = df[mc].copy(); r['step'] = 'contradictory labels'; r['kept_copy_raw_row'] = np.nan; removed.append(r)
agg = {'copies': ('raw_row', 'size'), 'projects': ('Project', 'nunique'),
       'project_list': ('Project', lambda x: ';'.join(sorted(set(x)))), 'thrift': ('thrift', 'max'),
       'first_raw_row': ('raw_row', 'min')}
agg.update({f'positive_copies_{SH[s]}': (s, 'sum') for s in LAB})
cs = df[mc].groupby('string_id').agg(**agg).reset_index().sort_values(['copies', 'first_raw_row'], ascending=[False, True])
cs.to_csv(os.path.join(OUT, 'contradictory_strings.csv'), index=False)
steps.append({'step': 'contradictory labels', 'strings': len(conf), 'rows_removed': int(mc.sum()),
              'rows_removed_thrift': int(df.loc[mc, 'thrift'].sum()),
              'positives_removed': positives(df[mc]), 'positives_removed_on_thrift_code': positives(df[mc & (df['thrift'] == 1)]),
              'distinct_strings_with_a_positive_copy': {SH[s]: int((cs[f'positive_copies_{SH[s]}'] > 0).sum()) for s in LAB}})
df = df[~mc].copy()
steps[-1].update({'rows_left': int(len(df)), 'positives_left': positives(df)})

# 3) further duplicate copies: keep the first copy (file order) of every remaining code string
dup = df.duplicated('_n', keep='first')
kept = df[~dup].set_index('_n')
r = df[dup].copy(); r['step'] = 'duplicate copy'; r['kept_copy_raw_row'] = r['_n'].map(kept['raw_row']); removed.append(r)
same_project = int((r['Project'].values == kept.loc[r['_n'], 'Project'].values).sum())
steps.append({'step': 'duplicate copies', 'rows_removed': int(dup.sum()), 'copies_from_the_same_project_as_the_kept_copy': same_project,
              'positives_removed': positives(df[dup]), 'rows_left': int((~dup).sum()), 'positives_left': positives(df[~dup]),
              'positives_left_on_thrift_code': positives(df[~dup & (df['thrift'] == 1)])})
df = df[~dup]

rec = pd.concat(removed)[['raw_row', 'Project', 'File', 'Class', 'step', 'string_id', 'kept_copy_raw_row'] + LAB + ['thrift']]
rec['kept_copy_raw_row'] = rec['kept_copy_raw_row'].astype('Int64')
rec.rename(columns=SH).sort_values('raw_row').to_csv(os.path.join(OUT, 'removed_samples.csv'), index=False)

# the kept rows must be exactly the cleaned set of step 1
clean = pd.read_pickle(os.path.join(W, 'clean.pkl'))
same = len(clean) == len(df) and (clean['_n'].values == df['_n'].values).all() and all(int(clean[s].sum()) == int(df[s].sum()) for s in LAB)
out = {'steps': steps, 'removed_rows': int(len(rec)), 'matches_step1_clean_set': bool(same)}
json.dump(out, open(os.path.join(OUT, 'cleaning_record.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
