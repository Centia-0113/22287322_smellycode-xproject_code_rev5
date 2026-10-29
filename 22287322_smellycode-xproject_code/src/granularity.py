"""Post hoc step A: identify each sample's granularity from its source code (rule-based, sample-wise, no labels, nothing fitted).
A sample is CLASS-level if, after any leading annotations and modifiers, it starts with a type declaration
(class, interface, enum, @interface or record); otherwise it is METHOD-level (methods and constructors).
Usage: python granularity.py <work dir>   ->  writes <work dir>/level.pkl and prints the distributions."""
import os, sys, re, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
MODS = {'public', 'private', 'protected', 'abstract', 'final', 'static', 'strictfp', 'sealed', 'native',
        'synchronized', 'transient', 'volatile', 'default'}
STR = re.compile(r'"(?:[^"\\]|\\.)*"')

def level(code):
    toks = STR.sub('""', str(code)).split(' ')
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        if t in ('', ';'): i += 1; continue
        if t == '@':                                            # annotation, or an annotation-type declaration
            if i + 1 < n and toks[i + 1] == 'interface': return 'class'
            i += 2
            while i + 1 < n and toks[i] == '.': i += 2          # qualified name, e.g. @ com . google . inject . Singleton
            if i < n and toks[i] == '(':                        # arguments, with nested parentheses
                depth = 0
                while i < n:
                    depth += (toks[i] == '(') - (toks[i] == ')'); i += 1
                    if depth == 0: break
            continue
        if t in MODS: i += 1; continue
        if t == 'non' and toks[i + 1:i + 3] == ['-', 'sealed']: i += 3; continue
        if t in ('class', 'interface', 'enum'): return 'class'
        if t == 'record' and i + 2 < n and toks[i + 2] in ('(', '<'): return 'class'
        return 'method'
    return 'method'

if __name__ == '__main__':
    W = sys.argv[1]
    df = load(W)
    lv = df['_n'].map(level).values
    pd.Series(lv, name='level').to_pickle(W + '/level.pkl')
    out = {'class_level': int((lv == 'class').sum()), 'method_level': int((lv == 'method').sum())}
    print('class-level %d (%.1f%%), method-level %d (%.1f%%)' % (out['class_level'], 100 * out['class_level'] / len(df), out['method_level'], 100 * out['method_level'] / len(df)))
    # consistency with the labels (the rule never looks at them)
    for s in LAB:
        pos = df[s].values == 1
        print('%s positives: class-level %d, method-level %d' % (SHORT[s], int((pos & (lv == 'class')).sum()), int((pos & (lv == 'method')).sum())))
    meth = lv == 'method'
    print('method-level samples without "(" (not a method signature):', int((~df.loc[meth, '_n'].str.contains('(', regex=False)).sum()))
    print('median Logical Lines: class-level %.0f, method-level %.0f' % (df.loc[~meth, 'Logical Lines'].median(), df.loc[meth, 'Logical Lines'].median()))
    rows = []
    for name, m in [('Full dataset', np.ones(len(df), bool)), ('Training', df.partition.values == 'train'), ('Test', df.partition.values == 'test')]:
        r = {'Partition': name}
        c = m & (lv == 'class'); me = m & (lv == 'method')
        r['Class-level'] = int(c.sum())
        for s in ['God class', 'Data class']: r[SHORT[s]] = '%d (%.2f)' % (df[s].values[c].sum(), 100 * df[s].values[c].mean())
        r['Method-level'] = int(me.sum())
        for s in ['Long method', 'Feature envy']: r[SHORT[s]] = '%d (%.2f)' % (df[s].values[me].sum(), 100 * df[s].values[me].mean())
        rows.append(r)
    t = pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(W + '/granularity_table.csv', index=False)
    # where do smell-free classes sit relative to Long Method in size?
    lm_med = df.loc[df['Long method'] == 1, 'Logical Lines'].median()
    free_cls = (lv == 'class') & (df[LAB].sum(axis=1).values == 0)
    print('median Logical Lines of Long Method samples %.0f; smell-free class-level samples at least that long: %.1f%%' % (lm_med, 100 * (df.loc[free_cls, 'Logical Lines'] >= lm_med).mean()))
    out['table'] = rows
    json.dump(out, open(W + '/granularity.json', 'w'), indent=1)
