"""Independent validation of the granularity rule (post hoc step E).
Each sample is parsed with a Java parser (javalang): unclosed braces are closed, the sample is wrapped in a class body,
and the type of its first member is read off the syntax tree (type declaration -> class-level; method or constructor -> method-level). Agreement with the
rule is reported for ALL samples, by label status, and for the declared type name against the file's Class column.
Usage: python validate_granularity.py <work dir> [n_jobs]"""
import os, sys, json, time, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, LAB, SHORT
import javalang

TYPES = ('ClassDeclaration', 'InterfaceDeclaration', 'EnumDeclaration', 'AnnotationDeclaration')
def balance(code):
    # many snippets end before their closing braces; close them (counted on lexer tokens, so braces in strings are ignored)
    depth = 0
    for t in javalang.tokenizer.tokenize(code):
        if isinstance(t, javalang.tokenizer.Separator):
            if t.value == '{': depth += 1
            elif t.value == '}': depth -= 1
    return code + ' }' * max(depth, 0)

def parse_level(code):
    try:
        tree = javalang.parse.parse('class __W__ { ' + balance(code) + ' }')
        body = tree.types[0].body
        if not body: return ('empty', None)
        m = body[0]; k = type(m).__name__
        if k in TYPES: return ('class', m.name)
        if k in ('MethodDeclaration', 'ConstructorDeclaration'): return ('method', m.name)
        return ('other:' + k, None)
    except Exception as e:
        return ('fail', type(e).__name__)

if __name__ == '__main__':
    W = sys.argv[1]; nj = int(sys.argv[2]) if len(sys.argv) > 2 else 2; t0 = time.time()
    df = load(W); lv = pd.read_pickle(W + '/level.pkl').values
    with Pool(nj) as p: res = p.map(parse_level, df['_n'].tolist(), chunksize=200)
    pk = np.array([r[0] for r in res], dtype=object); nm = np.array([r[1] for r in res], dtype=object)
    out = {'n': int(len(df)), 'seconds': round(time.time() - t0)}
    parsed = np.isin(pk, ['class', 'method'])
    out['parsed'] = int(parsed.sum()); out['failed'] = int((pk == 'fail').sum())
    out['other'] = {k: int(v) for k, v in pd.Series(pk[~parsed]).value_counts().items()}
    agree = parsed & (pk == lv)
    out['agreement_parsed'] = float(agree.sum() / parsed.sum())
    out['disagreements'] = int((parsed & (pk != lv)).sum())
    pos = df[LAB].sum(axis=1).values > 0
    for name, m in [('labelled_smell', pos), ('no_smell_label', ~pos)]:
        mm = m & parsed
        out[f'agreement_{name}'] = {'n_parsed': int(mm.sum()), 'agreement': float((pk[mm] == lv[mm]).mean())}
    for L in ['class', 'method']:
        mm = (lv == L)
        out[f'rule_{L}'] = {'n': int(mm.sum()), 'parsed': int((mm & parsed).sum()), 'parser_agrees': int((mm & parsed & (pk == lv)).sum())}
    # metadata check: the declared type name of class-level samples against the Class column
    cls = parsed & (pk == 'class')
    same = np.array([str(a).lower() == str(b).lower() for a, b in zip(nm[cls], df['Class'].values[cls])])
    out['class_name_matches_Class_column'] = {'n': int(cls.sum()), 'share': float(same.mean())}
    meth = parsed & (pk == 'method')
    same_m = np.array([str(a).lower() == str(b).lower() for a, b in zip(nm[meth], df['Class'].values[meth])])
    out['method_samples_whose_name_equals_Class_column'] = {'n': int(meth.sum()), 'share': float(same_m.mean())}
    dis = np.where(parsed & (pk != lv))[0][:20]
    out['examples_disagree'] = [{'rule': str(lv[i]), 'parser': str(pk[i]), 'code': df['_n'].iat[i][:160]} for i in dis]
    fail_ex = np.where(pk == 'fail')[0][:10]
    out['examples_failed'] = [{'rule': str(lv[i]), 'code': df['_n'].iat[i][:160]} for i in fail_ex]
    out['rule_level_of_failed'] = {k: int(v) for k, v in pd.Series(lv[pk == 'fail']).value_counts().items()}
    print(json.dumps(out, indent=1))
    json.dump(out, open(W + '/granularity_validation.json', 'w'), indent=1)
    pd.DataFrame({'parser_level': pk, 'parser_name': nm}).to_pickle(W + '/granularity_parser.pkl')
