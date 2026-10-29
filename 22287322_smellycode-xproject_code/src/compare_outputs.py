"""Compares a run's outputs with the reported outputs in results/ and lists every difference.
JSON files are compared value by value, CSV files cell by cell and logs line by line, after run times are removed.
Numbers count as identical when they agree to a relative tolerance of 1e-9, and as "agree to 6 decimal places" when
they differ by at most 1e-6 (absolute or relative): multi-threaded Random Forest predictions can change the last
digits of a validation score or permutation importance between runs, far below the 2-3 decimals reported in the paper. Rows of near_dup_project_pairs.csv and lines
of step1.log may come out in a different order when they are tied. Figures are not compared.
Writes <run dir>/reproduction_check.json.
Usage: python src/compare_outputs.py <run dir, e.g. work> <reported dir, e.g. results>"""
import os, re, sys, json, math
import pandas as pd

RUN, REF = sys.argv[1], sys.argv[2]
IGNORE_KEYS = {'seconds', 'time', 'runtime', 'elapsed'}
TIMING = [re.compile(p) for p in (r'\[\s*\d+(\.\d+)?\s*s\]', r'\bin \d+(\.\d+)?\s*s\b', r'done in \d+(\.\d+)?\s*s', r'"seconds":\s*\d+(\.\d+)?,?')]
NUM = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
EXACT, CLOSE = 1e-9, 1e-6


def level(a, b):
    """0 = identical, 1 = agree to 6 decimal places, 2 = different."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b): return 0
    if math.isclose(a, b, rel_tol=EXACT, abs_tol=1e-12): return 0
    if math.isclose(a, b, rel_tol=CLOSE, abs_tol=CLOSE): return 1
    return 2


def cmp_json(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in IGNORE_KEYS: continue
            if k not in a or k not in b: out.append((2, f'{path}/{k}: missing in {"run" if k not in a else "reported"}')); continue
            cmp_json(a[k], b[k], f'{path}/{k}', out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out.append((2, f'{path}: length {len(a)} vs {len(b)}')); return
        for i, (x, y) in enumerate(zip(a, b)): cmp_json(x, y, f'{path}[{i}]', out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        lv = level(float(a), float(b))
        if lv: out.append((lv, f'{path}: {a} vs {b}'))
    elif a != b:
        out.append((2, f'{path}: {str(a)[:60]} vs {str(b)[:60]}'))


def cmp_csv(fa, fb, name, out):
    a, b = pd.read_csv(fa), pd.read_csv(fb)
    if list(a.columns) != list(b.columns): out.append((2, 'columns differ')); return
    if name == 'near_dup_project_pairs.csv':
        a = a.sort_values(list(a.columns)).reset_index(drop=True); b = b.sort_values(list(b.columns)).reset_index(drop=True)
    if a.shape != b.shape: out.append((2, f'shape {a.shape} vs {b.shape}')); return
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_bool_dtype(x) or pd.api.types.is_bool_dtype(y) or not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
            bad = int((x.astype(str) != y.astype(str)).sum())
            if bad: out.append((2, f'column {c}: {bad} cells differ'))
            continue
        lv = max((level(float(p), float(q)) for p, q in zip(x, y)), default=0)
        if lv: out.append((lv, f'column {c}: values differ' + (' by more than 1e-6' if lv == 2 else ' by at most 1e-6')))


def strip_times(line):
    for p in TIMING: line = p.sub('', line)
    return line.rstrip()


def cmp_lines(x, y):
    nx, ny = NUM.findall(x), NUM.findall(y)
    if NUM.sub('#', x) != NUM.sub('#', y) or len(nx) != len(ny): return 2
    return max((level(float(p), float(q)) for p, q in zip(nx, ny)), default=0)


def cmp_log(fa, fb, name, out):
    a = [strip_times(l) for l in open(fa, encoding='utf-8', errors='replace')]
    b = [strip_times(l) for l in open(fb, encoding='utf-8', errors='replace')]
    if len(a) != len(b): out.append((2, f'{len(a)} vs {len(b)} lines')); return
    lv = [cmp_lines(x, y) for x, y in zip(a, b)]
    if name == 'step1.log' and max(lv, default=0) == 2 and sorted(a) == sorted(b):
        out.append((1, 'same lines, tied rows in a different order')); return
    if max(lv, default=0): out.append((max(lv), f'{sum(1 for v in lv if v)} lines differ' + ('' if max(lv) == 2 else ', only by at most 1e-6')))


report, worst = {}, {}
for root, _, files in os.walk(REF):
    for f in sorted(files):
        if f == 'reproduction_check.json': continue   # this script's own report
        rel = os.path.relpath(os.path.join(root, f), REF); fr = os.path.join(RUN, rel); fb = os.path.join(REF, rel)
        if f.endswith('.png'): report[rel] = 'not compared (figure)'; continue
        if not os.path.exists(fr): report[rel] = 'MISSING in run'; worst[rel] = 2; continue
        out = []
        try:
            if f.endswith('.json'): cmp_json(json.load(open(fr)), json.load(open(fb)), '', out)
            elif f.endswith('.csv'): cmp_csv(fr, fb, f, out)
            elif f.endswith('.log'): cmp_log(fr, fb, f, out)
            else: report[rel] = 'not compared'; continue
        except Exception as e:
            out.append((2, f'error: {type(e).__name__}: {e}'))
        w = max((lv for lv, _ in out), default=0); worst[rel] = w
        report[rel] = 'identical' if w == 0 else (('agree to 6 decimal places: ' if w == 1 else 'DIFFERS: ') + '; '.join(m for _, m in out[:6]))
for k, v in sorted(report.items()): print(f'{k:45s} {v}')
n0 = sum(1 for k in worst if worst[k] == 0); n1 = sum(1 for k in worst if worst[k] == 1); n2 = sum(1 for k in worst if worst[k] == 2)
print(f'\n{n0} identical, {n1} agree to 6 decimal places, {n2} different, {len(report) - len(worst)} not compared')
json.dump({'identical': n0, 'agree_to_6_decimal_places': n1, 'different': n2, 'files': report},
          open(os.path.join(RUN, 'reproduction_check.json'), 'w'), indent=1)
sys.exit(1 if n2 else 0)
