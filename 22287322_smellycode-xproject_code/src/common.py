import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support
from xgboost import XGBClassifier

LAB = ['God class', 'Data class', 'Long method', 'Feature envy']
SHORT = {'God class': 'GC', 'Data class': 'DC', 'Long method': 'LM', 'Feature envy': 'FE'}
FEAT = ['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length',
        'Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
SEED = 42

def load(W):
    df = pd.read_pickle(W + '/clean.pkl')
    sp = pd.read_pickle(W + '/split_cols.pkl')
    df['partition'] = sp['partition'].values; df['fold'] = sp['fold'].values
    return df

class CorrScreen(BaseEstimator, TransformerMixin):
    """Drops constant metrics and, of any pair with |Spearman rho| >= thr, the later one. Fitted on training data only."""
    def __init__(self, thr=0.99): self.thr = thr
    def fit(self, X, y=None):
        X = pd.DataFrame(np.asarray(X, float))
        keep = [j for j in range(X.shape[1]) if X[j].nunique() > 1]
        rho = X[keep].rank().corr().abs().values
        kept = []
        for a, j in enumerate(keep):
            if all(rho[a, keep.index(k)] < self.thr for k in kept): kept.append(j)
        self.keep_ = kept
        return self
    def transform(self, X): return np.asarray(X, float)[:, self.keep_]

class BalancedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost whose positive-class weight, when requested, is computed from the labels it is fitted on."""
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1, weighted=False):
        self.n_estimators, self.max_depth, self.learning_rate, self.weighted = n_estimators, max_depth, learning_rate, weighted
    def fit(self, X, y):
        y = np.asarray(y); spw = (len(y) - y.sum()) / max(y.sum(), 1) if self.weighted else 1.0
        self.m_ = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
                                scale_pos_weight=spw, tree_method='hist', n_jobs=2, random_state=SEED, eval_metric='logloss', verbosity=0)
        self.m_.fit(X, y); self.classes_ = self.m_.classes_; return self
    def predict_proba(self, X): return self.m_.predict_proba(X)

def make(model, params, weighted):
    cw = 'balanced' if weighted else None
    if model == 'LR':
        return Pipeline([('screen', CorrScreen()), ('log', FunctionTransformer(np.log1p)), ('scale', StandardScaler()),
                         ('clf', LogisticRegression(C=params['C'], class_weight=cw, max_iter=5000))])
    if model == 'RF':
        return Pipeline([('screen', CorrScreen()), ('clf', RandomForestClassifier(n_estimators=300, max_depth=params['max_depth'],
                         min_samples_leaf=params['min_samples_leaf'], class_weight=cw, n_jobs=2, random_state=SEED))])
    if model == 'XGB':
        return Pipeline([('screen', CorrScreen()), ('clf', BalancedXGB(params['n_estimators'], params['max_depth'], params['learning_rate'], weighted))])

GRIDS = {
    'LR': [{'C': c} for c in [0.01, 0.1, 1.0, 10.0]],
    'RF': [{'max_depth': d, 'min_samples_leaf': l} for d in [None, 20] for l in [1, 5]],
    'XGB': [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr} for n in [200, 500] for d in [4, 8] for lr in [0.05, 0.1]],
}

def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    i = int(np.nanargmax(f[:-1])) if len(th) else 0
    return float(th[i]) if len(th) else 0.5, float(f[i])

def scores(y, p, thr):
    yhat = (p >= thr).astype(int)
    P, R, F, _ = precision_recall_fscore_support(y, yhat, average='binary', zero_division=0)
    ap = average_precision_score(y, p) if y.sum() > 0 else np.nan
    return {'P': P, 'R': R, 'F1': F, 'PRAUC': ap}

def eval_mask(df, eval_idx, train_projects):
    """Exclude evaluation samples whose cross-project near-duplicate partner lies in the corresponding training data."""
    proj = df['Project'].values; part = df['nd_partners'].values
    keep = np.ones(len(eval_idx), bool)
    for n, i in enumerate(eval_idx):
        if part[i] and any(proj[j] in train_projects for j in part[i]): keep[n] = False
    return eval_idx[keep]
