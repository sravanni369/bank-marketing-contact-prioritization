import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/bank-full.csv', sep=';')
numeric, categorical = ['balance','pdays','previous'], ['housing','loan','poutcome']
features = numeric + categorical
split = int(len(df)*0.70); train, test = df.iloc[:split], df.iloc[split:]
# 1. leakage: identical feature rows across the split
key = df[features].astype(str).agg('|'.join, axis=1)
tr, te = set(key.iloc[:split]), set(key.iloc[split:])
print('distinct feature rows: train', len(tr), 'test', len(te), 'shared', len(tr & te), 'test rows with a train twin', int(key.iloc[split:].isin(tr).sum()))
# 2. chronological order of the file: month sequence by block
print('month at rows 0, split-1, split, last:', df.month.iloc[[0, split-1, split, len(df)-1]].tolist())
def fit_eval(train, test, fraction=0.20):
    y, truth = train.y.eq('yes').to_numpy(), test.y.eq('yes').to_numpy()
    prep = ColumnTransformer([('n', StandardScaler(), numeric), ('c', OneHotEncoder(handle_unknown='ignore'), categorical)])
    xtr, xte = prep.fit_transform(train[features]), prep.transform(test[features])
    k = max(1, int(len(test)*fraction)); out = {}
    for name, w in [('unweighted', np.ones(len(y))), ('book_weighted', np.where(y, 1/y.mean(), 1))]:
        m = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=2000).fit(xtr, y, sample_weight=w)
        s = m.predict_proba(xte)[:, 1]
        hits = int(truth[np.argsort(-s, kind='stable')[:k]].sum())
        out[name] = dict(hits=hits, auc=round(roc_auc_score(truth, s), 4), ap=round(average_precision_score(truth, s), 4))
    out['k'] = k; out['random'] = round(k*truth.mean(), 1); out['majority_acc'] = round(1-truth.mean(), 4)
    return out
print('chronological split:', fit_eval(train, test))
# 3. stratified shuffled splits, 10 seeds
rows = []
for seed in range(10):
    tr_, te_ = train_test_split(df, test_size=0.30, stratify=df.y, random_state=seed)
    r = fit_eval(tr_, te_); rows.append((seed, r['unweighted']['hits'], r['book_weighted']['hits'], r['unweighted']['auc'], r['book_weighted']['auc'], r['random']))
d = pd.DataFrame(rows, columns=['seed','hits_unw','hits_wt','auc_unw','auc_wt','random'])
d['diff'] = d.hits_wt - d.hits_unw
print(d.to_string(index=False))
print('diff mean %.1f std %.1f; auc_unw mean %.4f auc_wt mean %.4f' % (d['diff'].mean(), d['diff'].std(), d.auc_unw.mean(), d.auc_wt.mean()))
# 4. book's own C: 1e42 (effectively unpenalised) vs C=1.0 on the chronological split
y, truth = train.y.eq('yes').to_numpy(), test.y.eq('yes').to_numpy()
prep = ColumnTransformer([('n', StandardScaler(), numeric), ('c', OneHotEncoder(handle_unknown='ignore'), categorical)])
xtr, xte = prep.fit_transform(train[features]), prep.transform(test[features]); k = int(len(test)*0.2)
for C in (1.0, 1e42):
    for name, w in [('unweighted', np.ones(len(y))), ('book_weighted', np.where(y, 1/y.mean(), 1))]:
        m = LogisticRegression(C=C, solver='liblinear', random_state=42, max_iter=2000).fit(xtr, y, sample_weight=w)
        s = m.predict_proba(xte)[:, 1]; print('C=%g %-14s hits=%d calls@0.5=%d' % (C, name, truth[np.argsort(-s, kind='stable')[:k]].sum(), (s>=0.5).sum()))
