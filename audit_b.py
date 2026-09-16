import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
p = r"C:\Users\laksh\AppData\Local\Temp\claude\C--nvidia-notes\fa3ca4e5-d482-439f-8aff-2d2ece0b0be9\scratchpad\bank\data\bank-full.csv"
df = pd.read_csv(p, sep=';')
num, cat = ['balance','pdays','previous'], ['housing','loan','poutcome']; feats=num+cat
def evaluate(d, seed, strat=True):
    tr, te = train_test_split(d, test_size=0.3, random_state=seed, stratify=d.y if strat else None)
    y, truth = tr.y.eq('yes').to_numpy(), te.y.eq('yes').to_numpy()
    prep = ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(handle_unknown='ignore'),cat)])
    Xtr, Xte = prep.fit_transform(tr[feats]), prep.transform(te[feats]); k=int(len(te)*0.2)
    out = dict(seed=seed, n_test=len(te), train_rate=round(y.mean(),4), test_rate=round(truth.mean(),4), k=k,
               random=round(k*truth.mean(),1), majority=round(1-truth.mean(),4))
    for name, w in [('unw', np.ones(len(y))), ('wt', np.where(y, 1/y.mean(), 1))]:
        m = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=2000).fit(Xtr, y, sample_weight=w)
        s = m.predict_proba(Xte)[:,1]
        out[name+'_hits'] = int(truth[np.argsort(-s,kind='stable')[:k]].sum()); out[name+'_auc']=round(roc_auc_score(truth,s),4)
    out['diff'] = out['wt_hits']-out['unw_hits']; return out
print("== stratified shuffled 70/30, raw rows, seeds 0-9 ==")
r = pd.DataFrame([evaluate(df, s) for s in range(10)]); print(r.to_string(index=False))
print("unw hits min/mean/max: %d/%.1f/%d ; wt-unw diff min/mean/max: %d/%.1f/%d ; AUC unw mean %.4f wt mean %.4f" %
      (r.unw_hits.min(), r.unw_hits.mean(), r.unw_hits.max(), r['diff'].min(), r['diff'].mean(), r['diff'].max(), r.unw_auc.mean(), r.wt_auc.mean()))
print("== stratified shuffled 70/30, deduplicated on 6 features + y, seeds 0-9 ==")
dd = df.drop_duplicates(subset=feats+['y']); print("dedup rows:", len(dd), "pos rate %.4f" % dd.y.eq('yes').mean())
r2 = pd.DataFrame([evaluate(dd, s) for s in range(10)]); print(r2.to_string(index=False))
print("unw hits min/mean/max: %d/%.1f/%d ; wt-unw diff min/mean/max: %d/%.1f/%d" %
      (r2.unw_hits.min(), r2.unw_hits.mean(), r2.unw_hits.max(), r2['diff'].min(), r2['diff'].mean(), r2['diff'].max()))
# chronological: positive rate by decile of row order
dec = pd.qcut(np.arange(len(df)), 10, labels=False)
print("pos rate by row-order decile:", df.y.eq('yes').groupby(dec).mean().round(3).tolist())
print("pos rate of last 30%% rows: %.4f ; poutcome=='success' share first 70%% vs last 30%%: %.4f vs %.4f" %
      (df.y.iloc[int(len(df)*0.7):].eq('yes').mean(), df.poutcome.iloc[:int(len(df)*0.7)].eq('success').mean(), df.poutcome.iloc[int(len(df)*0.7):].eq('success').mean()))
