import numpy as np, pandas as pd, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
p = r"C:\Users\laksh\AppData\Local\Temp\claude\C--nvidia-notes\fa3ca4e5-d482-439f-8aff-2d2ece0b0be9\scratchpad\bank\data\bank-full.csv"
df = pd.read_csv(p, sep=';')
print("sklearn", sklearn.__version__, "pandas", pd.__version__)
print("columns:", list(df.columns))
num, cat = ['balance','pdays','previous'], ['housing','loan','poutcome']
feats = num+cat
split = int(len(df)*0.7); tr, te = df.iloc[:split], df.iloc[split:]
# 1. duplicates on the six features
print("total rows", len(df), "unique feature rows", len(df[feats].drop_duplicates()),
      "dup pct %.1f" % (100*(1-len(df[feats].drop_duplicates())/len(df))))
trkeys = set(map(tuple, tr[feats].astype(str).values))
tekeys = list(map(tuple, te[feats].astype(str).values))
inboth = np.array([k in trkeys for k in tekeys])
print("test rows whose 6-feature tuple appears in train: %d of %d (%.1f%%)" % (inboth.sum(), len(te), 100*inboth.mean()))
# label consistency of leaked keys: does train label predict test label for same key?
trlab = tr.assign(k=list(map(tuple, tr[feats].astype(str).values))).groupby('k').y.apply(lambda s: s.eq('yes').mean())
te2 = te.assign(k=tekeys); te2 = te2[inboth]
te2['trainrate'] = te2.k.map(trlab)
print("among leaked test rows: test pos rate %.3f, mean train pos rate for same key %.3f, corr %.3f" %
      (te2.y.eq('yes').mean(), te2.trainrate.mean(), np.corrcoef(te2.y.eq('yes'), te2.trainrate)[0,1]))
# full-row duplicates (all 17 cols)
print("full-row duplicates (all columns):", df.duplicated().sum())
# 2. feature provenance
print("pdays==-1 share", (df.pdays==-1).mean().round(3), "| poutcome value counts:", df.poutcome.value_counts().to_dict())
print("pos rate by poutcome:", df.groupby('poutcome').y.apply(lambda s: s.eq('yes').mean()).round(3).to_dict())
print("pos rate by pdays==-1:", df.groupby(df.pdays==-1).y.apply(lambda s: s.eq('yes').mean()).round(3).to_dict())
# 3/5. majority + AUC + scores on chronological split
y, truth = tr.y.eq('yes').to_numpy(), te.y.eq('yes').to_numpy()
prep = ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(handle_unknown='ignore'),cat)])
Xtr, Xte = prep.fit_transform(tr[feats]), prep.transform(te[feats])
k = int(len(te)*0.2)
print("test majority-class (no) rate: %.4f ; train 'no' rate %.4f" % (1-truth.mean(), 1-y.mean()))
def fit(w, rs=42):
    m = LogisticRegression(C=1.0, solver='liblinear', random_state=rs, max_iter=2000); m.fit(Xtr, y, sample_weight=w); return m
res = {}
for name, w in [('unw', np.ones(len(y))), ('wt', np.where(y, 1/y.mean(), 1))]:
    m = fit(w); s = m.predict_proba(Xte)[:,1]; res[name]=s
    chosen = np.argsort(-s, kind='stable')[:k]
    print(name, "hits", truth[chosen].sum(), "AUC %.4f" % roc_auc_score(truth, s),
          "max score %.4f" % s.max(), "n>=0.5:", (s>=0.5).sum(), "n_iter", m.n_iter_, "intercept %.3f" % m.intercept_[0])
    print("   coefs:", dict(zip(prep.get_feature_names_out(), m.coef_[0].round(3))))
print("rank corr of two score vectors: %.5f" % pd.Series(res['unw']).corr(pd.Series(res['wt']), method='spearman'))
print("overlap of top-k sets:", len(set(np.argsort(-res['unw'],kind='stable')[:k]) & set(np.argsort(-res['wt'],kind='stable')[:k])), "of", k)
# ties at the cutoff
for name,s in res.items():
    cut = np.sort(s)[::-1][k-1]; print(name, "score at cutoff %.5f, rows tied at cutoff: %d" % (cut, (s==cut).sum()))
# 4a. liblinear random_state noise on the same split
for name, w in [('unw', np.ones(len(y))), ('wt', np.where(y, 1/y.mean(), 1))]:
    hs=[]
    for rs in range(10):
        s = fit(w, rs).predict_proba(Xte)[:,1]; hs.append(int(truth[np.argsort(-s,kind='stable')[:k]].sum()))
    print(name, "hits across liblinear random_state 0-9:", hs)
# 4b. bootstrap of the holdout: distribution of (wt - unw) hits at 20% budget
rng = np.random.default_rng(0); diffs=[]; unws=[]
for b in range(500):
    idx = rng.integers(0, len(te), len(te)); t = truth[idx]; kk=int(len(idx)*0.2)
    h = {n:int(t[np.argsort(-res[n][idx],kind='stable')[:kk]].sum()) for n in res}
    diffs.append(h['wt']-h['unw']); unws.append(h['unw'])
diffs=np.array(diffs); unws=np.array(unws)
print("bootstrap resamples: %d" % len(unws)); print("bootstrap unweighted hits: mean %.1f sd %.1f [2.5,97.5]=%s" % (unws.mean(), unws.std(), np.percentile(unws,[2.5,97.5])))
print("bootstrap (wt-unw) hits: mean %.2f sd %.2f [2.5,97.5]=%s, P(diff<=0)=%.3f" % (diffs.mean(), diffs.std(), np.percentile(diffs,[2.5,97.5]), (diffs<=0).mean()))
