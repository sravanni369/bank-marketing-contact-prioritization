"""Adapted from Bruce/Bruce/Gedeck, Practical Statistics, 2e, p.232 (PDF250)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
def run(path, fraction=0.20):
    df = pd.read_csv(path, sep=';')
    numeric, categorical = ['balance', 'pdays', 'previous'], ['housing', 'loan', 'poutcome']
    features = numeric + categorical
    if len(df) < 20 or not 0 < fraction <= 1:
        raise ValueError('Need >=20 rows and a capacity fraction in (0,1].')
    if df[features + ['y']].isna().any().any() or not df.y.isin(['yes', 'no']).all():
        raise ValueError('Missing data or invalid target.')
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Numeric inputs must be finite.')
    split = int(len(df) * 0.70)
    train, test = df.iloc[:split], df.iloc[split:]
    y, truth = train.y.eq('yes').to_numpy(), test.y.eq('yes').to_numpy()
    if len(np.unique(y)) != 2 or not truth.any():
        raise ValueError('Training needs both classes; evaluation needs positives.')
    prep = ColumnTransformer([('n', StandardScaler(), numeric),
                              ('c', OneHotEncoder(handle_unknown='ignore'), categorical)])
    xtrain, xtest = prep.fit_transform(train[features]), prep.transform(test[features])
    k, predictions, rows = max(1, int(len(test) * fraction)), {}, []
    for name, weights in [('unweighted', np.ones(len(y))),
                          ('book_weighted', np.where(y, 1 / y.mean(), 1))]:
        model = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=2000)
        model.fit(xtrain, y, sample_weight=weights)
        scores = model.predict_proba(xtest)[:, list(model.classes_).index(True)]
        chosen = np.argsort(-scores, kind='stable')[:k]
        threshold = scores >= 0.5
        hits = int(truth[chosen].sum())
        rows.append(dict(model=name, selected=k, hits=hits, precision=hits/k,
                         recall=hits/int(truth.sum()), threshold_calls=int(threshold.sum()),
                         threshold_hits=int(truth[threshold].sum())))
        predictions[name] = scores
    result = dict(rows=len(df), train=split, test=len(test), test_positives=int(truth.sum()),
                  train_rate=float(y.mean()), test_rate=float(truth.mean()), capacity=k,
                  random_expected_hits=float(k*truth.mean()), models=rows, features=features)
    return result, pd.DataFrame(dict(row=df.index[split:], y=truth.astype(int), **predictions))

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    result, predictions = run(root / 'data/bank-full.csv')
    (root / 'results').mkdir(exist_ok=True)
    predictions.to_csv(root / 'results/predictions.csv', index=False)
    print(json.dumps(result, indent=2))
