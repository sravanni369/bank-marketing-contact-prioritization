# Up-weighting the rare class on a real bank: it moved the intercept, not the call list

**A 50-line Python run of the sample-weighting recipe from Bruce, Bruce and Gedeck's
*Practical Statistics for Data Scientists* (2nd ed., p. 232), applied to 45,211 real
telemarketing records where a team can contact only 20% of customers.** Weighting positives
by `1 / mean(y)` changed the number of buyers found inside that budget by **+4 of 2,712 calls**
(1,478 to 1,482). Three things the audit added before publishing:

- across ten stratified reshuffles the difference runs from **−5 to +17** (mean +1.4,
  sd 6.9), and a 500-resample bootstrap of the holdout puts it at −8 to +16, so +4 is noise;
- the two models rank customers almost identically (rank correlation 0.996, 2,645 of the
  2,712 top picks shared) and AUC is 0.7357 against 0.7356: weighting shifts the intercept
  from −1.055 to +0.196, which changes who crosses a 0.5 threshold, not who is at the top;
- with the book's own `C=1e42` instead of `C=1.0`, the sign flips (1,483 unweighted, 1,482
  weighted).

This is a retrospective experiment on public data, not a measured campaign result.

## Business question

A bank's outbound team can phone one customer in five. The book says that when the positive
class is rare you should up-weight it before fitting a classifier. Does that change who gets
on the call list, and does it find more subscribers inside the same budget?

Capacity here means **20% of the holdout rows**, an assumed constraint, not the bank's real
one. The outcome is the historical `y` (term-deposit subscription), not incremental sales.

## Book to implementation

Source: Bruce, P., Bruce, A., & Gedeck, P. (2020). *Practical Statistics for Data
Scientists*, 2nd ed. O'Reilly. Chapter 5, "Oversampling and Up/Down Weighting", printed
p. 232 (local PDF p. 250).

| Printed page | What the book says | What the code does |
|---|---|---|
| 232 | Weight the rare class by `1/p`, the others by 1, and pass `sample_weight` to `LogisticRegression.fit` | `np.where(y, 1 / y.mean(), 1)` as `sample_weight`, `solver='liblinear'` as in the book |
| 232 | Reports the effect as the fraction predicted positive at the default threshold rising from under 1% to about 50% | Reports that too (`threshold_calls`), but scores the models on top-k hits at a fixed budget, which is what a contact team uses |
| 232 | `C=1e42`, effectively unpenalised | `C=1.0`; the deviation is measured in `verify.py` |

The book's listing is on loan data and predicts the default class. The dataset, features,
split and evaluation here are all different; the weighting recipe is what is carried over.

## Dataset

[UCI Bank Marketing](https://doi.org/10.24432/C5K306), Moro, Rita and Cortez (2012), CC BY 4.0.
`data/bank-full.csv`, 45,211 rows × 17 columns, unmodified; 5,289 `yes` (11.7%). Download
URL and SHA-256 are in `data/provenance.json`. Features used: `balance`, `pdays`,
`previous` (standardised) and `housing`, `loan`, `poutcome` (one-hot). `duration` is
excluded because it is only known after the call.

## Evaluation

The file is in date order (May 2008 to November 2010). The first 70% of rows (31,647) train,
the last 30% (13,564) test. **The positive rate is 5.8% in training and 25.4% in the
holdout**: the later period was a different campaign regime, with `poutcome == 'success'`
at 0.26% of training rows and 10.5% of test rows. Every absolute number below is carried by
that holdout prevalence; the stratified table is the like-for-like comparison.

### Same 20% budget, chronological holdout (`analysis.py`)

| Model | Calls | Hits | Precision | Recall | AUC-ROC | Calls at 0.5 threshold |
|---|---:|---:|---:|---:|---:|---:|
| Random at the budget (baseline) | 2,712 | 689.6 | 0.254 | 0.200 | 0.500 | – |
| Unweighted logistic regression | 2,712 | 1,478 | 0.545 | 0.429 | 0.7357 | 1 |
| Book-weighted (`1/mean(y)`) | 2,712 | 1,482 | 0.546 | 0.430 | 0.7356 | 5,702 |

Majority-class accuracy on the holdout is 0.746. The single call at the 0.5 threshold for
the unweighted model is the imbalance effect the book describes, not a failed fit: with a
5.8% base rate its maximum test score is 0.668. Both fits converge in 8 and 7 iterations.

### Ten stratified shuffled 70/30 splits (`verify.py`)

| Seed | Unweighted hits | Weighted hits | Difference | AUC unweighted | AUC weighted | Random |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 727 | 735 | +8 | 0.7115 | 0.7115 | 317.3 |
| 1 | 721 | 738 | +17 | 0.7002 | 0.7014 | 317.3 |
| 2 | 701 | 705 | +4 | 0.6962 | 0.6970 | 317.3 |
| 3 | 683 | 680 | −3 | 0.6828 | 0.6821 | 317.3 |
| 4 | 722 | 724 | +2 | 0.6969 | 0.6969 | 317.3 |
| 5 | 744 | 739 | −5 | 0.7048 | 0.7039 | 317.3 |
| 6 | 741 | 739 | −2 | 0.7087 | 0.7089 | 317.3 |
| 7 | 738 | 733 | −5 | 0.7049 | 0.7040 | 317.3 |
| 8 | 745 | 746 | +1 | 0.7217 | 0.7212 | 317.3 |
| 9 | 735 | 732 | −3 | 0.7094 | 0.7096 | 317.3 |

Difference mean +1.4, sd 6.9. Precision 26.8% against an 11.7% prevalence. Lift over random is
2.29× here and 2.14× on the chronological holdout; the absolute hit count is not comparable.

### The book's regularisation

| C | Unweighted hits | Weighted hits |
|---:|---:|---:|
| 1.0 (this script) | 1,478 | 1,482 |
| 1e42 (the book) | 1,483 | 1,482 |

## What the audit found

`evaluation-auditor` verdict: the conclusion may be published; the headline numbers may not
stand alone. Checks and outcomes:

- No duplicate rows on all 17 columns. 5,768 of 13,564 test rows share their six-feature
  vector with a training row because `pdays` and `previous` are mostly −1 and 0 and the
  three categoricals have 2, 2 and 4 levels; the training label rate of a shared vector has
  correlation 0.037 with the test label. Granularity, not leakage.
- No target-derived feature.
- The chronological split is defensible as a forward-in-time test and must be disclosed
  with the 5.8% / 25.4% shift, as above.
- The book weight itself depends on the split: `1/mean(y)` is 17.2 on the chronological
  training set and 8.5 on a stratified one.

## Honest limits

- Outcomes are historical subscriptions during a campaign, not incremental sales or
  realised savings from prioritisation.
- The 20% capacity is an assumption.
- Six coarse features, no calibration step, no tuning of `C`; the point is the book's
  recipe, not the best model for this data.
- The holdout is one later period. The stratified table is a sensitivity check, not a
  second dataset.

## Reproduce

```
pip install -r requirements.txt
python analysis.py     # writes results/predictions.csv and prints the JSON in run1.txt
python verify.py       # leakage count, stratified seeds, book C; prints verify_output.txt
```

Python 3.13.5, NumPy 2.2.6, pandas 2.3.1, scikit-learn 1.8.0, SciPy 1.16.3, Windows 11.
Three runs of `analysis.py` on 2026-09-16 were byte-identical. The same numbers were first
obtained on scikit-learn 1.7.2. `audit_a.py` and `audit_b.py` are the auditor's own checks.

## Files

| File | What it is |
|---|---|
| `analysis.py` | The 50-line script: chronological split, two models, top-k scoring |
| `verify.py` | Leakage count, ten stratified seeds, AUC, book's `C` |
| `audit_a.py`, `audit_b.py` | Evaluation-auditor scripts (bootstrap, rank overlap, decile rates) |
| `run1.txt`, `run2.txt`, `run3.txt` | Three identical runs of `analysis.py` |
| `verify_output.txt` | Output of `verify.py` |
| `results/predictions.csv` | Holdout row index, label and both models' scores |
| `data/bank-full.csv`, `data/provenance.json` | The data and where it came from |
