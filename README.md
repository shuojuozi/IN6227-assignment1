# IN6227 Assignment 1 — Variant 1

Comparison of **Logistic Regression** and **Random Forest** on the provided binary
classification dataset (`label` ∈ {yes, no}), following a full data-mining pipeline:
exploration → cleaning → feature engineering/selection → cross-validated tuning →
held-out test evaluation.

## How to run

```bash
pip install -r requirements.txt
# put train.csv and test.csv in ./data/
python main.py              # full run (~20–30 min on 2 CPU cores)
python main.py --fast       # smaller Random-Forest search, for a quick check
```

All outputs are written to `results/`.

## Pipeline summary

| Step | What is done | Why |
|---|---|---|
| Cleaning | Drop rows with missing **label** (3 train / 1 test); strip whitespace | A missing target cannot be imputed |
| Missing features | Median (numeric) / most-frequent (categorical) imputation, **inside the pipeline** | <0.03% missing; fitting on training folds only prevents leakage |
| Outliers | Kept; `log1p` on right-skewed columns (skew > 1) + standardisation for LR | Values are plausible; trees are insensitive, LR needs reduced leverage |
| Categorical | One-hot encoding; levels with <1% frequency merged into an "infrequent" bucket | Highly dominated columns (e.g. `region` 90% one level) produce many sparse dummies |
| Feature selection | CV ablation: drop redundant `composite_rank` (r≈0.88 with others) / drop near-zero-correlation features | Checks whether removing features helps; only adopted if gain > 1 CV std |
| Tuning | 5-fold stratified CV, ROC-AUC as selection metric; Grid search (LR), Randomised search (RF) | Threshold-independent metric suited to 24/76 class imbalance |
| Threshold | F1-optimal threshold chosen on **out-of-fold training** predictions | Default 0.5 under-predicts the minority class; test set never used for tuning |
| Evaluation | Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, timing; majority-class baseline | Accuracy alone is misleading under imbalance |

## Output files (`results/`)

| File | Content |
|---|---|
| `run_log.txt` | Full log: EDA statistics, decisions, CV and test results |
| `fig1_eda.png` | Class balance, numeric correlation heatmap, feature–label correlation |
| `fig2_roc_pr.png` | ROC and Precision–Recall curves on the test set |
| `fig3_confusion.png` | Confusion matrices at the tuned thresholds |
| `fig4_importance.png` | Permutation importance (test ROC-AUC drop) for both models |
| `test_metrics.csv` | All test-set metrics |
| `cv_search_results.csv` | Every hyper-parameter configuration tried, with CV scores |
| `feature_ablation.csv` | Feature-selection ablation results |
| `best_params.json` | Selected hyper-parameters and thresholds |
| `permutation_importance.csv` | Importance values behind fig 4 |

Random seed is fixed (`RANDOM_STATE = 42`) for reproducibility.
