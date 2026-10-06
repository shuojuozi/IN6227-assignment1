"""
IN6227 Assignment 1 - Variant 1
Compare two classifiers (Logistic Regression vs Random Forest) on the provided
tabular dataset, following a full data-mining pipeline:

    1. Data exploration & cleaning
    2. Feature engineering / selection
    3. Model training with cross-validated hyper-parameter tuning
    4. Evaluation & comparison on the held-out test set

Usage:
    python main.py                    # expects data/train.csv and data/test.csv
    python main.py --data-dir path/to/data --fast   # quicker search for testing

All outputs (tables, figures, logs) are written to ./results/.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "label"
POSITIVE = "yes"

# Plot colours (blue / orange, colour-blind-safe pair) and text inks
C_LR, C_RF, C_GREY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ebeae6",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop a list of columns. Used for the feature-selection ablation study."""

    def __init__(self, drop=()):
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.drop if c in X.columns])


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")


# --------------------------------------------------------------------------- #
# 1. Exploration & cleaning
# --------------------------------------------------------------------------- #
def load_and_clean(data_dir, fh):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    log("=" * 70 + "\n1. DATA EXPLORATION & CLEANING\n" + "=" * 70, fh)
    log(f"Raw shapes  train={train.shape}  test={test.shape}", fh)

    # Strip stray whitespace in text columns (defensive; also normalises labels)
    for df in (train, test):
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].str.strip()

    num_cols = [c for c in train.select_dtypes("number").columns if c != TARGET]
    cat_cols = [c for c in train.columns if c not in num_cols + [TARGET]]
    log(f"Numeric features ({len(num_cols)}): {num_cols}", fh)
    log(f"Categorical features ({len(cat_cols)}): {cat_cols}", fh)

    # Missing values
    miss = pd.DataFrame({"train_missing": train.isna().sum(),
                         "test_missing": test.isna().sum()})
    log("\nMissing values per column:\n" + miss.to_string(), fh)
    log(f"Rows with any missing feature: train={train.drop(columns=TARGET).isna().any(axis=1).sum()}"
        f"  test={test.drop(columns=TARGET).isna().any(axis=1).sum()}", fh)

    # Duplicates
    log(f"Duplicate rows: train={train.duplicated().sum()}  test={test.duplicated().sum()}", fh)

    # Class balance
    vc = train[TARGET].value_counts(dropna=False)
    log("\nLabel distribution (train):\n" + vc.to_string(), fh)
    log(f"Positive rate (train) = {(train[TARGET] == POSITIVE).mean():.3f}", fh)

    # Numeric summary + outliers (IQR rule, reported not removed)
    desc = train[num_cols].describe(percentiles=[0.01, 0.5, 0.99]).T
    desc["skew"] = train[num_cols].skew()
    q1, q3 = train[num_cols].quantile(0.25), train[num_cols].quantile(0.75)
    iqr = q3 - q1
    desc["n_outliers_IQR"] = ((train[num_cols] < q1 - 1.5 * iqr) |
                              (train[num_cols] > q3 + 1.5 * iqr)).sum()
    log("\nNumeric summary (train):\n" + desc.round(3).to_string(), fh)

    # Categorical cardinality + dominance
    cat_info = pd.DataFrame({
        "n_levels": train[cat_cols].nunique(),
        "top_level": train[cat_cols].mode().iloc[0],
        "top_share": [train[c].value_counts(normalize=True).iloc[0] for c in cat_cols],
        "n_levels_<1%": [(train[c].value_counts(normalize=True) < 0.01).sum() for c in cat_cols],
    })
    log("\nCategorical summary (train):\n" + cat_info.round(3).to_string(), fh)

    # Feature-label association (point-biserial correlation) and redundancy
    y_tmp = (train[TARGET] == POSITIVE).astype(float)
    corr_y = train[num_cols].corrwith(y_tmp).sort_values(key=abs, ascending=False)
    log("\nCorrelation of numeric features with label:\n" + corr_y.round(3).to_string(), fh)
    corr = train[num_cols].corr()
    high = [(a, b, corr.loc[a, b]) for i, a in enumerate(num_cols)
            for b in num_cols[i + 1:] if abs(corr.loc[a, b]) > 0.7]
    log("Highly correlated numeric pairs (|r|>0.7): " +
        ", ".join(f"{a}~{b} r={r:.2f}" for a, b, r in high), fh)

    # ---- Cleaning decisions ----
    n_tr, n_te = len(train), len(test)
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test = test.dropna(subset=[TARGET]).reset_index(drop=True)
    log(f"\nDropped rows with missing label: train {n_tr - len(train)}, test {n_te - len(test)}", fh)
    log("Missing feature values are imputed INSIDE the pipeline (median / most-frequent, "
        "fitted on training folds only).", fh)
    log("Outliers are kept: the IQR rule flags many points simply because the "
        "distributions are wide/long-tailed, values are in a plausible range "
        "(no negatives or impossible values), and removing them would discard "
        "real signal. log1p + scaling limits their leverage in Logistic Regression; "
        "Random Forest splits on ranks and is insensitive to them.", fh)

    X_train, y_train = train.drop(columns=TARGET), (train[TARGET] == POSITIVE).astype(int)
    X_test, y_test = test.drop(columns=TARGET), (test[TARGET] == POSITIVE).astype(int)

    # Skewed, non-negative numeric columns get log1p for the linear model
    skewed = [c for c in num_cols if desc.loc[c, "skew"] > 1 and desc.loc[c, "min"] >= 0]
    log(f"Right-skewed (skew>1) columns to log-transform for LR: {skewed}", fh)

    eda = {"corr": corr, "corr_y": corr_y, "label_counts": vc, "train": train,
           "num_cols": num_cols}
    return X_train, y_train, X_test, y_test, num_cols, cat_cols, skewed, eda


def plot_eda(eda, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1),
                             gridspec_kw={"width_ratios": [0.7, 1.3, 1.2]})
    # (a) class balance
    vc = eda["label_counts"]
    vc = vc[vc.index.notna()]
    ax = axes[0]
    bars = ax.bar([str(i) for i in vc.index], vc.values, color=[C_GREY, C_LR], width=0.6)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / vc.sum():.0%})",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title("(a) Class balance (train)", loc="left", color=INK)
    ax.set_ylim(0, vc.max() * 1.3)
    ax.grid(axis="x", visible=False)

    # (b) correlation heatmap
    ax = axes[1]
    c = eda["corr"]
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [s.replace("_", "\n", 1) for s in c.columns]
    ax.set_xticks(range(len(c)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(c)), [s for s in c.columns], fontsize=7)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(c.values[i, j]) > 0.6 else INK)
    ax.grid(False)
    ax.set_title("(b) Numeric feature correlation", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04)

    # (c) correlation with label
    ax = axes[2]
    cy = eda["corr_y"].sort_values()
    ax.barh(cy.index, cy.values, color=C_LR, height=0.6)
    for i, v in enumerate(cy.values):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_xlim(min(0, cy.min()) - 0.02, cy.max() + 0.07)
    ax.set_title("(c) Correlation with label (yes=1)", loc="left", color=INK)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig1_eda.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Pre-processing pipelines (feature engineering)
# --------------------------------------------------------------------------- #
def make_preprocessor(num_cols, cat_cols, skewed, for_linear):
    ohe = OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist",
                        sparse_output=False)
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", ohe)])
    if for_linear:
        skew_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                              ("scale", StandardScaler())])
        plain = [c for c in num_cols if c not in skewed]
        plain_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())])
        transformers = [("num", plain_pipe, plain), ("skew", skew_pipe, skewed),
                        ("cat", cat_pipe, cat_cols)]
    else:  # trees: no scaling / log needed
        transformers = [("num", SimpleImputer(strategy="median"), num_cols),
                        ("cat", cat_pipe, cat_cols)]

    # make_column_selector is not used so that the ColumnDropper can remove
    # columns first; remainder='drop' + dynamic lists handled via callables.
    def sel(cols):
        return lambda X: [c for c in cols if c in X.columns]

    return ColumnTransformer([(n, p, sel(c)) for n, p, c in transformers],
                             remainder="drop", verbose_feature_names_out=False)


def build_pipelines(num_cols, cat_cols, skewed):
    lr = Pipeline([
        ("select", ColumnDropper()),
        ("prep", make_preprocessor(num_cols, cat_cols, skewed, for_linear=True)),
        ("clf", LogisticRegression(max_iter=5000, tol=1e-4, solver="lbfgs",
                                   random_state=RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("select", ColumnDropper()),
        ("prep", make_preprocessor(num_cols, cat_cols, skewed, for_linear=False)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    return lr, rf


# --------------------------------------------------------------------------- #
# 3. Training & tuning
# --------------------------------------------------------------------------- #
def tune(lr, rf, X, y, cv, fast, fh):
    log("\n" + "=" * 70 + "\n3. MODEL TRAINING & HYPER-PARAMETER TUNING\n" + "=" * 70, fh)
    log("5-fold stratified CV on the training set; selection metric = ROC-AUC "
        "(threshold-independent, robust to 24/76 imbalance). F1 also recorded.", fh)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "avg_precision": "average_precision"}

    lr_grid = {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "clf__class_weight": [None, "balanced"],
    }
    lr_search = GridSearchCV(lr, lr_grid, scoring=scoring, refit="roc_auc", cv=cv,
                             n_jobs=-1, return_train_score=True)
    t = time.time()
    lr_search.fit(X, y)
    lr_time = time.time() - t
    log(f"\n[LR] grid of {len(lr_search.cv_results_['params'])} configs, {lr_time:.1f}s", fh)
    log(f"[LR] best params: {lr_search.best_params_}  CV ROC-AUC={lr_search.best_score_:.4f}", fh)
    log(f"[LR] lbfgs converged in {lr_search.best_estimator_['clf'].n_iter_[0]} iterations "
        f"(stopping: tol=1e-4, max_iter=5000)", fh)

    rf_dist = {
        "clf__n_estimators": [200, 300],
        "clf__max_depth": [None, 10, 15, 20, 30],
        "clf__min_samples_leaf": [1, 2, 5, 10, 20],
        "clf__max_features": ["sqrt", 0.3, 0.5],
        "clf__class_weight": [None, "balanced", "balanced_subsample"],
    }
    n_iter = 6 if fast else 20
    rf_search = RandomizedSearchCV(rf, rf_dist, n_iter=n_iter, scoring=scoring,
                                   refit="roc_auc", cv=cv, n_jobs=1,
                                   random_state=RANDOM_STATE, return_train_score=True)
    t = time.time()
    rf_search.fit(X, y)
    rf_time = time.time() - t
    log(f"\n[RF] random search {n_iter} configs, {rf_time:.1f}s", fh)
    log(f"[RF] best params: {rf_search.best_params_}  CV ROC-AUC={rf_search.best_score_:.4f}", fh)

    rows = []
    for name, s in [("LogisticRegression", lr_search), ("RandomForest", rf_search)]:
        r = pd.DataFrame(s.cv_results_)
        keep = [c for c in r.columns if c.startswith("param_")] + [
            "mean_train_roc_auc", "mean_test_roc_auc", "std_test_roc_auc",
            "mean_test_f1", "mean_test_avg_precision", "mean_fit_time", "rank_test_roc_auc"]
        r = r[keep].copy()
        r.insert(0, "model", name)
        rows.append(r.sort_values("rank_test_roc_auc"))
    pd.concat(rows).to_csv(OUT / "cv_search_results.csv", index=False)
    return lr_search, rf_search


def feature_ablation(lr_best, rf_best, X, y, cv, fh):
    log("\n" + "=" * 70 + "\n2b. FEATURE SELECTION ABLATION (CV on train, tuned params)\n" + "=" * 70, fh)
    variants = {
        "all features": (),
        "drop composite_rank (redundant, r>0.8)": ("composite_rank",),
        "drop load_ratio+index_weight (|r_y|~0)": ("load_ratio", "index_weight"),
    }
    rows = []
    for mname, est in [("LogisticRegression", lr_best), ("RandomForest", rf_best)]:
        for vname, drop in variants.items():
            est.set_params(select__drop=drop)
            r = cross_validate(est, X, y, cv=cv, scoring=["roc_auc", "f1"], n_jobs=1)
            rows.append({"model": mname, "features": vname,
                         "cv_roc_auc": r["test_roc_auc"].mean(),
                         "cv_roc_auc_std": r["test_roc_auc"].std(),
                         "cv_f1": r["test_f1"].mean()})
        est.set_params(select__drop=())
    df = pd.DataFrame(rows)
    log(df.round(4).to_string(index=False), fh)
    df.to_csv(OUT / "feature_ablation.csv", index=False)
    return df


def choose_threshold(est, X, y, cv, fh, name):
    """Pick the F1-maximising decision threshold from OUT-OF-FOLD train predictions
    (the test set is never used for this)."""
    p = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    log(f"[{name}] F1-optimal threshold on OOF train predictions = {thr[i]:.3f} "
        f"(OOF F1={f1[i]:.4f})", fh)
    return float(thr[i])


# --------------------------------------------------------------------------- #
# 4. Evaluation
# --------------------------------------------------------------------------- #
def metrics_row(name, y, p, thr, fit_s=np.nan, pred_s=np.nan):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"model": name, "threshold": round(thr, 3),
            "accuracy": accuracy_score(y, yhat),
            "balanced_acc": balanced_accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
            "pr_auc": average_precision_score(y, p),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "fit_time_s": fit_s, "predict_time_s": pred_s}


def evaluate(models, X_tr, y_tr, X_te, y_te, fh):
    log("\n" + "=" * 70 + "\n4. EVALUATION ON HELD-OUT TEST SET (used once)\n" + "=" * 70, fh)
    rows, probs = [], {}
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    rows.append(metrics_row("Baseline (majority class)", y_te,
                            dummy.predict_proba(X_te)[:, 1], 0.5))
    for name, (est, thr) in models.items():
        t = time.time(); est.fit(X_tr, y_tr); fit_s = time.time() - t
        t = time.time(); p = est.predict_proba(X_te)[:, 1]; pred_s = time.time() - t
        probs[name] = p
        rows.append(metrics_row(f"{name} @0.5", y_te, p, 0.5, fit_s, pred_s))
        rows.append(metrics_row(f"{name} @tuned", y_te, p, thr, fit_s, pred_s))
        # train metrics for over-fitting check
        ptr = est.predict_proba(X_tr)[:, 1]
        log(f"[{name}] train ROC-AUC={roc_auc_score(y_tr, ptr):.4f}  "
            f"test ROC-AUC={roc_auc_score(y_te, p):.4f}", fh)
    df = pd.DataFrame(rows)
    log("\n" + df.round(4).to_string(index=False), fh)
    df.to_csv(OUT / "test_metrics.csv", index=False)
    return df, probs


def plot_curves(y_te, probs, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    colors = {"LogisticRegression": C_LR, "RandomForest": C_RF}
    ax = axes[0]
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, color=colors[n], lw=2, label=f"{n} (AUC={roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=C_GREY, lw=1, label="Random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(a) ROC curve (test)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax = axes[1]
    for n, p in probs.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax.plot(rc, pr, color=colors[n], lw=2,
                label=f"{n} (AP={average_precision_score(y_te, p):.3f})")
    ax.axhline(y_te.mean(), ls="--", color=C_GREY, lw=1, label=f"Prevalence ({y_te.mean():.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("(b) Precision-Recall curve (test)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig2_roc_pr.png")
    plt.close(fig)


def plot_confusion(metrics, out):
    sel = metrics[metrics["model"].str.endswith("@tuned")].reset_index(drop=True)
    fig, axes = plt.subplots(1, len(sel), figsize=(3.3 * len(sel), 2.8))
    for ax, (_, r) in zip(np.atleast_1d(axes), sel.iterrows()):
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["pred no", "pred yes"]); ax.set_yticks([0, 1], ["true no", "true yes"])
        ax.grid(False)
        ax.set_title(f"{r.model} (thr={r.threshold})", loc="left", color=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_confusion.png")
    plt.close(fig)


def plot_importance(models, X_te, y_te, out, fh):
    """Permutation importance on the ORIGINAL columns (comparable across models)."""
    res = {}
    for name, (est, _) in models.items():
        r = permutation_importance(est, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                   random_state=RANDOM_STATE, n_jobs=1)
        res[name] = pd.Series(r.importances_mean, index=X_te.columns)
    imp = pd.DataFrame(res).sort_values("RandomForest")
    log("\nPermutation importance (drop in test ROC-AUC):\n" +
        imp.sort_values("RandomForest", ascending=False).round(4).to_string(), fh)
    imp.to_csv(out / "permutation_importance.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    y = np.arange(len(imp)); h = 0.38
    ax.barh(y + h / 2, imp["LogisticRegression"], h, color=C_LR, label="LogisticRegression")
    ax.barh(y - h / 2, imp["RandomForest"], h, color=C_RF, label="RandomForest")
    ax.set_yticks(y, imp.index, fontsize=7)
    ax.set_xlabel("Mean decrease in test ROC-AUC when feature is shuffled")
    ax.set_title("Permutation feature importance", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_importance.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true", help="smaller RF search for quick testing")
    args = ap.parse_args()
    OUT = Path(args.out); OUT.mkdir(exist_ok=True)
    fh = open(OUT / "run_log.txt", "w")
    t0 = time.time()

    X_tr, y_tr, X_te, y_te, num_cols, cat_cols, skewed, eda = load_and_clean(Path(args.data_dir), fh)
    plot_eda(eda, OUT)

    log("\n" + "=" * 70 + "\n2. FEATURE ENGINEERING\n" + "=" * 70, fh)
    log("- Categorical: most-frequent imputation + one-hot; levels with <1% frequency "
        "merged into an 'infrequent' bucket (OneHotEncoder min_frequency=0.01); "
        "unseen test levels map to the same bucket.", fh)
    log(f"- Numeric (LR): median imputation, log1p on {skewed}, StandardScaler.", fh)
    log("- Numeric (RF): median imputation only (trees are scale-invariant).", fh)
    lr, rf = build_pipelines(num_cols, cat_cols, skewed)
    n_feat = lr["prep"].fit(X_tr).transform(X_tr).shape[1]
    log(f"- Encoded design matrix: {n_feat} columns (from {X_tr.shape[1]} raw features)", fh)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    lr_s, rf_s = tune(lr, rf, X_tr, y_tr, cv, args.fast, fh)
    lr_best, rf_best = lr_s.best_estimator_, rf_s.best_estimator_

    abl = feature_ablation(lr_best, rf_best, X_tr, y_tr, cv, fh)
    # Rule: drop a feature group only if it IMPROVES CV ROC-AUC by more than one
    # CV standard deviation (i.e. a real, not noise-level, gain); else keep all.
    drops = {"all features": (),
             "drop composite_rank (redundant, r>0.8)": ("composite_rank",),
             "drop load_ratio+index_weight (|r_y|~0)": ("load_ratio", "index_weight")}
    for mname, est in [("LogisticRegression", lr_best), ("RandomForest", rf_best)]:
        sub = abl[abl.model == mname].reset_index(drop=True)
        base = sub.iloc[0].cv_roc_auc
        best = sub.iloc[sub.cv_roc_auc.idxmax()]
        base_std = sub.iloc[0].cv_roc_auc_std
        chosen = best.features if best.cv_roc_auc - base > base_std else "all features"
        est.set_params(select__drop=drops[chosen])
        log(f"[{mname}] chosen feature set: {chosen}  (deltas vs all: " +
            ", ".join(f"{r.features}: {r.cv_roc_auc - base:+.4f}"
                      for r in sub.iloc[1:].itertuples()) + f"; CV std={base_std:.4f})", fh)

    log("\nDecision threshold tuning (imbalanced data -> default 0.5 is not F1-optimal):", fh)
    thr_lr = choose_threshold(lr_best, X_tr, y_tr, cv, fh, "LR")
    thr_rf = choose_threshold(rf_best, X_tr, y_tr, cv, fh, "RF")

    models = {"LogisticRegression": (lr_best, thr_lr), "RandomForest": (rf_best, thr_rf)}
    metrics, probs = evaluate(models, X_tr, y_tr, X_te, y_te, fh)
    plot_curves(y_te, probs, OUT)
    plot_confusion(metrics, OUT)
    plot_importance(models, X_te, y_te, OUT, fh)

    json.dump({"lr_best_params": {k: str(v) for k, v in lr_s.best_params_.items()},
               "rf_best_params": {k: str(v) for k, v in rf_s.best_params_.items()},
               "thresholds": {"LR": thr_lr, "RF": thr_rf}},
              open(OUT / "best_params.json", "w"), indent=2)
    log(f"\nTotal runtime {time.time() - t0:.0f}s. Outputs in {OUT.resolve()}", fh)
    fh.close()


if __name__ == "__main__":
    main()
