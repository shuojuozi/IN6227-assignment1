"""
Generate the two column-width figures used in the PDF report.

Run AFTER main.py (it reads results/best_params.json and
results/permutation_importance.csv, and refits the two tuned models once
to draw the test-set ROC curves; takes ~10 seconds).

    python report_figures.py
"""
import ast
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from main import C_GREY, C_LR, C_RF, INK, build_pipelines, load_and_clean

OUT = Path("results")


def parse(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main():
    X_tr, y_tr, X_te, y_te, num_cols, cat_cols, skewed, _ = load_and_clean(Path("data"), io.StringIO())
    best = json.load(open(OUT / "best_params.json"))
    lr, rf = build_pipelines(num_cols, cat_cols, skewed)
    lr.set_params(**{k: parse(v) for k, v in best["lr_best_params"].items()})
    rf.set_params(**{k: parse(v) for k, v in best["rf_best_params"].items()})

    plt.rcParams.update({"font.size": 8, "font.family": "serif"})

    # ---- Figure 1: ROC curves on the test set -------------------------------
    fig, ax = plt.subplots(figsize=(3.2, 2.15))
    for name, est, col, ls in [("Logistic Regression", lr, C_LR, "-"),
                               ("Random Forest", rf, C_RF, "--")]:
        est.fit(X_tr, y_tr)
        p = est.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, color=col, lw=1.6, ls=ls,
                label=f"{name} (AUC = {roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], color=C_GREY, lw=0.8, ls=":", label="Random guess")
    ax.set_xlabel("False positive rate", fontsize=8)
    ax.set_ylabel("True positive rate", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "report_fig1_roc.png", dpi=300)
    plt.close(fig)

    # ---- Figure 2: permutation importance -----------------------------------
    imp = pd.read_csv(OUT / "permutation_importance.csv", index_col=0)
    imp = imp.sort_values("RandomForest")
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    y = np.arange(len(imp))
    h = 0.38
    ax.barh(y + h / 2, imp["LogisticRegression"], h, color=C_LR, label="Logistic Regression")
    ax.barh(y - h / 2, imp["RandomForest"], h, color=C_RF, label="Random Forest",
            hatch="///", edgecolor="white", linewidth=0)
    ax.set_yticks(y, imp.index, fontsize=6.5)
    ax.set_xlabel("Drop in test ROC-AUC when shuffled", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "report_fig2_importance.png", dpi=300)
    plt.close(fig)

    # ---- LR coefficients of the numeric features (standardised scale) -------
    names = lr["prep"].get_feature_names_out()
    coef = pd.Series(lr["clf"].coef_[0], index=names)
    print("LR coefficients (standardised numeric features):")
    print(coef[[c for c in num_cols]].round(3).to_string())
    print(f"Saved figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
