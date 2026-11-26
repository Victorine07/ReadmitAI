"""Phase 3d: Strong non-graph tabular baselines (Logistic Regression, XGBoost, XGBoost - history).
Run from project root: python -m models.baselines
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from eval.metrics import (
    basic_metrics,
    bootstrap_ci,
    calibration_plot,
    paired_bootstrap_diff,
    rare_code_buckets,
    shap_summary,
    subgroup_metrics,
)
from features.tabular import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "d130_clean.parquet"
OUT_DIR = ROOT / "results" / "baselines"
SEED = 42


def tune_logistic_regression(X_train, y_train, X_val, y_val) -> Tuple[LogisticRegression, Dict[str, Any]]:
    best_model = None
    best_val_auroc = -1.0
    best_params = {}

    for c_val in (0.01, 0.1, 1.0):
        clf = LogisticRegression(C=c_val, max_iter=2000, random_state=SEED)
        clf.fit(X_train, y_train)
        val_preds = clf.predict_proba(X_val)[:, 1]
        val_auroc = roc_auc_score(y_val, val_preds)
        print(f"  [LR] C={c_val:<4} -> val AUROC={val_auroc:.4f}")
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model = clf
            best_params = {"C": c_val, "max_iter": 2000, "val_auroc": round(val_auroc, 4)}

    return best_model, best_params


def tune_xgboost(X_train, y_train, X_val, y_val) -> Tuple[XGBClassifier, Dict[str, Any]]:
    best_model = None
    best_val_auroc = -1.0
    best_params = {}

    for max_depth in (3, 4, 6):
        for min_child_weight in (1, 5):
            clf = XGBClassifier(
                tree_method="hist",
                learning_rate=0.05,
                n_estimators=2000,
                early_stopping_rounds=50,
                eval_metric="aucpr",
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
            )
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_preds = clf.predict_proba(X_val)[:, 1]
            val_auroc = roc_auc_score(y_val, val_preds)
            print(
                f"  [XGB] max_depth={max_depth}, min_child_weight={min_child_weight} "
                f"(best_iter={clf.best_iteration}) -> val AUROC={val_auroc:.4f}"
            )
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_model = clf
                best_params = {
                    "max_depth": max_depth,
                    "min_child_weight": min_child_weight,
                    "best_iteration": int(clf.best_iteration),
                    "val_auroc": round(val_auroc, 4),
                }

    return best_model, best_params


def train_xgboost_no_history(
    X_train_nohist, y_train, X_val_nohist, y_val, best_xgb_params: Dict[str, Any]
) -> XGBClassifier:
    clf = XGBClassifier(
        tree_method="hist",
        learning_rate=0.05,
        n_estimators=2000,
        early_stopping_rounds=50,
        eval_metric="aucpr",
        max_depth=best_xgb_params["max_depth"],
        min_child_weight=best_xgb_params["min_child_weight"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(
        X_train_nohist,
        y_train,
        eval_set=[(X_val_nohist, y_val)],
        verbose=False,
    )
    return clf


def evaluate_on_test(
    name: str, preds: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame
) -> Dict[str, Any]:
    y_test = df_test["label"].to_numpy()
    groups_test = df_test["patient_nbr"].to_numpy()

    point = basic_metrics(y_test, preds)
    auroc_ci = bootstrap_ci(y_test, preds, groups_test, roc_auc_score, n_boot=1000, seed=SEED)
    auprc_ci = bootstrap_ci(y_test, preds, groups_test, average_precision_score, n_boot=1000, seed=SEED)
    brier_ci = bootstrap_ci(y_test, preds, groups_test, brier_score_loss, n_boot=1000, seed=SEED)

    sub_race = subgroup_metrics(df_test, preds, "race")
    sub_gender = subgroup_metrics(df_test, preds, "gender")
    sub_age = subgroup_metrics(df_test, preds, "age")
    rare_df = rare_code_buckets(df_train, df_test, preds)

    print(f"\n=== Test Results: {name} ===")
    print(f"  AUROC: {point['auroc']:.4f} (95% CI: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}])")
    print(f"  AUPRC: {point['auprc']:.4f} (95% CI: [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}])")
    print(f"  Brier: {point['brier']:.4f} (95% CI: [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}])")
    print("\nRarest diagnosis category train-frequency buckets (test):")
    print(rare_df.to_string(index=False))

    return {
        "metrics": {
            "auroc": round(point["auroc"], 4),
            "auroc_95ci": [round(auroc_ci[0], 4), round(auroc_ci[1], 4)],
            "auprc": round(point["auprc"], 4),
            "auprc_95ci": [round(auprc_ci[0], 4), round(auprc_ci[1], 4)],
            "brier": round(point["brier"], 4),
            "brier_95ci": [round(brier_ci[0], 4), round(brier_ci[1], 4)],
        },
        "subgroups": {
            "race": sub_race.to_dict(orient="records"),
            "gender": sub_gender.to_dict(orient="records"),
            "age": sub_age.to_dict(orient="records"),
        },
        "rare_code_buckets": rare_df.to_dict(orient="records"),
    }


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    y_train = df_train["label"].to_numpy()
    y_val = df_val["label"].to_numpy()

    print("Building full feature matrices (include_history=True)...")
    X_train, X_val, X_test, feat_names = build_features(df, include_history=True)
    print(f"Feature shapes -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    print("\nTuning Logistic Regression on val...")
    lr_model, lr_params = tune_logistic_regression(X_train, y_train, X_val, y_val)

    print("\nTuning XGBoost on val...")
    xgb_model, xgb_params = tune_xgboost(X_train, y_train, X_val, y_val)

    print("\nBuilding ablation feature matrices (include_history=False)...")
    X_train_nh, X_val_nh, X_test_nh, _ = build_features(df, include_history=False)
    xgb_nohist_model = train_xgboost_no_history(X_train_nh, y_train, X_val_nh, y_val, xgb_params)

    # Single evaluation pass on test set
    preds_lr = lr_model.predict_proba(X_test)[:, 1]
    preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
    preds_xgb_nh = xgb_nohist_model.predict_proba(X_test_nh)[:, 1]

    results = {
        "feature_count_with_history": len(feat_names),
        "models": {
            "logistic_regression": {
                "selected_params": lr_params,
                **evaluate_on_test("Logistic Regression", preds_lr, df_train, df_test),
            },
            "xgboost": {
                "selected_params": xgb_params,
                **evaluate_on_test("XGBoost", preds_xgb, df_train, df_test),
            },
            "xgboost_no_history": {
                "selected_params": {k: v for k, v in xgb_params.items() if k != "val_auroc"},
                **evaluate_on_test("XGBoost (no history)", preds_xgb_nh, df_train, df_test),
            },
        },
    }

    y_test = df_test["label"].to_numpy()
    groups_test = df_test["patient_nbr"].to_numpy()

    def _compare(p_a, p_b):
        return {
            "auroc": paired_bootstrap_diff(y_test, p_a, p_b, groups_test, roc_auc_score),
            "auprc": paired_bootstrap_diff(y_test, p_a, p_b, groups_test, average_precision_score),
        }

    results["paired_comparisons"] = {
        "xgboost_minus_lr": _compare(preds_xgb, preds_lr),
        "xgboost_minus_xgboost_no_history": _compare(preds_xgb, preds_xgb_nh),
    }
    print("\nPaired comparisons:", json.dumps(results["paired_comparisons"], indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    top_shap = shap_summary(xgb_model, X_test, feat_names, OUT_DIR / "shap_summary.png")
    results["shap_top_features"] = top_shap.round(5).to_dict()
    print("\nTop SHAP features:\n", top_shap.to_string())

    results_path = OUT_DIR / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    np.savez(
        OUT_DIR / "test_predictions.npz",
        y_test=y_test,
        patient_nbr=groups_test,
        encounter_id=df_test["encounter_id"].to_numpy(),
        lr=preds_lr,
        xgb=preds_xgb,
        xgb_no_history=preds_xgb_nh,
    )

    calibration_plot(
        y_test,
        {
            "Logistic Regression": preds_lr,
            "XGBoost": preds_xgb,
            "XGBoost (no history)": preds_xgb_nh,
        },
        OUT_DIR / "calibration_curve.png",
    )
    print(f"\nSaved baseline metrics, paired comparisons, SHAP summary, and calibration plot to {OUT_DIR}")


if __name__ == "__main__":
    main()