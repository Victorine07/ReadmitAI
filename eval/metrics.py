"""Evaluation metrics, cluster bootstrap CIs, paired comparisons, rare-code analysis, SHAP, and calibration."""

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import xgboost as xgb

from features.icd9 import category

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]


def _patient_row_index(groups: np.ndarray) -> list:
    """List of row-position arrays, one per patient (for cluster bootstrap)."""
    group_df = pd.DataFrame({"group": np.asarray(groups), "idx": np.arange(len(groups))})
    return [grp["idx"].to_numpy() for _, grp in group_df.groupby("group", sort=False)]


def basic_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Compute primary discrimination and calibration metrics."""
    y_arr = np.asarray(y)
    p_arr = np.asarray(p)
    return {
        "auroc": float(roc_auc_score(y_arr, p_arr)),
        "auprc": float(average_precision_score(y_arr, p_arr)),
        "brier": float(brier_score_loss(y_arr, p_arr)),
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Patient-level cluster bootstrap 95% confidence interval."""
    y_arr = np.asarray(y)
    p_arr = np.asarray(p)
    g_arr = np.asarray(groups)

    patient_indices = _patient_row_index(g_arr)
    n_patients = len(patient_indices)

    rng = np.random.default_rng(seed)
    boot_scores = []

    for _ in range(n_boot):
        sampled_patients = rng.integers(0, n_patients, size=n_patients)
        row_idx = np.concatenate([patient_indices[i] for i in sampled_patients])
        y_b = y_arr[row_idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_scores.append(metric(y_b, p_arr[row_idx]))

    low, high = np.percentile(boot_scores, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Patient-level paired bootstrap of metric(A) - metric(B)."""
    y_arr, a, b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    patient_indices = _patient_row_index(groups)
    n_patients = len(patient_indices)
    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_boot):
        sampled = rng.integers(0, n_patients, size=n_patients)
        row_idx = np.concatenate([patient_indices[i] for i in sampled])
        y_b = y_arr[row_idx]
        if len(np.unique(y_b)) < 2:
            continue
        diffs.append(metric(y_b, a[row_idx]) - metric(y_b, b[row_idx]))

    diffs_arr = np.asarray(diffs)
    low, high = np.percentile(diffs_arr, [2.5, 97.5])
    return {
        "diff": float(metric(y_arr, a) - metric(y_arr, b)),
        "ci95": [float(low), float(high)],
        "share_diff_le_0": float((diffs_arr <= 0).mean()),
    }


def subgroup_metrics(
    df_test: pd.DataFrame, p: np.ndarray, col: str, min_n: int = 100
) -> pd.DataFrame:
    """Evaluate AUROC and AUPRC across demographic or clinical subgroups."""
    work = df_test[[col, "label"]].copy()
    work[col] = work[col].fillna("missing").astype(str)
    work["pred"] = np.asarray(p)

    rows = []
    for val, grp in work.groupby(col, sort=True):
        n = len(grp)
        if n < min_n:
            continue
        n_pos = int(grp["label"].sum())
        has_both_classes = grp["label"].nunique() == 2
        rows.append(
            {
                "subgroup_col": col,
                "subgroup": val,
                "n": n,
                "positives": n_pos,
                "pos_rate": round(n_pos / n, 4),
                "auroc": float(roc_auc_score(grp["label"], grp["pred"])) if has_both_classes else np.nan,
                "auprc": float(average_precision_score(grp["label"], grp["pred"])) if n_pos > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def rare_code_buckets(
    df_train: pd.DataFrame, df_test: pd.DataFrame, p: np.ndarray, min_positives: int = 30
) -> pd.DataFrame:
    """AUROC by the train frequency of each test encounter's RAREST diagnosis category."""
    # 1. Frequency of each 3-character category across all 3 diagnosis slots in TRAIN only
    train_counts = pd.concat([df_train[c].map(category) for c in DIAG_COLS]).value_counts()

    # 2. Per-slot train frequency in TEST; keep missing diagnosis slots as NaN so they are ignored
    per_slot = pd.concat(
        [
            df_test[c].map(category).map(train_counts).fillna(0).where(df_test[c].notna())
            for c in DIAG_COLS
        ],
        axis=1,
    )
    rarest = per_slot.min(axis=1)

    # 3. Bucket rarest category frequency; "<20" aligns with CountVectorizer min_df=20
    buckets = pd.cut(
        rarest,
        bins=[-1, 19, 100, 1000, np.inf],
        labels=["<20 (dropped by featurizer)", "20-100", "101-1000", ">1000"],
    )

    work = pd.DataFrame(
        {
            "bucket": buckets,
            "label": df_test["label"].to_numpy(),
            "pred": np.asarray(p),
        }
    )

    bucket_order = ["<20 (dropped by featurizer)", "20-100", "101-1000", ">1000"]
    rows = []
    for b in bucket_order:
        grp = work[work["bucket"] == b]
        n = len(grp)
        n_pos = int(grp["label"].sum())
        valid_auroc = n_pos >= min_positives and grp["label"].nunique() == 2
        rows.append(
            {
                "rarest_cat_train_freq_bucket": b,
                "n": n,
                "positives": n_pos,
                "pos_rate": round(n_pos / n, 4) if n > 0 else np.nan,
                "auroc": float(roc_auc_score(grp["label"], grp["pred"])) if valid_auroc else np.nan,
            }
        )

    return pd.DataFrame(rows)


def calibration_plot(
    y: np.ndarray, p_dict: Dict[str, np.ndarray], path: Path, max_prob: float = 0.35
) -> None:
    """Save reliability diagram comparing predicted vs empirical probabilities."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    for name, probs in p_dict.items():
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label=name)

    ax.set_xlim(0, max_prob)
    ax.set_ylim(0, max_prob)
    ax.set_title("Test Set Calibration Curve (Quantile Bins)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Readmission Rate")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shap_summary(
    model,
    X_sample,
    feature_names,
    path: Path,
    n_sample: int = 2000,
    seed: int = 42,
    max_display: int = 20,
) -> pd.Series:
    """Compute XGBoost native SHAP values on sparse input, save summary plot, and return top mean |SHAP|."""
    import shap

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X_sample.shape[0], size=min(n_sample, X_sample.shape[0]), replace=False))
    X_s = X_sample[idx]

    best_iter = getattr(model, "best_iteration", None)
    iter_range = (0, best_iter + 1) if best_iter is not None and best_iter >= 0 else (0, 0)

    contribs = model.get_booster().predict(
        xgb.DMatrix(X_s),
        pred_contribs=True,
        iteration_range=iter_range,
    )
    vals = contribs[:, :-1]

    # Additivity guard: SHAP contributions + bias must equal raw margin predictions
    margin = model.predict(X_s, output_margin=True)
    assert np.allclose(contribs.sum(axis=1), margin, atol=1e-3), "SHAP additivity check failed"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X_plot = X_s.toarray() if sparse.issparse(X_s) else X_s
    shap.summary_plot(
        vals,
        X_plot,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs = pd.Series(np.abs(vals).mean(axis=0), index=feature_names)
    return mean_abs.sort_values(ascending=False).head(max_display)