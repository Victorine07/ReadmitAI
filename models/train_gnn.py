"""Phase 5 training and evaluation script for HeteroGNN, MLP control, and graph ablations.
Run from project root:  python -m models.train_gnn
"""

import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from eval.metrics import (
    basic_metrics,
    bootstrap_ci,
    calibration_plot,
    paired_bootstrap_diff,
    rare_code_buckets,
    subgroup_metrics,
)
from models.gnn import HIERARCHY, VISITS, HeteroGNN, MLP

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "processed" / "d130_graph.pt"
DF_PATH = ROOT / "data" / "processed" / "d130_clean.parquet"
BASELINE_PREDS_PATH = ROOT / "results" / "baselines" / "test_predictions.npz"
OUT_DIR = ROOT / "results" / "gnn"

SEEDS = [42, 43, 44, 45, 46]

# separate epoch budgets. The MLP is cheap and was still improving at 300 epochs
# (best epochs 286-299), so it gets a much larger budget; the GNN peaked around epoch 120.
MAX_EPOCHS = {"gnn": 400, "mlp": 1000}
PATIENCE = 30          # in epochs
EVAL_EVERY = 2         # validate every 2 epochs: ~25% faster, negligible effect on early stopping

# tuning grid over architecture AND optimisation, identical for GNN and MLP
# (same tuning budget for both, so any difference isn't just "more tuning").
# lr/wd pairs: (1e-3, 1e-3) won both previous grids; 3e-3 tests faster learning.
PARAM_GRID: list[dict[str, float]] = [
    {"hidden": hidden, "dropout": dropout, "lr": lr, "weight_decay": wd}
    for hidden in (64, 128)
    for dropout in (0.2, 0.4)
    for lr, wd in ((1e-3, 1e-3), (3e-3, 1e-3))
]
NUM_LAYERS = 3         # fixed: minimum receptive field for chapter -> category -> code -> encounter


def train_one(
    make_model: Callable[[], torch.nn.Module],
    data,
    lr: float,
    weight_decay: float,
    seed: int,
    max_epochs: int = 400,
    patience: int = PATIENCE,
    eval_every: int = EVAL_EVERY,
) -> tuple[torch.nn.Module, float, int]:
    """Train a single seed full-batch on CPU with early stopping on validation AUROC."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    enc = data["encounter"]
    y = enc.y.float()
    train_mask = enc.train_mask
    val_mask = enc.val_mask

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_since_best = 0

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(data)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        is_last = epoch == max_epochs - 1
        if (epoch + 1) % eval_every != 0 and not is_last:
            continue

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(data))
        val_auc = float(roc_auc_score(y[val_mask].numpy(), probs[val_mask].numpy()))

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += eval_every
            if epochs_since_best >= patience:
                break

    assert best_state is not None, "Training failed to record a best checkpoint."
    # flag runs that were still improving when the budget ran out
    if best_epoch >= max_epochs - patience:
        print(f"    WARNING: best epoch {best_epoch} is within {patience} epochs of max_epochs={max_epochs}; "
              f"the model may still have been improving.")
    model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def predict_probs(model: torch.nn.Module, data, mask: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data))[mask].cpu().numpy()
    return probs.astype(np.float64)


def run_multi_seed(
    name: str,
    make_model: Callable[[], torch.nn.Module],
    data,
    config: dict[str, float],
    max_epochs: int,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train across SEEDS, average test probability predictions, and compute full metrics."""
    enc = data["encounter"]
    test_mask = enc.test_mask
    y_test = df_test["label"].to_numpy()
    groups_test = df_test["patient_nbr"].to_numpy()

    seed_val_aucs = []
    seed_epochs = []
    seed_test_preds = []

    print(f"\nTraining {name} on {len(SEEDS)} seeds with {config} (max_epochs={max_epochs})...")
    for seed in SEEDS:
        model, best_val_auc, best_epoch = train_one(
            make_model, data, config["lr"], config["weight_decay"], seed, max_epochs=max_epochs
        )
        print(f"  seed {seed}: best epoch {best_epoch}, val AUROC {best_val_auc:.4f}")
        seed_val_aucs.append(round(best_val_auc, 4))
        seed_epochs.append(best_epoch)
        seed_test_preds.append(predict_probs(model, data, test_mask))

    mean_test_preds = np.mean(seed_test_preds, axis=0)

    point = basic_metrics(y_test, mean_test_preds)
    auroc_ci = bootstrap_ci(y_test, mean_test_preds, groups_test, roc_auc_score, n_boot=1000, seed=42)
    auprc_ci = bootstrap_ci(
        y_test, mean_test_preds, groups_test, average_precision_score, n_boot=1000, seed=42
    )
    brier_ci = bootstrap_ci(y_test, mean_test_preds, groups_test, brier_score_loss, n_boot=1000, seed=42)

    sub_race = subgroup_metrics(df_test, mean_test_preds, "race")
    sub_gender = subgroup_metrics(df_test, mean_test_preds, "gender")
    sub_age = subgroup_metrics(df_test, mean_test_preds, "age")
    rare_df = rare_code_buckets(df_train, df_test, mean_test_preds)

    print(f"\n=== Test Results (5-seed mean): {name} ===")
    print(f"  Val AUROC per seed: {seed_val_aucs} (mean={np.mean(seed_val_aucs):.4f})")
    print(f"  AUROC: {point['auroc']:.4f} (95% CI: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}])")
    print(f"  AUPRC: {point['auprc']:.4f} (95% CI: [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}])")
    print(f"  Brier: {point['brier']:.4f} (95% CI: [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}])")

    summary = {
        "config": {**config, "num_layers": NUM_LAYERS, "max_epochs": max_epochs, "seeds": SEEDS},
        "val_auroc_per_seed": seed_val_aucs,
        "best_epochs": seed_epochs,
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
    return mean_test_preds, summary


def tune_model(
    label: str,
    make_model_fn: Callable[[dict[str, float]], Callable[[], torch.nn.Module]],
    data,
    param_grid: list[dict[str, float]],
    max_epochs: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Tune hidden / dropout / lr / weight_decay on validation AUROC using seed 42.
    Returns the best config and the full tuning log (saved to the results JSON)."""
    best_config, best_val_auc = param_grid[0], -1.0
    tuning_log: list[dict[str, Any]] = []
    print(f"\nTuning {label} on validation AUROC ({len(param_grid)} configs)...")
    for i, cfg in enumerate(param_grid, start=1):
        _, val_auc, best_ep = train_one(
            make_model_fn(cfg), data, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            seed=42, max_epochs=max_epochs,
        )
        tuning_log.append({**cfg, "best_epoch": best_ep, "val_auroc": round(val_auc, 4)})
        print(f"  [{label} {i}/{len(param_grid)}] {cfg} (ep={best_ep}) -> val AUROC={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_config = cfg
    print(f"  -> Selected {label}: {best_config} (val AUROC={best_val_auc:.4f})")
    return best_config, tuning_log


def main() -> None:
    device = torch.device("cpu")
    assert GRAPH_PATH.exists(), f"Graph not found at {GRAPH_PATH}. Run `python -m features.graph` first."
    assert DF_PATH.exists(), f"Cleaned dataframe not found at {DF_PATH}."

    data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    # Clip standardized encounter features at +/-5 SD for neural network stability
    data["encounter"].x = data["encounter"].x.clamp(-5.0, 5.0)

    df = pd.read_parquet(DF_PATH)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    all_edge_types = list(data.edge_types)
    no_hier_edges = [et for et in all_edge_types if et not in HIERARCHY]
    no_visits_edges = [et for et in all_edge_types if et not in VISITS]

    # model factories take a config dict, so tuned hidden size and dropout are
    # carried through to the multi-seed runs AND the ablations.
    def gnn_factory(edge_types):
        return lambda cfg: (lambda: HeteroGNN(
            data, hidden=int(cfg["hidden"]), num_layers=NUM_LAYERS,
            dropout=cfg["dropout"], edge_types=edge_types,
        ))

    def mlp_factory(cfg):
        return lambda: MLP(data, hidden=int(cfg["hidden"]), num_layers=NUM_LAYERS, dropout=cfg["dropout"])

    # 1. Tune and evaluate HeteroGNN (full graph)
    gnn_cfg, gnn_tuning_log = tune_model(
        "HeteroGNN", gnn_factory(all_edge_types), data, PARAM_GRID, MAX_EPOCHS["gnn"]
    )
    preds_gnn, gnn_summary = run_multi_seed(
        "HeteroGNN (full graph)", gnn_factory(all_edge_types)(gnn_cfg), data,
        gnn_cfg, MAX_EPOCHS["gnn"], df_train, df_test,
    )

    # 2. Tune and evaluate no-graph MLP control (same grid, larger epoch budget)
    mlp_cfg, mlp_tuning_log = tune_model(
        "MLP (no graph)", mlp_factory, data, PARAM_GRID, MAX_EPOCHS["mlp"]
    )
    preds_mlp, mlp_summary = run_multi_seed(
        "MLP (no graph control)", mlp_factory(mlp_cfg), data,
        mlp_cfg, MAX_EPOCHS["mlp"], df_train, df_test,
    )

    # 3. Structural ablations: same tuned config as the full GNN, only the edge types change
    preds_no_hier, no_hier_summary = run_multi_seed(
        "HeteroGNN (no hierarchy)", gnn_factory(no_hier_edges)(gnn_cfg), data,
        gnn_cfg, MAX_EPOCHS["gnn"], df_train, df_test,
    )
    preds_no_visits, no_visits_summary = run_multi_seed(
        "HeteroGNN (no past visits)", gnn_factory(no_visits_edges)(gnn_cfg), data,
        gnn_cfg, MAX_EPOCHS["gnn"], df_train, df_test,
    )

    y_test = df_test["label"].to_numpy()
    groups_test = df_test["patient_nbr"].to_numpy()

    def _compare(p_a: np.ndarray, p_b: np.ndarray) -> dict[str, Any]:
        return {
            "auroc": paired_bootstrap_diff(y_test, p_a, p_b, groups_test, roc_auc_score),
            "auprc": paired_bootstrap_diff(y_test, p_a, p_b, groups_test, average_precision_score),
            "brier": paired_bootstrap_diff(y_test, p_a, p_b, groups_test, brier_score_loss),
        }

    paired_comparisons: dict[str, Any] = {
        "hetero_gnn_minus_mlp": _compare(preds_gnn, preds_mlp),
        "hetero_gnn_minus_no_hierarchy": _compare(preds_gnn, preds_no_hier),
        "hetero_gnn_minus_no_visits": _compare(preds_gnn, preds_no_visits),
    }

    npz_payload: dict[str, Any] = {
        "y_test": y_test,
        "patient_nbr": groups_test,
        "encounter_id": df_test["encounter_id"].to_numpy(),
        "hetero_gnn": preds_gnn,
        "mlp": preds_mlp,
        "hetero_gnn_no_hierarchy": preds_no_hier,
        "hetero_gnn_no_visits": preds_no_visits,
    }

    cal_curves: dict[str, np.ndarray] = {
        "HeteroGNN": preds_gnn,
        "MLP (no graph)": preds_mlp,
        "HeteroGNN (no hierarchy)": preds_no_hier,
        "HeteroGNN (no visits)": preds_no_visits,
    }

    if BASELINE_PREDS_PATH.exists():
        base_npz = np.load(BASELINE_PREDS_PATH)
        # paired comparisons are only valid if both prediction vectors refer to the
        # same encounters in the same order.
        assert np.array_equal(base_npz["encounter_id"], df_test["encounter_id"].to_numpy()), (
            "Baseline predictions are not aligned with the GNN test set (encounter_id order differs)!"
        )
        assert np.array_equal(base_npz["y_test"], y_test), "Baseline test labels differ from GNN test labels!"
        if "xgb" in base_npz:
            preds_xgb = base_npz["xgb"]
            paired_comparisons["hetero_gnn_minus_xgboost"] = _compare(preds_gnn, preds_xgb)
            npz_payload["xgb"] = preds_xgb
            cal_curves["XGBoost"] = preds_xgb

    print("\n=== Paired Bootstrap Comparisons (5-seed mean predictions) ===")
    print(json.dumps(paired_comparisons, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "feature_clip_range": [-5.0, 5.0],
        "device": str(device),
        "training": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "eval_every": EVAL_EVERY},
        "tuning": {"param_grid_size": len(PARAM_GRID), "hetero_gnn": gnn_tuning_log, "mlp": mlp_tuning_log},
        "models": {
            "hetero_gnn": gnn_summary,
            "mlp": mlp_summary,
            "hetero_gnn_no_hierarchy": no_hier_summary,
            "hetero_gnn_no_visits": no_visits_summary,
        },
        "paired_comparisons": paired_comparisons,
    }

    with open(OUT_DIR / "gnn_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    np.savez(OUT_DIR / "test_predictions.npz", **npz_payload)
    calibration_plot(y_test, cal_curves, OUT_DIR / "calibration_curve.png", max_prob=0.35)

    print(f"\nSaved GNN results, predictions, and calibration plot to {OUT_DIR}")


if __name__ == "__main__":
    main()