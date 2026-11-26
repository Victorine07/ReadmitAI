"""Phase 4: build the leakage-free heterogeneous graph.
Run from project root:  python -m features.graph
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from features.icd9 import category, chapter
from features.tabular import MED_COLS, build_features

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "d130_clean.parquet"
OUT_PATH = ROOT / "data" / "processed" / "d130_graph.pt"


def _to_edge_index(src: Iterable[int], dst: Iterable[int]) -> torch.Tensor:
    src_arr = np.asarray(list(src), dtype=np.int64)
    dst_arr = np.asarray(list(dst), dtype=np.int64)
    if src_arr.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.vstack([src_arr, dst_arr]), dtype=torch.long)


def encounter_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Phase 3 features (fit on train), WITHOUT 'diag__' columns, as one dense float32 matrix in df row order."""
    X_tr, X_va, X_te, names = build_features(df, include_history=True)
    keep = [i for i, n in enumerate(names) if not n.startswith("diag__")]
    kept_names = [names[i] for i in keep]

    X = np.zeros((len(df), len(keep)), dtype=np.float32)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    X[train_mask] = X_tr[:, keep].toarray().astype(np.float32)
    X[val_mask] = X_va[:, keep].toarray().astype(np.float32)
    X[test_mask] = X_te[:, keep].toarray().astype(np.float32)

    return X, kept_names


def vocab(values: pd.Series) -> dict[str, int]:
    """Sorted unique non-null string values -> integer node index."""
    cleaned = values.dropna().astype(str)
    cleaned = cleaned[cleaned != ""]
    unique_sorted = sorted(cleaned.unique())
    return {val: idx for idx, val in enumerate(unique_sorted)}


def build_graph(df: pd.DataFrame) -> HeteroData:
    df = df.reset_index(drop=True)
    data = HeteroData()

    # --- 1. Encounter nodes ---
    enc_x, _ = encounter_features(df)
    data["encounter"].x = torch.tensor(enc_x, dtype=torch.float32)
    data["encounter"].y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    data["encounter"].train_mask = torch.tensor((df["split"] == "train").to_numpy(), dtype=torch.bool)
    data["encounter"].val_mask = torch.tensor((df["split"] == "val").to_numpy(), dtype=torch.bool)
    data["encounter"].test_mask = torch.tensor((df["split"] == "test").to_numpy(), dtype=torch.bool)
    data["encounter"].encounter_id = torch.tensor(df["encounter_id"].to_numpy(), dtype=torch.long)
    data["encounter"].patient_nbr = torch.tensor(df["patient_nbr"].to_numpy(), dtype=torch.long)

    # --- 2. Vocabulary nodes (counts only; embeddings learned in Phase 5) ---
    all_raw_codes = pd.concat([df[c] for c in DIAG_COLS], ignore_index=True).dropna().astype(str)
    code_idx = vocab(all_raw_codes)

    all_cats = all_raw_codes.map(category).dropna()
    cat_idx = vocab(all_cats)

    all_chaps = all_raw_codes.map(chapter).dropna()
    chap_idx = vocab(all_chaps)

    med_list = [c for c in MED_COLS if c in df.columns]
    med_idx = {m: i for i, m in enumerate(med_list)}

    data["code"].num_nodes = len(code_idx)
    data["category"].num_nodes = len(cat_idx)
    data["chapter"].num_nodes = len(chap_idx)
    data["medication"].num_nodes = len(med_list)

    # --- 3. Directed edges: code -> encounter (primary vs secondary; NO reverse edges) ---
    primary_src, primary_dst = [], []
    for row_pos, raw_val in enumerate(df["diag_1"]):
        if pd.notna(raw_val):
            c_str = str(raw_val)
            if c_str in code_idx:
                primary_src.append(code_idx[c_str])
                primary_dst.append(row_pos)
    data["code", "primary_of", "encounter"].edge_index = _to_edge_index(primary_src, primary_dst)

    sec_src, sec_dst = [], []
    for col in ("diag_2", "diag_3"):
        for row_pos, raw_val in enumerate(df[col]):
            if pd.notna(raw_val):
                c_str = str(raw_val)
                if c_str in code_idx:
                    sec_src.append(code_idx[c_str])
                    sec_dst.append(row_pos)
    data["code", "secondary_of", "encounter"].edge_index = _to_edge_index(sec_src, sec_dst)

    # --- 4. Directed edges: medication -> encounter (status != "No"; NO reverse edges) ---
    med_src, med_dst = [], []
    for med_name, m_id in med_idx.items():
        statuses = df[med_name].fillna("No").astype(str).to_numpy()
        active_rows = np.flatnonzero(statuses != "No")
        med_src.extend([m_id] * len(active_rows))
        med_dst.extend(active_rows.tolist())
    data["medication", "prescribed_in", "encounter"].edge_index = _to_edge_index(med_src, med_dst)

    # --- 5. ICD-9 ontology edges: code <-> category <-> chapter (vocabulary only) ---
    code_to_cat_src, code_to_cat_dst = [], []
    cat_to_code_src, cat_to_code_dst = [], []
    for c_str, c_id in code_idx.items():
        cat_str = category(c_str)
        if cat_str is not None and cat_str in cat_idx:
            t_id = cat_idx[cat_str]
            code_to_cat_src.append(c_id)
            code_to_cat_dst.append(t_id)
            cat_to_code_src.append(t_id)
            cat_to_code_dst.append(c_id)

    data["code", "in_category", "category"].edge_index = _to_edge_index(code_to_cat_src, code_to_cat_dst)
    data["category", "has_code", "code"].edge_index = _to_edge_index(cat_to_code_src, cat_to_code_dst)

    cat_to_chap_src, cat_to_chap_dst = [], []
    chap_to_cat_src, chap_to_cat_dst = [], []
    for cat_str, t_id in cat_idx.items():
        chap_str = chapter(cat_str)
        if chap_str in chap_idx:
            h_id = chap_idx[chap_str]
            cat_to_chap_src.append(t_id)
            cat_to_chap_dst.append(h_id)
            chap_to_cat_src.append(h_id)
            chap_to_cat_dst.append(t_id)

    data["category", "in_chapter", "chapter"].edge_index = _to_edge_index(cat_to_chap_src, cat_to_chap_dst)
    data["chapter", "has_category", "category"].edge_index = _to_edge_index(chap_to_cat_src, chap_to_cat_dst)

    # --- 6. Temporal visit edges: earlier encounter -> later encounter (same patient only) ---
    patient_arr = df["patient_nbr"].to_numpy()
    enc_id_arr = df["encounter_id"].to_numpy()
    same_patient = patient_arr[1:] == patient_arr[:-1]
    prev_pos = np.arange(len(df) - 1, dtype=np.int64)[same_patient]
    next_pos = prev_pos + 1

    # Guard against any non-monotonic encounter_id ordering within a patient
    assert np.all(enc_id_arr[prev_pos] <= enc_id_arr[next_pos]), (
        "Encounters for a patient are not sorted non-decreasingly by encounter_id!"
    )

    data["encounter", "next_visit", "encounter"].edge_index = _to_edge_index(prev_pos, next_pos)

    return data


def validate(data: HeteroData, df: pd.DataFrame) -> None:
    """Assert structural integrity, split disjointness, and absence of leakage edges."""
    n_enc = len(df)
    enc = data["encounter"]

    assert enc.x.shape[0] == n_enc
    assert enc.x.dtype == torch.float32
    assert not torch.isnan(enc.x).any(), "NaNs found in encounter features!"
    assert not torch.isinf(enc.x).any(), "Infs found in encounter features!"

    # Mutually exclusive and collectively exhaustive split masks
    split_sum = enc.train_mask.int() + enc.val_mask.int() + enc.test_mask.int()
    assert torch.all(split_sum == 1), "Split masks must partition every encounter exactly once!"

    # No patient spans multiple splits
    df_check = df.copy()
    df_check["split_code"] = (
        enc.train_mask.numpy().astype(int) * 1
        + enc.val_mask.numpy().astype(int) * 2
        + enc.test_mask.numpy().astype(int) * 3
    )
    assert (df_check.groupby("patient_nbr")["split_code"].nunique() == 1).all(), (
        "Patient spans multiple splits!"
    )

    # Forbidden reverse/leakage edge types must not exist
    forbidden_edge_types = {
        ("encounter", "primary_of", "code"),
        ("encounter", "secondary_of", "code"),
        ("encounter", "prescribed_in", "medication"),
        ("encounter", "prev_visit", "encounter"),
        ("encounter", "rev_next_visit", "encounter"),
    }
    assert set(data.edge_types).isdisjoint(forbidden_edge_types), (
        f"Forbidden edge type(s) found: {set(data.edge_types) & forbidden_edge_types}"
    )

    # Validate edge index bounds and same-patient constraint on next_visit
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        ei = data[edge_type].edge_index
        assert ei.ndim == 2 and ei.shape[0] == 2, f"Invalid edge_index shape for {edge_type}"
        assert ei.dtype == torch.long, f"edge_index must be int64 for {edge_type}"

        n_src = data[src_type].num_nodes if data[src_type].num_nodes is not None else data[src_type].x.shape[0]
        n_dst = data[dst_type].num_nodes if data[dst_type].num_nodes is not None else data[dst_type].x.shape[0]

        if ei.numel() > 0:
            assert int(ei[0].min()) >= 0 and int(ei[0].max()) < n_src, f"Source index out of bounds in {edge_type}"
            assert int(ei[1].min()) >= 0 and int(ei[1].max()) < n_dst, f"Target index out of bounds in {edge_type}"

    nv_ei = data["encounter", "next_visit", "encounter"].edge_index
    if nv_ei.numel() > 0:
        u = nv_ei[0].numpy()
        v = nv_ei[1].numpy()
        assert np.all(u < v), "next_visit edges must point strictly forward in row order!"
        assert np.all(enc.patient_nbr.numpy()[u] == enc.patient_nbr.numpy()[v]), (
            "next_visit edge connects two different patients!"
        )
        assert np.all(enc.encounter_id.numpy()[u] <= enc.encounter_id.numpy()[v]), (
            "next_visit edge violates encounter_id ordering!"
        )


def main() -> None:
    assert IN_PATH.exists(), f"Cleaned dataset not found: {IN_PATH}. Run `python -m ingest.diabetes130` first."
    df = pd.read_parquet(IN_PATH)
    data = build_graph(df)
    validate(data, df)

    print("=== Built HeteroData Summary ===")
    print(data)
    for et in data.edge_types:
        print(f"  {et}: {data[et].edge_index.shape[1]:,} edges")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUT_PATH)
    print(f"\nSaved heterogeneous graph to: {OUT_PATH}")


if __name__ == "__main__":
    main()