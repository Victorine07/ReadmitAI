"""Phase 2: clean, label and split Diabetes 130 by patient.
Run from project root:  python -m ingest.diabetes130
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "diabetes130" / "diabetic_data.csv"
OUT = ROOT / "data" / "processed" / "d130_clean.parquet"

EXPIRED_OR_HOSPICE = {11, 13, 14, 19, 20, 21}
DROP_COLS = ["weight", "examide", "citoglipton"]
SEED = 42


def load_raw() -> pd.DataFrame:
    """Step 1: Load raw Diabetes 130 CSV with '?' parsed as NaN."""
    df = pd.read_csv(RAW, na_values="?", low_memory=False)
    print(f"[Step 1] Loaded raw data: {len(df):,} rows x {df.shape[1]} cols")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Steps 2-6: Filter rows, drop uninformative columns, create binary label, and sort chronologically."""
    df = df.copy()

    # Step 2: Drop encounters ending in death or hospice
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)]
    print(f"[Step 2] Dropped expired/hospice dispositions: {before:,} -> {len(df):,} (removed {before - len(df):,})")

    # Step 3: Drop Unknown/Invalid gender rows
    before = len(df)
    df = df[df["gender"] != "Unknown/Invalid"]
    print(f"[Step 3] Dropped Unknown/Invalid gender: {before:,} -> {len(df):,} (removed {before - len(df):,})")

    # Step 4: Drop near-empty and constant medication columns
    existing_drop_cols = [col for col in DROP_COLS if col in df.columns]
    df = df.drop(columns=existing_drop_cols)
    print(f"[Step 4] Dropped columns {existing_drop_cols}: {df.shape[1]} cols remaining")

    # Step 5: Binary 30-day readmission label
    df["label"] = (df["readmitted"] == "<30").astype(int)
    print(f"[Step 5] Created binary label '<30': positive rate = {df['label'].mean():.4f} ({df['label'].sum():,} positives)")

    # Step 6: Sort by patient_nbr then encounter_id and reset index for positional alignment
    df = df.sort_values(["patient_nbr", "encounter_id"], kind="mergesort").reset_index(drop=True)
    print("[Step 6] Sorted by ['patient_nbr', 'encounter_id'] and reset index")

    return df


def split_by_patient(df: pd.DataFrame) -> pd.Series:
    """Step 7: Assign 'train'/'val'/'test' using 20-fold StratifiedGroupKFold on patient_nbr."""
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    splits = pd.Series(index=df.index, dtype="object")

    for fold_idx, (_, test_idx) in enumerate(sgkf.split(df, df["label"], groups=df["patient_nbr"])):
        if fold_idx in {0, 1, 2}:
            assigned_split = "test"
        elif fold_idx in {3, 4, 5}:
            assigned_split = "val"
        else:
            assigned_split = "train"
        splits.iloc[test_idx] = assigned_split

    # every row must be assigned a split
    assert splits.notna().all(), f"{splits.isna().sum()} rows were not assigned a split"
    return splits


def verify_and_summarize(df: pd.DataFrame) -> None:
    """Step 8: Run integrity assertions and print split summary."""
    # Check 1: No patient appears in more than one split
    distinct_splits_per_patient = df.groupby("patient_nbr")["split"].nunique()
    assert (distinct_splits_per_patient == 1).all(), "Patient leakage detected across splits!"

    # Check 2: Label is strictly binary {0, 1} and no excluded disposition codes remain
    assert set(df["label"].unique()).issubset({0, 1}), f"Unexpected label values: {df['label'].unique()}"
    assert not df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE).any(), "Excluded disposition codes remain!"

    # Check 3: Split proportions (~70/15/15) and positive rates stay consistent across splits
    total = len(df)
    overall_pos_rate = df["label"].mean()
    expected_shares = {"train": 0.70, "val": 0.15, "test": 0.15}

    print("\n=== Split Summary ===")
    for split_name, expected_share in expected_shares.items():
        subset = df[df["split"] == split_name]
        actual_share = len(subset) / total
        pos_rate = subset["label"].mean()
        n_patients = subset["patient_nbr"].nunique()

        assert abs(actual_share - expected_share) < 0.03, (
            f"Split '{split_name}' share {actual_share:.3f} deviates too far from {expected_share:.3f}"
        )
        # positive rate must stay within ±0.5 percentage points of the overall rate
        assert abs(pos_rate - overall_pos_rate) < 0.005, (
            f"Split '{split_name}' positive rate {pos_rate:.4f} deviates too far from overall {overall_pos_rate:.4f}"
        )

        print(
            f"  {split_name:<5} | rows: {len(subset):>6,} ({actual_share * 100:5.2f}%) "
            f"| patients: {n_patients:>6,} | pos_rate: {pos_rate * 100:5.2f}%"
        )

    print(f"  TOTAL | rows: {total:>6,} (100.00%) | patients: {df['patient_nbr'].nunique():>6,} | pos_rate: {overall_pos_rate * 100:5.2f}%")


def main() -> None:
    raw_df = load_raw()
    df = clean(raw_df)
    df["split"] = split_by_patient(df)
    verify_and_summarize(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nSaved clean split dataset to: {OUT}")


if __name__ == "__main__":
    main()