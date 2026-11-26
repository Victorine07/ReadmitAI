"""Tabular feature pipeline for Diabetes 130.
All transformers fit strictly on train rows and transform train, val, and test.
"""

import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.icd9 import category, chapter

AGE_MAP = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

ID_COLS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

MISSING_CAT_COLS = [
    "race",
    "payer_code",
    "medical_specialty",
    "A1Cresult",
    "max_glu_serum",
]

MED_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

BASE_NUMERIC_COLS = [
    "age_num",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

HISTORY_NUMERIC_COLS = [
    "prior_encounters_in_data",
]


def _parse_age(val: object) -> float:
    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in AGE_MAP:
        return float(AGE_MAP[s])
    m = re.match(r"\[(\d+)-(\d+)\)", s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2.0
    return np.nan


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering without fitting (safe across splits)."""
    out = df.copy()

    # Numeric midpoint for age bracket
    out["age_num"] = out["age"].apply(_parse_age)

    # Convert coded IDs to categorical strings
    for col in ID_COLS:
        out[col] = out[col].fillna(-1).astype(int).astype(str)

    # Explicit 'missing' category for informative missingness
    for col in MISSING_CAT_COLS:
        out[col] = out[col].fillna("missing").astype(str)

    # ICD-9 chapters and multi-hot document of 3-character categories
    cat_series = []
    for col in ("diag_1", "diag_2", "diag_3"):
        out[f"{col}_chapter"] = out[col].apply(chapter)
        cat_series.append(out[col].apply(lambda x: category(x) or ""))

    out["diag_doc"] = (
        cat_series[0] + " " + cat_series[1] + " " + cat_series[2]
    ).str.strip()

    # Past-only encounter counter (relies on Phase 2 sort by patient_nbr, encounter_id)
    out["prior_encounters_in_data"] = out.groupby("patient_nbr").cumcount()

    return out


def build_features(
    df: pd.DataFrame, include_history: bool = True
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, List[str]]:
    """Fit encoders on train split only and transform train/val/test."""
    prepared = prepare(df)

    numeric_cols = list(BASE_NUMERIC_COLS)
    if include_history:
        numeric_cols.extend(HISTORY_NUMERIC_COLS)

    categorical_cols = (
        ["gender", "change", "diabetesMed"]
        + ID_COLS
        + MISSING_CAT_COLS
        + ["diag_1_chapter", "diag_2_chapter", "diag_3_chapter"]
        + [c for c in MED_COLS if c in prepared.columns]
    )

    for col in categorical_cols:
        prepared[col] = prepared[col].fillna("missing").astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=True,
                ),
                categorical_cols,
            ),
            (
                "diag",
                CountVectorizer(
                    binary=True,
                    min_df=20,
                    token_pattern=r"\S+",
                ),
                "diag_doc",
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    train_mask = prepared["split"] == "train"
    val_mask = prepared["split"] == "val"
    test_mask = prepared["split"] == "test"

    X_train = preprocessor.fit_transform(prepared.loc[train_mask]).tocsr()
    X_val = preprocessor.transform(prepared.loc[val_mask]).tocsr()
    X_test = preprocessor.transform(prepared.loc[test_mask]).tocsr()

    feature_names = list(preprocessor.get_feature_names_out())

    # Sanity checks: column alignment and zero NaNs
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature dimension mismatch across splits!"
    for name, mat in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
        assert not np.isnan(mat.data).any(), f"NaNs found in {name}!"

    return X_train, X_val, X_test, feature_names