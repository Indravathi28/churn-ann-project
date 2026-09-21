"""Data loading, cleaning, splitting and the scikit-learn preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_PASSTHROUGH, CATEGORICAL_FEATURES, DATA_PATH, ID_COL,
    NUMERIC_FEATURES, RANDOM_STATE, TARGET,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data-quality issues in the raw Telco dataframe.

    * ``TotalCharges`` is stored as text and contains 11 blank strings. Those
      customers all have ``tenure == 0`` (brand new, not yet billed), so 0 is
      the correct value rather than a guess such as the column mean.
    * ``customerID`` is an identifier with no predictive meaning -> dropped.
    * ``Churn`` is mapped from "No"/"Yes" to 0/1.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0.0)
    df = df.drop(columns=[ID_COL], errors="ignore")
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype("int64")
    return df


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return clean_data(pd.read_csv(path))


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categoricals, pass binary through."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_PASSTHROUGH),
        ],
        remainder="drop",
    )


def make_splits(df: pd.DataFrame, test_size=0.20, val_size=0.20):
    """Stratified train / validation / test split (about 64 / 16 / 20).

    The test set is never used for training or for choosing the threshold.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=val_size,
        stratify=y_tr_full, random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
