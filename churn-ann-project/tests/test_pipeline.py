"""Tests that do not need TensorFlow. Run with:  pytest -q"""
import numpy as np
import pytest

from src.config import CATEGORY_OPTIONS, FEATURE_ORDER, INTERNET_ADDONS
from src.evaluate import best_threshold, compute_metrics
from src.predict import normalize_inputs, predict_churn, risk_band, to_dataframe
from src.preprocess import build_preprocessor, load_data, make_splits


@pytest.fixture(scope="module")
def df():
    return load_data()


def sample_customer(**overrides):
    base = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0,
        "TotalCharges": 475.0,
    }
    base.update(overrides)
    return base


def test_cleaning(df):
    assert len(df) == 7043
    assert "customerID" not in df.columns
    assert df.isna().sum().sum() == 0
    assert set(df["Churn"].unique()) == {0, 1}
    assert df["TotalCharges"].dtype.kind == "f"


def test_splits_are_stratified_and_disjoint(df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(df)
    assert len(X_tr) + len(X_va) + len(X_te) == len(df)
    assert not (set(X_tr.index) & set(X_te.index))
    assert not (set(X_tr.index) & set(X_va.index))
    rates = [y.mean() for y in (y_tr, y_va, y_te)]
    assert max(rates) - min(rates) < 0.01


def test_preprocessor_output(df):
    X_tr, X_va, *_ = make_splits(df)
    pre = build_preprocessor().fit(X_tr)
    Xt = pre.transform(X_tr)
    assert Xt.shape[0] == len(X_tr)
    assert not np.isnan(Xt).any()
    # numeric columns are scaled to ~zero mean on the training data
    assert abs(Xt[:, 0].mean()) < 1e-6
    # validation data has the same width
    assert pre.transform(X_va).shape[1] == Xt.shape[1]


def test_form_options_match_training_data(df):
    for col, options in CATEGORY_OPTIONS.items():
        assert set(options) == set(df[col].unique()), col


def test_normalize_inputs_enforces_dependencies():
    out = normalize_inputs(sample_customer(InternetService="No", PhoneService="No"))
    assert out["MultipleLines"] == "No phone service"
    assert all(out[c] == "No internet service" for c in INTERNET_ADDONS)


def test_to_dataframe_column_order():
    assert list(to_dataframe(sample_customer()).columns) == FEATURE_ORDER


class DummyModel:
    """Stands in for a Keras model: exposes predict(X, verbose=0)."""
    def __init__(self, p):
        self.p = p

    def predict(self, X, verbose=0):
        return np.full((len(X), 1), self.p, dtype="float32")


def test_predict_churn_end_to_end(df):
    X_tr, *_ = make_splits(df)
    pre = build_preprocessor().fit(X_tr)
    hi = predict_churn(DummyModel(0.8), pre, sample_customer(), threshold=0.5)
    lo = predict_churn(DummyModel(0.1), pre, sample_customer(), threshold=0.5)
    assert hi["will_churn"] and hi["risk"] == "High"
    assert not lo["will_churn"] and lo["risk"] == "Low"
    assert 0 <= hi["probability"] <= 1


def test_risk_band():
    assert risk_band(0.55, 0.5) == "High"
    assert risk_band(0.35, 0.5) == "Medium"
    assert risk_band(0.10, 0.5) == "Low"


def test_threshold_and_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 500), 0, 1)
    t = best_threshold(y, p)
    m = compute_metrics(y, p, t)
    assert 0.1 <= t <= 0.9
    assert m["roc_auc"] > 0.8
    assert sum(map(sum, m["confusion_matrix"])) == 500
