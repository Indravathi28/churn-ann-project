"""Inference helpers shared by the Streamlit app and the tests."""
import json

import joblib
import pandas as pd

from src.config import (
    FEATURE_ORDER, INTERNET_ADDONS, METADATA_PATH, MODEL_PATH, PREPROCESSOR_PATH,
)


def load_artifacts():
    """Load the trained ANN, the fitted preprocessor and the metadata JSON."""
    if not (MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and METADATA_PATH.exists()):
        raise FileNotFoundError(
            "Trained artifacts not found in the 'models/' folder. "
            "Run `python train.py` first."
        )
    from tensorflow import keras  # imported here so tests can run without TF
    model = keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, preprocessor, metadata


def normalize_inputs(raw: dict) -> dict:
    """Make form values consistent with how the dataset encodes dependencies.

    In the Telco data, customers without phone service always have
    MultipleLines = "No phone service", and customers without internet always
    have "No internet service" for every add-on. Enforcing this keeps the input
    inside the distribution the model was trained on.
    """
    data = dict(raw)
    if data.get("PhoneService") == "No":
        data["MultipleLines"] = "No phone service"
    if data.get("InternetService") == "No":
        for col in INTERNET_ADDONS:
            data[col] = "No internet service"
    return data


def to_dataframe(raw: dict) -> pd.DataFrame:
    """One-row DataFrame with columns in the training order."""
    data = normalize_inputs(raw)
    return pd.DataFrame([data])[FEATURE_ORDER]


def risk_band(probability: float, threshold: float) -> str:
    """Human-friendly band that is consistent with the yes/no decision.

    High   : probability >= threshold  (model says "likely to churn")
    Medium : within 60-100% of the threshold
    Low    : below 60% of the threshold
    """
    if probability >= threshold:
        return "High"
    if probability >= 0.6 * threshold:
        return "Medium"
    return "Low"


def predict_churn(model, preprocessor, raw: dict, threshold: float = 0.5) -> dict:
    X = preprocessor.transform(to_dataframe(raw)).astype("float32")
    probability = float(model.predict(X, verbose=0).ravel()[0])
    return {
        "probability": probability,
        "will_churn": probability >= threshold,
        "risk": risk_band(probability, threshold),
    }
