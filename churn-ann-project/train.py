"""Train the churn ANN, evaluate it, and save everything the Streamlit app needs.

Usage:
    python train.py
    python train.py --epochs 150 --dropout 0.4 --l2 5e-4

Outputs:
    models/churn_ann.keras      trained network
    models/preprocessor.joblib  fitted scaler + one-hot encoder
    models/metadata.json        threshold, metrics, hyper-parameters
    reports/*.png, reports/classification_report.txt
"""
import argparse
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.config import (
    METADATA_PATH, MODEL_DIR, MODEL_PATH, PREPROCESSOR_PATH, RANDOM_STATE, REPORT_DIR,
)
from src.evaluate import (
    best_threshold, compute_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_training_history, text_report,
)
from src.model import build_ann
from src.preprocess import build_preprocessor, load_data, make_splits


def parse_args():
    p = argparse.ArgumentParser(description="Train the Telco churn ANN")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    keras.utils.set_random_seed(RANDOM_STATE)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    # ---- 1. Load, clean, split -------------------------------------------------
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)} "
          f"| churn rate {df['Churn'].mean():.1%}")

    # ---- 2. Encode + scale (fit on TRAIN only to avoid data leakage) -----------
    preprocessor = build_preprocessor()
    Xtr = preprocessor.fit_transform(X_train).astype("float32")
    Xva = preprocessor.transform(X_val).astype("float32")
    Xte = preprocessor.transform(X_test).astype("float32")
    feature_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]
    print(f"Input features after encoding: {Xtr.shape[1]}")

    # ---- 3. Handle class imbalance (about 27% churn) ---------------------------
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print(f"Class weights: {class_weight}")

    # ---- 4. Build + train the ANN ----------------------------------------------
    model = build_ann(Xtr.shape[1], dropout=args.dropout, l2=args.l2,
                      learning_rate=args.lr)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=5, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(
        Xtr, y_train.values,
        validation_data=(Xva, y_val.values),
        epochs=args.epochs, batch_size=args.batch_size,
        class_weight=class_weight, callbacks=callbacks, verbose=2,
    )

    # ---- 5. Pick the decision threshold on VALIDATION data ---------------------
    val_prob = model.predict(Xva, verbose=0).ravel()
    threshold = best_threshold(y_val.values, val_prob)
    print(f"\nBest validation threshold (max F1): {threshold}")

    # ---- 6. Final evaluation on the untouched TEST set -------------------------
    test_prob = model.predict(Xte, verbose=0).ravel()
    ann_default = compute_metrics(y_test.values, test_prob, 0.5)
    ann_tuned = compute_metrics(y_test.values, test_prob, threshold)

    # Simple baseline so the ANN result has something to be compared against
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(Xtr, y_train)
    base_prob = baseline.predict_proba(Xte)[:, 1]
    base_metrics = compute_metrics(y_test.values, base_prob, 0.5)

    print("\n=== TEST SET RESULTS ===")
    print(f"{'Model':<32}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}")
    for name, m in [("ANN @ 0.50", ann_default),
                    (f"ANN @ {threshold:.2f} (tuned)", ann_tuned),
                    ("Logistic regression @ 0.50", base_metrics)]:
        print(f"{name:<32}{m['accuracy']:>7.3f}{m['precision']:>7.3f}"
              f"{m['recall']:>7.3f}{m['f1']:>7.3f}{m['roc_auc']:>7.3f}")

    report = text_report(y_test.values, test_prob, threshold)
    print(f"\nClassification report (ANN, threshold {threshold}):\n{report}")
    (REPORT_DIR / "classification_report.txt").write_text(report)

    # ---- 7. Plots --------------------------------------------------------------
    plot_training_history(history, REPORT_DIR / "training_curves.png")
    plot_confusion_matrix(ann_tuned["confusion_matrix"], REPORT_DIR / "confusion_matrix.png")
    plot_roc_curves({"ANN": (y_test.values, test_prob),
                     "Logistic regression": (y_test.values, base_prob)},
                    REPORT_DIR / "roc_curve.png")

    # ---- 8. Save artifacts for the app ----------------------------------------
    model.save(MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    METADATA_PATH.write_text(json.dumps({
        "threshold": threshold,
        "input_dim": int(Xtr.shape[1]),
        "feature_names": feature_names,
        "hyperparameters": vars(args),
        "epochs_run": len(history.history["loss"]),
        "test_metrics_ann_tuned": ann_tuned,
        "test_metrics_ann_default": ann_default,
        "test_metrics_baseline": base_metrics,
    }, indent=2))
    print(f"\nSaved model, preprocessor and metadata to {MODEL_DIR}")
    print(f"Saved plots and report to {REPORT_DIR}")
    print("Next step:  streamlit run app.py")


if __name__ == "__main__":
    main()
