"""Keras ANN definition."""
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_ann(input_dim: int, hidden_units=(64, 32), dropout=0.3,
              l2=1e-4, learning_rate=1e-3) -> keras.Model:
    """Feed-forward ANN for binary classification.

    Regularisation used:
      * L2 weight penalty on every hidden Dense layer
      * Dropout after every hidden layer
      * Batch normalisation (stabilises training, mild regularising effect)
    Early stopping and learning-rate reduction are added as callbacks in
    ``train.py``.
    """
    model = keras.Sequential(name="churn_ann")
    model.add(keras.Input(shape=(input_dim,)))
    for i, units in enumerate(hidden_units, start=1):
        model.add(layers.Dense(units, activation="relu",
                               kernel_regularizer=regularizers.l2(l2),
                               name=f"dense_{i}"))
        model.add(layers.BatchNormalization(name=f"bn_{i}"))
        model.add(layers.Dropout(dropout, name=f"dropout_{i}"))
    model.add(layers.Dense(1, activation="sigmoid", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
