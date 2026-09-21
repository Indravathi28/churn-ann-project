# Customer Churn Prediction using an ANN

Predict whether a telecom customer is likely to churn, using an Artificial Neural
Network (Keras / TensorFlow) trained on the IBM / Kaggle **Telco Customer Churn**
dataset (7,043 customers, 20 features, 26.5% churn). Includes a Streamlit app where
you enter a customer's details and get a churn prediction.

## Project structure

```
churn-ann-project/
├── data/Telco-Customer-Churn.csv   raw dataset
├── src/
│   ├── config.py                   paths, column groups, form options
│   ├── preprocess.py               cleaning, train/val/test split, encoder + scaler
│   ├── model.py                    Keras ANN (dropout + L2 + batch-norm)
│   ├── evaluate.py                 metrics, threshold search, plots
│   └── predict.py                  inference helpers used by the app
├── eda.py                          exploratory analysis -> reports/eda/
├── train.py                        train + evaluate + save artifacts
├── app.py                          Streamlit form + prediction
├── tests/test_pipeline.py          tests that do not need TensorFlow
├── models/                         created by train.py
├── reports/                        plots + classification report
└── requirements.txt
```

## How to run

```bash
# 1. (recommended) create a virtual environment - Python 3.10-3.12 is a safe choice for TensorFlow
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) exploratory analysis -> reports/eda/*.png
python eda.py

# 4. train the ANN (small dataset: a few minutes at most on a laptop CPU)
python train.py

# 5. launch the web app
streamlit run app.py
```

`train.py` accepts options, e.g. `python train.py --epochs 200 --dropout 0.4 --l2 5e-4`.
Run the tests any time with `pytest -q`.

> You must run `python train.py` before `streamlit run app.py`. Training creates the
> files in `models/` that the app loads.

## Method, step by step

### 1. Data cleaning
* `TotalCharges` is stored as text and has **11 blank values**. All 11 customers have
  `tenure = 0` (new customers, not billed yet), so the value is set to **0** instead
  of a mean imputation.
* `customerID` is dropped (identifier, no predictive value).
* `Churn` is mapped `No -> 0`, `Yes -> 1`.

### 2. Encoding and scaling
| Group | Columns | Treatment |
|---|---|---|
| Numeric | `tenure`, `MonthlyCharges`, `TotalCharges` | `StandardScaler` |
| Categorical (15) | gender, Contract, PaymentMethod, InternetService, ... | `OneHotEncoder(handle_unknown="ignore")` |
| Already binary | `SeniorCitizen` | passed through |

This yields **45 input features**. Both steps live in one scikit-learn
`ColumnTransformer`, which is fitted on the **training split only** (no data leakage)
and saved as `models/preprocessor.joblib`, so the app applies exactly the same
transformation as training.

### 3. Splitting
Stratified **64% train / 16% validation / 20% test**. The test set is used once, for
the final evaluation.

### 4. ANN architecture
```
Input (45) -> Dense 64, ReLU, L2 -> BatchNorm -> Dropout 0.3
           -> Dense 32, ReLU, L2 -> BatchNorm -> Dropout 0.3
           -> Dense 1, sigmoid
```
Loss: binary cross-entropy. Optimiser: Adam (lr 1e-3).

**Regularisation techniques used**
* **Dropout (0.3)** after each hidden layer
* **L2 weight penalty (1e-4)** on hidden layers
* **Batch normalisation**
* **Early stopping** on validation AUC (`restore_best_weights=True`)
* **ReduceLROnPlateau** to lower the learning rate when validation loss stalls

### 5. Class imbalance
About 27% of customers churn, so accuracy alone is misleading (predicting "stays" for
everyone gives about 73% accuracy). Training uses **balanced class weights**, and the
decision threshold is tuned (next step).

### 6. Threshold selection
The default cut-off of 0.5 is not optimal for imbalanced data. `train.py` picks the
threshold that maximises **F1 on the validation set**, saves it in
`models/metadata.json`, and the app uses it.

### 7. Evaluation (on the test set)
Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report,
training curves and a ROC curve. A **logistic-regression baseline** is trained on the
same features so you can see what the ANN adds. Outputs are written to `reports/`.

For a telecom company, **recall on churners** matters: a missed churner is a lost
customer, while a false alarm only costs a retention offer.

### 8. Streamlit app
The form collects all 19 customer fields. After you press **Predict churn** it shows
the churn probability, a Low/Medium/High risk level, a likely / unlikely verdict, and
rule-based retention suggestions. If phone or internet service is "No", the dependent
fields are set automatically to "No phone service" / "No internet service", as they
are in the dataset.

## What to expect

On this dataset, well-tuned models usually reach roughly **ROC-AUC 0.83-0.85** and
**accuracy 0.74-0.80** (depending on the threshold: a lower threshold catches more
churners but lowers accuracy). A simple logistic regression scores about the same, so
don't be surprised if the ANN is not dramatically better: the data is small and
tabular, where simple models are strong. Report your own numbers from `train.py`;
they will vary slightly between runs and machines.

## Findings from the exploratory analysis (`python eda.py`)

| Factor | Churn rate |
|---|---|
| Month-to-month contract / One year / Two year | 42.7% / 11.3% / 2.8% |
| Fiber optic / DSL / no internet | 41.9% / 19.0% / 7.4% |
| Electronic check / other payment methods | 45.3% / 15-19% |
| No tech support / with tech support | 41.6% / 15.2% |
| Tenure 0-12 months / 49+ months | 47.4% / 9.5% |

The retention tips in the app are based on these patterns. They are general
suggestions, not an explanation of an individual prediction.

## Troubleshooting
* **`pip install tensorflow` fails**: check `python --version`. TensorFlow only
  supports certain Python versions (see its install page); a virtual environment with
  Python 3.10-3.12 is a safe choice.
* **App says "Trained artifacts not found"**: run `python train.py` first.
* **`ModuleNotFoundError: No module named 'src'`**: run all commands from the project
  root folder (the one containing `app.py`).
* **Model file won't load after upgrading TensorFlow/Keras**: re-run `python train.py`.

## Limitations and ideas for extension
* The dataset is a single snapshot; real churn models use time-based validation.
* Add SHAP or permutation importance to explain individual predictions.
* Compare with gradient boosting (XGBoost / LightGBM), often the strongest choice on
  tabular data like this.
* Tune hyper-parameters with Keras Tuner or Optuna.
* Use cost-based threshold selection (e.g. retention offer cost vs customer lifetime
  value) instead of maximising F1.

## Data source
IBM Sample Data Sets - Telco Customer Churn (also on Kaggle).
