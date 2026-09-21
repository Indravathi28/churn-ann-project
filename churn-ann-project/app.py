"""Streamlit app: enter a customer's details and get a churn prediction.

Run:  streamlit run app.py     (after `python train.py`)
"""
import streamlit as st

from src.config import CATEGORY_OPTIONS
from src.predict import load_artifacts, normalize_inputs, predict_churn

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


try:
    model, preprocessor, metadata = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

threshold = metadata["threshold"]

# ------------------------------------------------------------------ Sidebar --
with st.sidebar:
    st.header("About the model")
    st.write("Artificial neural network (Keras) trained on the IBM/Kaggle "
             "Telco Customer Churn dataset.")
    m = metadata["test_metrics_ann_tuned"]
    st.caption("Held-out test-set performance")
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("Accuracy", f"{m['accuracy']:.3f}")
    c1.metric("Recall", f"{m['recall']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    st.caption(f"Decision threshold: {threshold:.2f} (chosen on validation data "
               "to maximise F1).")

# --------------------------------------------------------------------- Form --
st.title("📉 Customer Churn Prediction")
st.write("Fill in the customer's details and press **Predict churn**.")

with st.form("customer_form"):
    st.subheader("Demographics")
    c1, c2, c3, c4 = st.columns(4)
    gender = c1.selectbox("Gender", CATEGORY_OPTIONS["gender"])
    senior = c2.selectbox("Senior citizen", ["No", "Yes"])
    partner = c3.selectbox("Has partner", CATEGORY_OPTIONS["Partner"])
    dependents = c4.selectbox("Has dependents", CATEGORY_OPTIONS["Dependents"])

    st.subheader("Services")
    c1, c2, c3 = st.columns(3)
    phone = c1.selectbox("Phone service", CATEGORY_OPTIONS["PhoneService"])
    multiple = c2.selectbox("Multiple lines", CATEGORY_OPTIONS["MultipleLines"])
    internet = c3.selectbox("Internet service", CATEGORY_OPTIONS["InternetService"])

    c1, c2, c3 = st.columns(3)
    security = c1.selectbox("Online security", CATEGORY_OPTIONS["OnlineSecurity"])
    backup = c2.selectbox("Online backup", CATEGORY_OPTIONS["OnlineBackup"])
    protection = c3.selectbox("Device protection", CATEGORY_OPTIONS["DeviceProtection"])
    c1, c2, c3 = st.columns(3)
    tech = c1.selectbox("Tech support", CATEGORY_OPTIONS["TechSupport"])
    tv = c2.selectbox("Streaming TV", CATEGORY_OPTIONS["StreamingTV"])
    movies = c3.selectbox("Streaming movies", CATEGORY_OPTIONS["StreamingMovies"])
    st.caption("If phone service is 'No' or internet service is 'No', the related "
               "options are set automatically to 'No phone service' / "
               "'No internet service', as in the training data.")

    st.subheader("Account & billing")
    c1, c2, c3 = st.columns(3)
    contract = c1.selectbox("Contract", CATEGORY_OPTIONS["Contract"])
    paperless = c2.selectbox("Paperless billing", CATEGORY_OPTIONS["PaperlessBilling"])
    payment = c3.selectbox("Payment method", CATEGORY_OPTIONS["PaymentMethod"])

    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure (months)", 0, 72, 12)
    monthly = c2.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
    with c3:
        auto_total = st.checkbox("Estimate total charges (tenure × monthly)", value=True)
        total_input = st.number_input("Total charges ($) — used if the box above is unticked",
                                      0.0, 9000.0, 0.0, step=10.0)

    submitted = st.form_submit_button("Predict churn", type="primary")

# --------------------------------------------------------------- Prediction --
if submitted:
    total = tenure * monthly if auto_total else total_input
    customer = normalize_inputs({
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    })
    result = predict_churn(model, preprocessor, customer, threshold)
    prob = result["probability"]

    st.divider()
    st.subheader("Result")
    left, right = st.columns([1, 2])
    left.metric("Churn probability", f"{prob:.1%}")
    left.metric("Risk level", result["risk"])
    with right:
        if result["will_churn"]:
            st.error("⚠️ This customer is **likely to churn**.")
        elif result["risk"] == "Medium":
            st.warning("This customer is **not predicted to churn**, but is close to the "
                       "decision boundary.")
        else:
            st.success("✅ This customer is **unlikely to churn**.")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"Predicted churn if probability ≥ {threshold:.2f}.")

    # Simple rule-based suggestions. These reflect patterns seen in the training
    # data (see reports/eda) - they are NOT an explanation of this specific
    # prediction.b
    tips = []
    if customer["Contract"] == "Month-to-month":
        tips.append("Month-to-month customers churn far more often than annual/2-year "
                    "customers. Offer a discount for a longer contract.")
    if customer["InternetService"] == "Fiber optic" and customer["TechSupport"] == "No":
        tips.append("Fiber customers without tech support churn often. Consider "
                    "bundling tech support or online security.")
    if customer["PaymentMethod"] == "Electronic check":
        tips.append("Electronic-check payers churn most. Encourage automatic payment.")
    if customer["tenure"] <= 12:
        tips.append("The first year is the riskiest period. An onboarding or "
                    "check-in call can help.")
    if tips and (result["will_churn"] or result["risk"] == "Medium"):
        st.subheader("Possible retention actions")
        for t in tips:
            st.markdown(f"- {t}")

    with st.expander("Show the data sent to the model"):
        st.json(customer)
