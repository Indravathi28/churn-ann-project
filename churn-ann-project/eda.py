"""Exploratory data analysis. Saves figures to reports/eda/ and prints key stats.

Usage:  python eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import REPORT_DIR
from src.preprocess import load_data

OUT = REPORT_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def churn_rate_by(df, col):
    return (df.groupby(col)["Churn"].agg(["mean", "count"])
              .rename(columns={"mean": "churn_rate"})
              .sort_values("churn_rate", ascending=False))


def main():
    df = load_data()
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Missing values after cleaning: {int(df.isna().sum().sum())}")
    print(f"Overall churn rate: {df['Churn'].mean():.1%}\n")

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
    ax.bar(counts.index, counts.values, color=["#4C78A8", "#E45756"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 60, f"{v:,}\n({v / len(df):.1%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(OUT / "01_class_balance.png", dpi=130)
    plt.close(fig)

    # 2. Churn rate for the most informative categorical features
    cols = ["Contract", "InternetService", "PaymentMethod", "TechSupport",
            "OnlineSecurity", "PaperlessBilling"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.ravel(), cols):
        t = churn_rate_by(df, col)
        ax.barh(t.index, t["churn_rate"] * 100, color="#E45756")
        ax.set_title(f"Churn rate by {col}")
        ax.set_xlabel("Churn rate (%)")
        for i, v in enumerate(t["churn_rate"] * 100):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
        ax.set_xlim(0, max(60, (t["churn_rate"] * 100).max() + 10))
        print(f"--- {col} ---\n{t.round(3)}\n")
    fig.tight_layout()
    fig.savefig(OUT / "02_churn_by_category.png", dpi=130)
    plt.close(fig)

    # 3. Numeric feature distributions by churn
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.histplot(data=df, x=col, hue=df["Churn"].map({0: "Stayed", 1: "Churned"}),
                     bins=30, stat="density", common_norm=False, element="step", ax=ax)
        ax.set_title(f"{col} by churn")
    fig.tight_layout()
    fig.savefig(OUT / "03_numeric_distributions.png", dpi=130)
    plt.close(fig)

    # 4. Tenure buckets
    df["tenure_bucket"] = np.select(
        [df["tenure"] <= 12, df["tenure"] <= 24, df["tenure"] <= 48],
        ["0-12 mo", "13-24 mo", "25-48 mo"], default="49+ mo")
    print(f"--- tenure bucket ---\n{churn_rate_by(df, 'tenure_bucket').round(3)}\n")

    # 5. Correlation of numeric features with churn
    corr = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation (numeric features)")
    fig.tight_layout()
    fig.savefig(OUT / "04_correlation.png", dpi=130)
    plt.close(fig)

    print(f"Figures saved to {OUT}")


if __name__ == "__main__":
    main()
