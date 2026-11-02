import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import ChurnFeatureEngineer

TARGET = "Churn"
DROP_COLS = ["Customer_ID", TARGET]
NUMERIC = [
    "Tenure_Months", "Monthly_Charges_INR", "Total_Charges_INR",
    "Support_Tickets_6M", "Avg_Data_Usage_GB",
    "Average_Monthly_Spending", "Monthly_Total_Charge_Ratio", "Ticket_Intensity"
]
CATEGORICAL = [
    "Gender", "Senior_Citizen", "Contract_Type", "Internet_Service",
    "Payment_Method", "Tech_Support", "Online_Security",
    "Tenure_Category", "Service_Usage_Indicator"
]
FEATURES = NUMERIC + CATEGORICAL

def load_data(path):
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name="Customer Churn Data")
    else:
        df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET}'")
    df = df.drop_duplicates().copy()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()
    df = df[df[TARGET].isin(["Churn", "Not Churn"])].copy()
    return df

def build_pipeline(model):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])
    return Pipeline([
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocessing", preprocess),
        ("classifier", model)
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/customer_churn.xlsx")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()

    df = load_data(args.data)
    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced",
            min_samples_leaf=2
        )
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_churn": round(float(precision_score(y_test, predictions, pos_label="Churn", zero_division=0)), 4),
            "recall_churn": round(float(recall_score(y_test, predictions, pos_label="Churn", zero_division=0)), 4),
            "f1_churn": round(float(f1_score(y_test, predictions, pos_label="Churn", zero_division=0)), 4)
        }
        results[name] = metrics
        print(f"\n{name}: {metrics}")
        print(classification_report(y_test, predictions, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, predictions, labels=["Not Churn", "Churn"]))
        if metrics["f1_churn"] > best_f1:
            best_name, best_model, best_f1 = name, pipeline, metrics["f1_churn"]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output / "churn_model.joblib")
    metadata = {
        "best_model": best_name,
        "selection_metric": "f1_churn",
        "test_metrics": results,
        "classes": ["Not Churn", "Churn"]
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {output / 'churn_model.joblib'}")

if __name__ == "__main__":
    main()
