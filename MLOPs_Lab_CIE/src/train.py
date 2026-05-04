"""
train.py — MLOps Lab CIE
Train a classification model and save results to results/
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv')
MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
SCALER_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """Load training data from CSV."""
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess(df):
    """Split features and labels, scale features."""
    X = df.drop(columns=['label'])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Scaler saved to {SCALER_PATH}")

    return X_scaled, y, scaler


def train(X, y):
    """Train a Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    print(f"[INFO] Model metrics: {metrics}")

    # MLflow tracking
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    return model, metrics


def save_results(model, metrics):
    """Save model and step1 results JSON."""
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    result = {
        "step": "step1_training",
        "model": "RandomForestClassifier",
        "parameters": {"n_estimators": 100, "random_state": 42},
        "metrics": metrics,
    }

    result_path = os.path.join(RESULTS_DIR, 'step1_s1.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] Results saved to {result_path}")


if __name__ == '__main__':
    df = load_data()
    X, y, scaler = preprocess(df)
    model, metrics = train(X, y)
    save_results(model, metrics)
    print("[DONE] Training complete.")
