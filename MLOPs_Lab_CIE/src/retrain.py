"""
retrain.py — MLOps Lab CIE
Retraining pipeline for the model and save step4 results.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'training_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    if not os.path.exists(SCALER_PATH):
        print(f"[ERROR] Scaler not found at {SCALER_PATH}. Run train.py first.")
        return

    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=['label']).values
    y  = df['label'].values
    X_scaled = scaler.transform(X)

    print("[INFO] Retraining model on full dataset...")
    # Retrain
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save the retrained model
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Retrained model saved to {MODEL_PATH}")

    y_pred   = model.predict(X_scaled)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    result = {
        "step": "step4_retraining",
        "accuracy":   round(accuracy_score(y, y_pred), 4),
        "precision":  round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y, y_pred, zero_division=0), 4),
        "f1_score":   round(f1_score(y, y_pred, zero_division=0), 4),
        "cross_val_accuracy_mean": round(cv_scores.mean(), 4),
        "cross_val_accuracy_std":  round(cv_scores.std(), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    out_path = os.path.join(RESULTS_DIR, 'step4_s8.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"[INFO] Retraining results saved to {out_path}")


if __name__ == '__main__':
    main()
