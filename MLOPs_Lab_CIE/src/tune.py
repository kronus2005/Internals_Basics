"""
tune.py — MLOps Lab CIE
Hyperparameter tuning for the model and save results to results/step2_s2.json
"""

import pandas as pd
import json
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'training_data.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'best_tuned_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    if not os.path.exists(SCALER_PATH):
        print(f"[ERROR] Scaler not found at {SCALER_PATH}. Run train.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    X = df.drop(columns=['label'])
    y = df['label']
    X_scaled = scaler.transform(X)

    print("[INFO] Starting hyperparameter tuning...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid.fit(X_scaled, y)
    
    print(f"[INFO] Best parameters: {grid.best_params_}")
    print(f"[INFO] Best cross-validation score: {grid.best_score_:.4f}")
    
    # Save best model
    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print(f"[INFO] Best tuned model saved to {MODEL_PATH}")
    
    # Save step2_s2.json
    result = {
        "step": "step2_tuning",
        "best_params": grid.best_params_,
        "best_score": round(grid.best_score_, 4),
        "model": "RandomForestClassifier"
    }
    
    result_path = os.path.join(RESULTS_DIR, 'step2_s2.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"[INFO] Tuning results saved to {result_path}")

if __name__ == '__main__':
    main()
