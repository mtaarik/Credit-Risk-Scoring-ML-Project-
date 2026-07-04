import os
import json
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
import xgboost as xgb
from src.app.schemas import CreditRequest, PredictionResponse
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.xgboost

app = FastAPI(title="Credit Risk Scoring API", description="API for predicting credit card default")

def get_latest_model_uri():
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = MlflowClient()
        exp = client.get_experiment_by_name("Credit Risk Scoring")
        if exp is None: return "artifacts/model"
        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
        if not runs: return "artifacts/model"
        # Use the idiomatic MLflow runs:/ scheme
        run_id = runs[0].info.run_id
        return f"runs:/{run_id}/model"
    except Exception as e:
        print(f"Error fetching latest model URI from MLflow: {e}")
        return "artifacts/model"

MODEL_URI = os.getenv("MODEL_URI", get_latest_model_uri())
FEATURE_COLS_PATH = "artifacts/feature_columns.json"
LIMITS_PATH = "artifacts/preprocessing_limits.pkl"

model = None
feature_cols = None
limits = None
THRESHOLD = 0.35

@app.on_event("startup")
def load_artifacts():
    global model, feature_cols, limits
    try:
        # Load model using MLflow
        print(f"Loading model from {MODEL_URI}")
        model = mlflow.xgboost.load_model(MODEL_URI)
            
        if os.path.exists(FEATURE_COLS_PATH):
            with open(FEATURE_COLS_PATH, "r") as f:
                feature_cols = json.load(f)
                
        if os.path.exists(LIMITS_PATH):
            with open(LIMITS_PATH, "rb") as f:
                limits = pickle.load(f)
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: CreditRequest):
    if model is None or feature_cols is None or limits is None:
        details = {
            "model_loaded": model is not None,
            "feature_cols_loaded": feature_cols is not None,
            "limits_loaded": limits is not None,
            "MODEL_URI": MODEL_URI
        }
        raise HTTPException(status_code=503, detail=f"Model artifacts not loaded properly. Debug: {details}")
        
    # Convert request to DataFrame
    df_raw = pd.DataFrame([request.model_dump()])
    
    # Preprocess
    df_clean = preprocess_data(df_raw)
    
    # Feature Engineering
    df_eng, _ = build_features(df_clean, is_training=False, limits=limits)
    
    # Ensure correct feature order
    # Fill missing columns if any
    for col in feature_cols:
        if col not in df_eng.columns:
            df_eng[col] = 0
    X = df_eng[feature_cols]
    
    # Predict
    prob = model.predict_proba(X)[0][1]
    pred = 1 if prob >= THRESHOLD else 0
    risk = "High Risk" if pred == 1 else "Low Risk"
    
    return PredictionResponse(probability=float(prob), prediction=pred, risk_level=risk)
