import os
import sys
import json
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.xgboost

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Credit Risk Scoring")

    data_path = os.path.join("data", "raw", "UCI_Credit_Card.csv")
    
    with mlflow.start_run():
        logger.info("Pipeline started.")
        
        # 2. Data Loading
        try:
            df = load_data(data_path)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return

        # 3. Data Preprocessing
        df_clean = preprocess_data(df)
        
        # 4. Target / Features Separation
        X = df_clean.drop('TARGET', axis=1) # Target column is TARGET based on notebook
        # Drop ID if it exists
        if 'ID' in X.columns:
            X = X.drop('ID', axis=1)
        y = df_clean['TARGET']

        # 5. Train/Test Split (Stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        
        # 6. Feature Engineering & Winsorization
        X_train_eng, limits = build_features(X_train, is_training=True)
        X_test_eng, _ = build_features(X_test, is_training=False, limits=limits)
        
        # Save feature ordering and limits
        feature_cols = list(X_train_eng.columns)
        os.makedirs("artifacts", exist_ok=True)
        with open(os.path.join("artifacts", "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        with open(os.path.join("artifacts", "preprocessing_limits.pkl"), "wb") as f:
            pickle.dump(limits, f)
            
        mlflow.log_artifact(os.path.join("artifacts", "feature_columns.json"))
        mlflow.log_artifact(os.path.join("artifacts", "preprocessing_limits.pkl"))

        # 7. Model Training
        hyperparameters = {
            'n_estimators': 301,
            'learning_rate': 0.034,
            'max_depth': 7,
            'subsample': 0.95,
            'colsample_bytree': 0.98,
            'random_state': 42
        }
        mlflow.log_params(hyperparameters)
        
        model = train_model(X_train_eng, y_train, params=hyperparameters)
        
        # 8. Evaluation
        threshold = 0.35
        mlflow.log_param("decision_threshold", threshold)
        
        metrics = evaluate_model(model, X_test_eng, y_test, threshold=threshold)
        mlflow.log_metrics(metrics)
        
        # 9. Log Model
        mlflow.xgboost.log_model(model, artifact_path="model")
        
        logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
