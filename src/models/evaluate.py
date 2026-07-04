import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.35) -> dict:
    """
    Evaluates the model and returns metrics.
    Uses the specified decision threshold for predictions.
    """
    logger.info(f"Evaluating model with threshold {threshold}...")
    
    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Apply custom threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics
