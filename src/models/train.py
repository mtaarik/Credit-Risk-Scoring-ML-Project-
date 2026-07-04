import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import logging

logger = logging.getLogger(__name__)

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> XGBClassifier:
    """
    Trains an XGBoost classifier.
    Calculates scale_pos_weight dynamically to handle class imbalance.
    """
    logger.info("Starting model training (XGBoost)...")
    
    # Calculate scale_pos_weight for imbalanced classes
    # count(negative examples)/count(Positive examples)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    logger.info(f"Class imbalance handling: scale_pos_weight = {scale_pos_weight:.2f}")
    
    if params is None:
        params = {
            'n_estimators': 301,
            'learning_rate': 0.034,
            'max_depth': 7,
            'subsample': 0.95,
            'colsample_bytree': 0.98,
            'random_state': 42
        }
    
    params['scale_pos_weight'] = scale_pos_weight
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info("Model training completed.")
    return model
