import pandas as pd
import logging
import json

logger = logging.getLogger(__name__)

PAY_STATUS_COLS = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL_AMT_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
PAY_AMT_COLS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

def cap_outliers(series: pd.Series, percentile: float = 0.99) -> pd.Series:
    """Caps outliers at the specified percentile."""
    limit = series.quantile(percentile)
    return series.clip(upper=limit)

def build_features(df: pd.DataFrame, is_training: bool = True, limits: dict = None) -> tuple:
    """
    Builds new features and applies winsorization.
    If is_training is True, calculates and returns the limits for winsorization.
    If is_training is False, uses the provided limits.
    """
    logger.info("Starting feature engineering...")
    df = df.copy()

    # 1. Feature Engineering
    df['SD_COUNT'] = df[PAY_STATUS_COLS].apply(lambda x: (x >= 2).sum(), axis=1)

    df['TOTAL_PAY_RATIO'] = df[PAY_AMT_COLS].sum(axis=1) / (df[BILL_AMT_COLS].sum(axis=1) + 1)
    mask_negative_bill = df[BILL_AMT_COLS].sum(axis=1) <= 0
    df.loc[mask_negative_bill, 'TOTAL_PAY_RATIO'] = 1.0
    df['TOTAL_PAY_RATIO'] = df['TOTAL_PAY_RATIO'].clip(lower=0, upper=10)

    df['UTILIZATION_RATIO'] = df['BILL_AMT1'] / df['LIMIT_BAL']
    df['UTILIZATION_RATIO'] = df['UTILIZATION_RATIO'].clip(lower=0)

    # 2. Outlier Management (Winsorization at 99th percentile)
    cols_to_cap = ['LIMIT_BAL'] + BILL_AMT_COLS + PAY_AMT_COLS
    
    if is_training:
        limits = {}
        for col in cols_to_cap:
            limit = df[col].quantile(0.99)
            limits[col] = limit
            df[col] = df[col].clip(upper=limit)
        
        logger.info("Feature engineering and winsorization completed.")
        return df, limits
    else:
        if limits is None:
            raise ValueError("limits dictionary must be provided during inference")
        for col in cols_to_cap:
            if col in limits:
                df[col] = df[col].clip(upper=limits[col])
        return df, limits
