import pandas as pd
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe.
    - Corrects EDUCATION and MARRIAGE categories
    """
    logger.info("Starting data preprocessing...")
    df = df.copy()

    # Rename columns to standard names if they exist (handles both CSV and API inputs)
    rename_cols = {}
    if 'default.payment.next.month' in df.columns:
        rename_cols['default.payment.next.month'] = 'TARGET'
    if 'PAY_0' in df.columns:
        rename_cols['PAY_0'] = 'PAY_1'
    
    if rename_cols:
        df = df.rename(columns=rename_cols)

    # Correct EDUCATION
    # 0, 5, 6 -> 4 (Others)
    fill_edu = (df['EDUCATION'] == 0) | (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6)
    df.loc[fill_edu, 'EDUCATION'] = 4

    # Correct MARRIAGE
    # 0 -> 3 (Others)
    fill_mar = (df['MARRIAGE'] == 0)
    df.loc[fill_mar, 'MARRIAGE'] = 3

    logger.info("Data preprocessing completed.")
    return df
