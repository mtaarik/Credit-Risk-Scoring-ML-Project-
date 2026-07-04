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
