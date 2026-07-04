import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw dataset from the given filepath.
    """
    logger.info(f"Loading data from {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df
