import os
import pickle
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def save_pickle(obj, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"[OK] Saved {os.path.basename(path)}")
    except Exception as e:
        logging.error(f"Error saving pickle to {path}: {e}")

def load_pickle(path: str):
    if not os.path.exists(path):
        logging.warning(f"File {os.path.basename(path)} not found. Will regenerate.")
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logging.info(f"Loaded {type(obj)} from {os.path.basename(path)}.")
        return obj
    except Exception as e:
        logging.error(f"Error loading pickle from {path}: {e}")
        return None

def save_all_results_csv(df_combined: pd.DataFrame, path: str):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    df_combined.to_csv(path, index=False, encoding='utf-8')
    logging.info(f"[OK] All results saved to CSV: {path}")
    