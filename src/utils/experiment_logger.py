import pandas as pd
import os
import logging
from datetime import datetime

def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def log_run_to_csv(config: dict, performance_summary: dict, master_log_path: str):
    logging.info(f"Logging experiment summary to {master_log_path}...")
    try:
        run_info = {
            'run_timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'output_dir': config.get('output_dir')
        }
        
        flat_config = flatten_dict(config)
        
        log_entry = {**run_info, **performance_summary, **flat_config}
        new_row_df = pd.DataFrame([log_entry])

        if os.path.exists(master_log_path):
            master_df = pd.read_csv(master_log_path)
            master_df = pd.concat([master_df, new_row_df], ignore_index=True)
        else:
            master_df = new_row_df
            
        master_df.to_csv(master_log_path, index=False)
        logging.info("Successfully logged experiment summary.")

    except Exception as e:
        logging.error(f"Failed to log experiment to master CSV: {e}", exc_info=True)