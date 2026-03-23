import yaml
import argparse
import logging
import os
import json
import numpy as np
from datetime import datetime
from src.pipeline import Pipeline
from src.analyzers.performance_analyzer import PerformanceAnalyzer
from src.utils.experiment_logger import log_run_to_csv
from src.utils.io import load_pickle


def main():
    parser = argparse.ArgumentParser(description="Run and evaluate the HybridVision Pipeline.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config YAML file.")
    parser.add_argument("-gt", "--ground_truth", type=str, required=True, help="Path to the ground truth pickle file.")
    args = parser.parse_args()

    logging.info(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}"
    main_output_dir = config.get('output_dir', './output')
    run_output_dir = os.path.join(main_output_dir, run_name)
    config['output_dir'] = run_output_dir
    os.makedirs(run_output_dir, exist_ok=True)
    logging.info(f"Results for this run will be saved in: {run_output_dir}")

    with open(os.path.join(run_output_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f)

    try:
        pipeline = Pipeline(config)
        pipeline_results = pipeline.run()
        
        logging.info(f"Loading ground truth data from: {args.ground_truth}")
        ground_truth_data = load_pickle(args.ground_truth)
        
        analyzer = PerformanceAnalyzer(pipeline_results, ground_truth_data)
        detailed_metrics_df = analyzer.run_evaluation()
        
        detailed_csv_path = os.path.join(run_output_dir, "detailed_metrics.csv")
        detailed_metrics_df.to_csv(detailed_csv_path, index=False)
        logging.info(f"Saved detailed metrics to {detailed_csv_path}")

        performance_summary = analyzer.get_summary()
        summary_path = os.path.join(run_output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(performance_summary, f, indent=4)
        logging.info(f"Saved run summary to {summary_path}")

        master_log_path = os.path.join(main_output_dir, "master_log.csv")
        log_run_to_csv(config, performance_summary, master_log_path)
        
        logging.info("--- Experiment Run Finished Successfully ---")

    except Exception as e:
        logging.error(f"An error occurred during the run: {e}", exc_info=True)

if __name__ == "__main__":
    main()
    