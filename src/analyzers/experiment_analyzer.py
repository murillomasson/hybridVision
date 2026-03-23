import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_and_plot(results_csv_path, output_plot_path):
    if not os.path.exists(results_csv_path):
        logging.error(f"Results file not found at: {results_csv_path}")
        return

    logging.info(f"Loading results from: {results_csv_path}")
    study_df = pd.read_csv(results_csv_path)
    
    param_cols = [col for col in study_df.columns if col.startswith('params_')]
    user_attrs_cols = [col for col in study_df.columns if col.startswith('user_attrs_')]
    
    analysis_df = study_df[['value'] + param_cols + user_attrs_cols]
    
    analysis_df.columns = analysis_df.columns.str.replace('params_', '').str.replace('user_attrs_', '')
    
    numeric_df = analysis_df.select_dtypes(include=np.number)
    
    correlation_matrix = numeric_df.corr()
    
    if 'value' not in correlation_matrix:
        logging.warning("'value' column not found in correlation matrix. Skipping heatmap generation.")
        return
    
    plt.figure(figsize=(20, 24))
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5, 
        annot_kws={"size": 8}
    )
    plt.title('Full Correlation Matrix of Hyperparameters and Performance Metrics')
    plt.tight_layout()
    
    plt.savefig(output_plot_path)
    logging.info(f"Correlation heatmap saved to: {output_plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from an Optuna tuning run.")
    parser.add_argument('--results', type=str, required=True, help='Path to the tuner_results.csv file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output correlation heatmap.')
    args = parser.parse_args()
    
    analyze_and_plot(args.results, args.output)
    