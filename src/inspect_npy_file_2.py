import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

# Setup logger
def setup_logger():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs_3/{timestamp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup logging
    log_file = os.path.join(output_dir, "analysis.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, output_dir

# Check data basic information
def verify_data(file_path, logger):
    logger.info(f"Verifying data file: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    
    if np.issubdtype(data.dtype, np.floating):
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        zero_count = (data == 0).sum()
        
        logger.info(f"NaN count: {nan_count}")
        logger.info(f"Infinity count: {inf_count}")
        logger.info(f"Zero count: {zero_count}")
        
        if nan_count > 0:
            logger.warning(f"Warning: File {file_path} contains {nan_count} NaN values")
        if inf_count > 0:
            logger.warning(f"Warning: File {file_path} contains {inf_count} infinite values")
    
    return data

# Analyze stock features correlation (averaged across all stocks)
def analyze_stocks_data(stocks_data, output_dir, logger):
    try:
        # Take average across stocks dimension
        avg_stock_data = np.mean(stocks_data, axis=0)
        
        # Create feature names
        num_features = stocks_data.shape[2]
        feature_names = [f"F{i+1}" for i in range(num_features)]
        
        # Create DataFrame
        df = pd.DataFrame()
        for i in range(num_features):
            df[feature_names[i]] = avg_stock_data[:, i]
        
        # Calculate correlation
        corr_matrix = df.corr(method='pearson')
        
        # Plot heatmap (simple version, no annotations)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", 
                    vmin=-1, vmax=1, square=True)
        plt.title("Stock Features Correlation (Averaged)")
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "stock_features_corr.png"))
        plt.close()
        
        logger.info("Generated stock features correlation heatmap (averaged across all stocks)")
            
    except Exception as e:
        logger.error(f"Error processing stock data: {str(e)}")

# Analyze market data correlation
def analyze_market_data(market_data, output_dir, logger):
    try:
        # Create feature names
        num_features = market_data.shape[1]
        feature_names = [f"MF{i+1}" for i in range(num_features)]
        
        # Create DataFrame
        df = pd.DataFrame()
        for i in range(num_features):
            df[feature_names[i]] = market_data[:, i]
        
        # Check for NaN values, fill if necessary
        if df.isna().any().any():
            logger.warning("Market data contains NaN values, filling with mean values")
            df = df.fillna(df.mean())
        
        # Calculate correlation
        corr_matrix = df.corr(method='pearson')
        
        # Plot heatmap (simple version, no annotations)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", 
                    vmin=-1, vmax=1, square=True)
        plt.title("Market Features Correlation")
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "market_features_corr.png"))
        plt.close()
        
        logger.info("Generated market features correlation heatmap")
                    
    except Exception as e:
        logger.error(f"Error analyzing market data: {str(e)}")

# Analyze relation matrix
def analyze_relation_matrix(relation_matrix, output_dir, logger):
    try:
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(relation_matrix, annot=False, cmap="viridis", square=True)
        plt.title("Industry Classification Matrix")
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "industry_classification.png"))
        plt.close()
        
        logger.info("Generated industry classification matrix heatmap")
        
    except Exception as e:
        logger.error(f"Error analyzing relation matrix: {str(e)}")

# Main function
def main():
    DATA_PATH = r"data\DJIA\feature33-fill"
    # DATA_PATH = r"data\DJIA"

    num_train_days = 2086  # train = [0:2086), test = [2086:6260)

    # Setup logger and output directory
    logger, output_dir = setup_logger()
    logger.info(f"Starting data analysis, source path: {DATA_PATH}")
    
    # Data file paths
    stocks_data_path = os.path.join(DATA_PATH, "stocks_data.npy")
    market_data_path = os.path.join(DATA_PATH, "market_data.npy")
    ror_data_path = os.path.join(DATA_PATH, "ror.npy")
    relation_matrix_path = os.path.join(DATA_PATH, "industry_classification.npy")
    
    try:
        # Check if files exist
        for file_path in [stocks_data_path, market_data_path, ror_data_path, relation_matrix_path]:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return
        
        # Load and verify data
        logger.info("Loading and verifying data files")
        stocks_data = verify_data(stocks_data_path, logger)
        market_data = verify_data(market_data_path, logger)
        ror_data = verify_data(ror_data_path, logger)
        relation_matrix = verify_data(relation_matrix_path, logger)

        # 切出 train / test
        # stocks_data shape: (num_stocks, num_days, num_features)
        stocks_data_train = stocks_data[:, :num_train_days, :]
        stocks_data_test  = stocks_data[:, num_train_days:, :]

        # market_data shape: (num_days, num_market_features)
        market_data_train = market_data[:num_train_days, :]
        market_data_test  = market_data[num_train_days:, :]

        # ror_data shape: (num_days, ...)
        ror_data_train = ror_data[:num_train_days]
        ror_data_test  = ror_data[num_train_days:]

        # relation_matrix 不含時間維度，不需切分
        # relation_matrix_train = relation_matrix
        
        # Analyze stock data (averaged)
        logger.info("Analyzing stock features correlation (averaged)")
        analyze_stocks_data(stocks_data_train, output_dir, logger)
        
        # Analyze market data
        logger.info("Analyzing market features correlation")
        analyze_market_data(market_data_train, output_dir, logger)
        
        # Analyze relation matrix
        logger.info("Analyzing industry classification matrix")
        analyze_relation_matrix(relation_matrix, output_dir, logger)
        
        logger.info("Data analysis complete")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()