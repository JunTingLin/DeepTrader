import numpy as np
import pandas as pd

def verify_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    
    print(f"File Name: {file_path}")
    print(f"Data Shape: {data.shape}")
    print(f"Data Type: {data.dtype}")

    # 檢查 NaN 和 Inf 值
    print(f"Checking {file_path}")
    print("NaN count:", np.isnan(data).sum() if np.issubdtype(data.dtype, np.floating) else "Not Applicable")
    print("Inf count:", np.isinf(data).sum() if np.issubdtype(data.dtype, np.floating) else "Not Applicable")

    print("\n")


stocks_data_file_path = r"data\DJIA\stocks_data.npy"
market_data_file_path = r"data\DJIA\market_data.npy"
ror_file_path = r"data\DJIA\ror.npy"
industry_classification_file_path = r"data\DJIA\industry_classification.npy"


verify_data(stocks_data_file_path)
verify_data(market_data_file_path)
verify_data(ror_file_path)
verify_data(industry_classification_file_path)



