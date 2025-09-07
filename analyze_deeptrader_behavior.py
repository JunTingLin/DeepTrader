#!/usr/bin/env python3
"""
Analyze DeepTrader behavior using improved approach:
1. Keep 2D structure (time_steps, features) instead of flattening
2. Use multi-label classification approach
"""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
import pandas as pd

def load_data():
    """Load stocks data, val results (train) and test results (test)"""
    stocks_data = np.load('src/data/DJIA/feature34-Inter-P532-0831/stocks_data.npy')
    
    # Load validation results as training data
    with open('src/outputs/0906/023903/json_file/val_results.json') as f:
        val_results = json.load(f)
    
    # Load test results as test data  
    with open('src/outputs/0906/023903/json_file/test_results.json') as f:
        test_results = json.load(f)
    
    return stocks_data, val_results, test_results

def extract_features_and_labels(stocks_data, test_results):
    """
    Extract features keeping time structure and use multi-label approach
    """
    X_list = []  # 每個樣本是 (time_steps, features)
    Y_list = []  # 每個樣本是 30維二元向量
    
    for step_data in test_results['portfolio_records']:
        input_start = step_data['input_start']
        input_end = step_data['input_end']
        
        # Get window data: (30, time_steps, 34)
        window_data = stocks_data[:, input_start:input_end, :]
        
        # Average across stocks: (time_steps, 34) - 保持2D結構
        window_features = np.mean(window_data, axis=0)
        
        # 使用統計特徵降維到1D
        # 對時間軸計算統計量：mean, std, min, max, 最近值
        time_stats = []
        time_stats.append(np.mean(window_features, axis=0))  # 34個特徵的時間平均
        time_stats.append(np.std(window_features, axis=0))   # 34個特徵的時間標準差
        time_stats.append(np.min(window_features, axis=0))   # 34個特徵的時間最小值
        time_stats.append(np.max(window_features, axis=0))   # 34個特徵的時間最大值
        time_stats.append(window_features[-1])               # 34個特徵的最近值
        time_stats.append(window_features[-5])               # 34個特徵的T-5值
        time_stats.append(window_features[-10])              # 34個特徵的T-10值
        
        # 合併統計特徵: 7 * 34 = 238個特徵
        features = np.concatenate(time_stats)
        X_list.append(features)
        
        # Create Y label vector (30 stocks)
        y_label = np.zeros(30, dtype=int)
        for pos in step_data['long_positions']:
            stock_idx = pos['stock_index']
            y_label[stock_idx] = 1
        Y_list.append(y_label)
    
    X = np.array(X_list)  # (33, 238)
    Y = np.array(Y_list)  # (33, 30)
    
    print(f"Features X shape: {X.shape}")
    print(f"Labels Y shape: {Y.shape}")
    print(f"Average stocks selected per step: {np.mean(np.sum(Y, axis=1)):.2f}")
    
    return X, Y

def train_multi_label_model(X_train, Y_train, X_test, Y_test):
    """Train multi-label classifier with proper train/test split"""
    # 檢查哪些股票在訓練集中被選中過
    train_selected_stocks = np.sum(Y_train, axis=0) > 0
    test_selected_stocks = np.sum(Y_test, axis=0) > 0
    
    print(f"Stocks ever selected in training: {np.sum(train_selected_stocks)} out of 30")
    print(f"Training selected stock indices: {np.where(train_selected_stocks)[0]}")
    
    print(f"Stocks ever selected in testing: {np.sum(test_selected_stocks)} out of 30")
    print(f"Testing selected stock indices: {np.where(test_selected_stocks)[0]}")
    
    # 只訓練被選中過的股票，避免無正樣本問題
    Y_train_filtered = Y_train[:, train_selected_stocks]
    Y_test_filtered = Y_test[:, train_selected_stocks]
    stock_indices = np.where(train_selected_stocks)[0]
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Selected stocks for training: {Y_train_filtered.shape[1]}")
    
    # 使用MultiOutputClassifier進行多標籤分類
    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    )
    
    # 訓練模型
    model.fit(X_train, Y_train_filtered)
    
    # 在測試集上預測和評估
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    print(f"\n=== Training Performance ===")
    print(f"Hamming Loss: {hamming_loss(Y_train_filtered, Y_train_pred):.4f}")
    
    print(f"\n=== Test Performance ===")
    print(f"Hamming Loss: {hamming_loss(Y_test_filtered, Y_test_pred):.4f}")
    
    # 計算每個股票在測試集上的準確率
    print(f"\n=== Per-stock Test Accuracy ===")
    for i, stock_idx in enumerate(stock_indices):
        train_accuracy = np.mean(Y_train_filtered[:, i] == Y_train_pred[:, i])
        test_accuracy = np.mean(Y_test_filtered[:, i] == Y_test_pred[:, i])
        train_selection_rate = np.mean(Y_train_filtered[:, i])
        test_selection_rate = np.mean(Y_test_filtered[:, i])
        
        print(f"Stock {stock_idx}: train_acc={train_accuracy:.3f}, test_acc={test_accuracy:.3f}, "
              f"train_sel={train_selection_rate:.2%}, test_sel={test_selection_rate:.2%}")
    
    return model, train_selected_stocks, stock_indices

def analyze_feature_importance(model, train_selected_stocks, stock_indices, n_time_stats=7, n_features=34):
    """分析多標籤模型的特徵重要性"""
    
    # 收集所有分類器的特徵重要性
    importances_list = []
    
    for i, estimator in enumerate(model.estimators_):
        importances_list.append(estimator.feature_importances_)
    
    # 平均特徵重要性
    avg_importances = np.mean(importances_list, axis=0)  # (238,)
    
    # 重新組織為 (7統計量, 34特徵)
    importance_matrix = avg_importances.reshape(n_time_stats, n_features)
    
    # 統計量重要性
    stat_names = ['Mean', 'Std', 'Min', 'Max', 'T-1', 'T-5', 'T-10']
    stat_importance = np.sum(importance_matrix, axis=1)
    
    print("\n=== 時間統計量重要性分析 ===")
    for i, (name, imp) in enumerate(zip(stat_names, stat_importance)):
        print(f"{name}: {imp:.4f}")
    
    # 特徵重要性（跨所有統計量）
    feature_importance = np.sum(importance_matrix, axis=0)
    
    print("\n=== Top 10 最重要特徵 ===")
    top_features = np.argsort(feature_importance)[::-1][:10]
    for i, feat_idx in enumerate(top_features):
        print(f"Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
    
    # 顯示每個統計量下的前5重要特徵
    print("\n=== 各統計量下的重要特徵 ===")
    for i, stat_name in enumerate(stat_names):
        top_feats_for_stat = np.argsort(importance_matrix[i, :])[::-1][:3]
        print(f"{stat_name}: Features {top_feats_for_stat} ({importance_matrix[i, top_feats_for_stat]})")
    
    return feature_importance, stat_importance, importance_matrix

def main():
    print("=== DeepTrader Behavior Analysis ===")

    print("Loading data...")
    stocks_data, val_results, test_results = load_data()
    
    print("Extracting training features and labels from val_results...")
    X_train, Y_train = extract_features_and_labels(stocks_data, val_results)
    
    print("Extracting test features and labels from test_results...")
    X_test, Y_test = extract_features_and_labels(stocks_data, test_results)
    
    print("Training multi-label model...")
    model, train_selected_stocks, stock_indices = train_multi_label_model(
        X_train, Y_train, X_test, Y_test
    )
    
    print("Analyzing feature importance...")
    feature_imp, stat_imp, imp_matrix = analyze_feature_importance(
        model, train_selected_stocks, stock_indices
    )
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()