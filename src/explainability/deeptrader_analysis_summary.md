# DeepTrader 行為分析結果

## 3-Class Classification Analysis (analyze_deeptrader_behavior.py)

### 分析方法概述
使用 Random Forest 分類器分析 DeepTrader 的交易行為，將每支股票的每個時間窗口分類為：
- **Buy (1)**: 做多（長倉）
- **Hold (0)**: 不交易（既無多空倉位）
- **Sell (-1)**: 做空（短倉）

### 資料結構與特徵提取

#### 原始資料
- **stocks_data.npy**: (30支股票, 2782個時間點, 34個特徵)
- **訓練資料**: val_results.json (驗證集結果)
- **測試資料**: test_results.json (測試集結果)

#### 特徵提取過程
```python
# 對每個交易步驟和每支股票：
1. 提取時間窗口: stocks_data[stock_idx, input_start:input_end, :] → (70, 34)
2. 展平為一維: flatten() → (70×34 = 2380,)
3. 標記交易動作: 
   - 在 long_positions 中 → Buy (1)
   - 在 short_positions 中 → Sell (-1)
   - 都不在 → Hold (0)
```

### 模型訓練與評估結果

#### 訓練集表現（嚴重過擬合）
```
Training Performance:
              precision    recall  f1-score   support
        Sell       1.00      1.00      1.00       224
        Hold       1.00      1.00      1.00      1232
         Buy       1.00      1.00      1.00       224
    accuracy                           1.00      1680
```

#### 測試集表現
```
Test Performance:
              precision    recall  f1-score   support
        Sell       1.00      0.05      0.10        56
        Hold       0.77      0.95      0.85       308
         Buy       0.58      0.34      0.43        56
    accuracy                           0.75       420
   macro avg       0.78      0.45      0.46       420
weighted avg       0.77      0.75      0.69       420
```

#### 混淆矩陣分析
```
實際\預測   Sell  Hold  Buy
Sell         3    53    0    # 56個賣出信號中，只有3個被正確識別
Hold         0   294   14    # 308個持有信號中，294個正確
Buy          0    37   19    # 56個買入信號中，只有19個正確
```

### 分類指標解釋

#### Precision（精確率）
- **定義**: 預測為某類別中，實際確實是該類別的比例
- **Sell: 1.00** = 3/3，模型預測的3個賣出全部正確
- **Hold: 0.77** = 294/(0+294+37)，預測持有的準確率77%
- **Buy: 0.58** = 19/(0+14+19)，預測買入的準確率58%

#### Recall（召回率）
- **定義**: 實際為某類別中，被正確預測出來的比例
- **Sell: 0.05** = 3/56，只找出5%的賣出信號
- **Hold: 0.95** = 294/308，找出95%的持有信號
- **Buy: 0.34** = 19/56，只找出34%的買入信號

#### F1-score
- **定義**: Precision和Recall的調和平均數
- **公式**: 2 × (Precision × Recall) / (Precision + Recall)
- 綜合反映模型在該類別的整體表現

#### Macro vs Weighted Average
- **Macro avg**: 三個類別指標的簡單平均，每類權重相等
- **Weighted avg**: 按樣本數量加權平均，反映整體表現
- **Accuracy**: 所有預測中正確的比例 = (3+294+19)/420 = 0.75

### 特徵重要性分析

#### 時間維度重要性（前10名）
```
Time 69 (0 days ago): 0.0237   # 最近一天最重要
Time 68 (1 days ago): 0.0206   # 前一天次之
Time 40 (29 days ago): 0.0196  # 一個月前也重要
Time 36 (33 days ago): 0.0195
...
```
- **發現**: 最近幾天的資料對預測影響最大
- **總和**: 所有2380個特徵的重要性總和 = 1.0

#### 時間-特徵組合分析
1. **Top 20 Most Important Time-Feature Combinations**
   - 識別2380個特徵中最重要的20個
   - 格式：「第X天的第Y個特徵」

2. **Feature Patterns by Action**
   - 分析不同交易動作的特徵模式
   - 計算Buy/Hold/Sell樣本在重要特徵上的平均值
   - 幫助理解什麼特徵值導致什麼交易決策

### 關鍵問題與發現

#### 1. 嚴重的類別不平衡
- Hold佔73.3% (1540/2100)
- Buy佔13.3% (280/2100)
- Sell佔13.3% (280/2100)

#### 2. 模型偏向性
- 強烈偏向預測Hold（保守策略）
- 幾乎無法識別Sell信號（recall只有5%）
- Buy信號識別能力也不佳（recall只有34%）

#### 3. 過擬合問題
- 訓練集100%準確率 vs 測試集75%
- Random Forest的max_depth=20可能太深
- 需要更強的正則化

#### 4. 實務影響
- 會錯過95%的賣出機會
- 會錯過66%的買入機會
- 這樣的模型在實際交易中效果不佳

### 改進建議

1. **處理類別不平衡**
   - 使用SMOTE等過採樣技術
   - 調整class_weight參數
   - 考慮成本敏感學習

2. **減少過擬合**
   - 降低max_depth（如10-15）
   - 增加min_samples_split
   - 使用交叉驗證選擇參數

3. **特徵工程**
   - 使用滑動窗口統計特徵而非原始展平
   - 加入技術指標
   - 考慮時間序列特徵

4. **模型選擇**
   - 嘗試XGBoost/LightGBM
   - 考慮時序模型如LSTM
   - 使用集成方法

5. **評估指標**
   - 關注交易類別(Buy/Sell)的recall
   - 使用自定義損失函數
   - 考慮交易成本和風險