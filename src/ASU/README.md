# ASU 監督式預訓練

監督式預訓練 ASU 模組，用於遷移學習到 DeepTrader。

## 快速開始

### 1. 訓練 ASU

```bash
# 訓練 GCN 版本 (TCN + SAGCN + Spatial Attention)
bash src/ASU/train_asu_gcn.sh

# 或訓練 TE2D 版本 (TCN + Transformer + Spatial Attention)
bash src/ASU/train_asu_te2d.sh
```

訓練完成後，權重保存在：
```
src/ASU/checkpoints/ASU_GCN_Combined_<timestamp>/
├── best_asu.pth          # 最佳權重（用於遷移學習）⭐
├── args.json             # 訓練參數（包含 timestamp）
├── test_results.json     # 測試結果
└── tensorboard/          # TensorBoard 日誌
```

### 2. 測試並生成評估 JSON

```bash
# 在測試集上評估
python src/ASU/test_asu_supervised.py \
    --checkpoint_path src/ASU/checkpoints/ASU_GCN_*/best_asu.pth \
    --period test \
    --k 4

# 在驗證集上評估
python src/ASU/test_asu_supervised.py \
    --checkpoint_path src/ASU/checkpoints/ASU_GCN_*/best_asu.pth \
    --period val \
    --k 4
```

生成結果：
```
src/outputs/asu_supervised_<timestamp>/
└── json_file/
    ├── test_results.json  # portfolio_records 格式 ⭐
    ├── val_results.json   # portfolio_records 格式
    └── training_args.json
```

**JSON 格式** (兼容 `src/plot/main.py`):
```json
{
  "portfolio_records": [
    {
      "step": 0,
      "time_idx": 1000,
      "long_positions": [
        {"stock_index": 5, "weight": 0.25, "score": 0.85, "rank": 1, "future_return": 0.12},
        ...
      ],
      "short_positions": [
        {"stock_index": 3, "weight": 0.25, "score": -0.32, "rank": 1, "future_return": -0.08},
        ...
      ]
    },
    ...
  ]
}
```

### 3. 使用 plot/main.py 評估

```bash
# 評估 ASU 監督學習的選股能力
python src/plot/main.py --exp_id asu_supervised_<timestamp> --period test
```

輸出：
- **Precision@K / Recall@K**: 選股準確率
- **Portfolio Heatmaps**: 每步的持倉視覺化
- **Future Return Heatmaps**: 實際收益對比
- **Stock Trends**: 個股表現分析

## 核心組件

### 文件結構

```
src/ASU/
├── losses_asu.py              # 損失函數（Ranking, Regression, Combined）
├── asu_dataset.py             # 數據加載器（周採樣，模仿 DataGenerator）
├── train_utils.py             # 共享訓練邏輯
├── train_asu_gcn.py           # GCN 版本訓練腳本（transformer_asu_bool=False）
├── train_asu_te2d.py          # TE2D 版本訓練腳本（transformer_asu_bool=True）
├── train_asu_gcn.sh           # GCN Shell 腳本
├── train_asu_te2d.sh          # TE2D Shell 腳本
├── test_asu_supervised.py     # 測試並生成 JSON
└── checkpoints/               # 訓練輸出

src/model/
└── ASU.py                     # 統一的 ASU 模型（支持 GCN/TE2D 切換）
```

### 損失函數

**Combined Loss（推薦）**:
```python
loss = 0.8 * PairwiseRankingLoss + 0.2 * RegressionLoss
```

- **PairwiseRankingLoss**: 學習股票相對排序（主要任務）
- **RegressionLoss**: 預測收益率數值（輔助任務）

其他選項：`Pairwise`, `IC`, `MSE`, `MAE`

### 數據格式

- **Input**: `[num_stocks, 13, num_features]` - 13 週的週五數據
- **Ground Truth**: 未來 21 天的累積收益率
- **Output**: `[num_stocks]` - 股票評分（用於排序）

數據預處理模仿 `DataGenerator`:
1. 取 70 天原始數據
2. Reshape 成 14 週 × 5 天
3. 取後 13 週的最後一天（週五）

### 數據採樣策略

- **Train**: `stride=1` - 密集採樣，最大化訓練數據量
- **Val**: `stride=21` - 無重疊，模擬實際交易週期
- **Test**: `stride=21` - 無重疊，每 21 天評估一次

**為何 Val/Test 用 stride=21?**
1. 避免數據洩漏和過於樂觀的驗證指標
2. 匹配 DeepTrader 的 `trade_len=21` 交易週期
3. 兼容 `src/plot/main.py` 評估管道

## 遷移學習到 DeepTrader

### 方案 1: 預訓練初始化 + 微調（推薦）

```python
import torch
from model.ASU import ASU

# 1. 創建 ASU
asu = ASU(num_nodes=30, in_features=34, hidden_dim=128, ...)

# 2. 載入預訓練權重
checkpoint = torch.load('src/ASU/checkpoints/ASU_GCN_*/best_asu.pth')
asu.load_state_dict(checkpoint['asu_state_dict'])
print(f"✓ Loaded pretrained ASU (Val IC: {checkpoint['val_ic']:.4f})")

# 3. 使用小學習率微調
optimizer = torch.optim.Adam([
    {'params': asu.parameters(), 'lr': 1e-6},       # ASU: 小 lr
    {'params': other_params, 'lr': 1e-4}            # 其他: 正常 lr
])
```

### 方案 2: 固定權重

```python
# 載入後凍結
checkpoint = torch.load('src/ASU/checkpoints/ASU_GCN_*/best_asu.pth')
asu.load_state_dict(checkpoint['asu_state_dict'])

for param in asu.parameters():
    param.requires_grad = False
```

## 配置參數

### 訓練參數（在 .sh 中修改）

```bash
# 損失函數
LOSS_FUNCTION="Combined"     # Pairwise, IC, MSE, Combined
ALPHA=0.8                    # Ranking 權重
BETA=0.2                     # Regression 權重

# 數據
WINDOW_LEN=13                # 輸入窗口（週）
HORIZON=21                   # 未來收益期（天）

# 模型
HIDDEN_DIM=128
NUM_BLOCKS=4
SPATIAL_BOOL=false           # 啟用空間注意力

# 訓練
EPOCHS=100
BATCH_SIZE=32
LR=0.0001
PATIENCE=20
```

## 評估指標

### Information Coefficient (IC)

- **定義**: 預測 scores 和實際收益的 Spearman 相關性
- **目標**: IC > 0.10 (較強預測能力)
- **最佳**: IC > 0.15 (優秀)

### 選股準確度（from plot/main.py）

- **Precision@K**: 選中的 K 支股票中，方向正確的比例
- **Recall@K**: 實際表現最好/最差的 K 支股票中，被選中的比例

## 常見問題

### Q: GCN 和 TE2D 版本有什麼區別？

- **GCN** (`transformer_asu_bool=False`): 使用 GraphConvNet 處理股票關係
- **TE2D** (`transformer_asu_bool=True`): 使用 Transformer (ViT-like) 處理股票關係
- 兩者都使用 `src/model/ASU.py`，只是參數不同
- TE2D 通常性能更好，但計算量更大

### Q: Combined Loss 的權重如何選擇？

- 推薦 `alpha=0.8, beta=0.2`（主要關注排序）
- 如果更關注數值預測，可以用 `0.5/0.5`

### Q: 預訓練真的有幫助嗎？

理論上可以帶來：
- 訓練加速: 20-40%
- 性能提升: 10-20%
- 更穩定的訓練

**需要實驗驗證！**

### Q: 時間戳如何對應？

訓練時生成的 `timestamp` 會保存在：
- `args.json` (檢查點目錄)
- `best_asu.pth` (checkpoint)
- 測試時自動讀取並用於命名輸出目錄

## 工作流程

```
1. 訓練 ASU
   bash src/ASU/train_asu_gcn.sh
   ↓
   生成: src/ASU/checkpoints/ASU_GCN_<timestamp>/best_asu.pth

2. 測試並生成 JSON
   python src/ASU/test_asu_supervised.py --checkpoint_path ... --period test
   ↓
   生成: src/outputs/asu_supervised_<timestamp>/json_file/test_results.json
   格式: portfolio_records (兼容 src/plot/main.py)

3. 評估選股能力
   python src/plot/main.py --exp_id asu_supervised_<timestamp> --period test
   ↓
   輸出: Precision@K, Recall@K, 熱力圖, 趨勢分析

4. 遷移到 DeepTrader（可選）
   在 agent.py 中載入預訓練權重
   ↓
   對比實驗: 有/無預訓練的性能差異
```

## 技術細節

### 周採樣實現

```python
# 取 70 天數據
raw = stocks_data[:, t - 70 + 1:t + 1, :]  # [num_stocks, 70, features]

# Reshape 成 14 週 × 5 天
raw = raw.reshape(num_stocks, 14, 5, features)

# 取後 13 週的週五
window = raw[:, 1:, -1, :]  # [num_stocks, 13, features]
```

與 `DataGenerator._get_data()` 第 184-191 行完全一致。

### JSON 格式

生成的 `test_results.json` 格式：
```json
{
  "experiment_type": "asu_supervised",
  "steps": [
    {
      "step": 0,
      "time_idx": 2087,
      "long_indices": [2, 5, 12, 18],
      "short_indices": [7, 9, 24, 28],
      "all_scores": [...],
      "actual_returns": [...]
    },
    ...
  ]
}
```

與 `src/outputs/random/01/json_file/test_results_random.json` 兼容。

---

**作者**: DeepTrader Team
**日期**: 2026-01-02
**版本**: 2.0 (重新設計版)
