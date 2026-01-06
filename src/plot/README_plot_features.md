# Feature Plotting Tool

這個工具可以繪製 ASU (Agent State Unit - 個股特徵) 和 MSU (Market State Unit - 市場特徵) 的數據。

## 功能特點

1. **MSU (市場特徵) 繪圖**
   - 按類別分組的正規化特徵圖 (債券/國庫券、道瓊指數、黃金、VIX、標普500)
   - 所有 27 個特徵的網格圖 (原始尺度)

2. **ASU (個股特徵) 繪圖**
   - 每支股票的詳細特徵圖 (OHLCV、技術指標、Alpha 因子)
   - 跨股票的特徵比較圖 (Close、RSI、Alpha001 等)

## 使用方法

### 基本用法

```bash
# 使用 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DeepTrader-pip

# 繪製所有圖表 (MSU + ASU)
python src/plot/plot_features.py --data_dir src/data/DJIA/feature34-Inter-P532

# 只繪製市場特徵 (MSU)
python src/plot/plot_features.py --data_dir src/data/DJIA/feature34-Inter-P532 --plot_type msu

# 只繪製個股特徵 (ASU)
python src/plot/plot_features.py --data_dir src/data/DJIA/feature34-Inter-P532 --plot_type asu
```

### 指定個別檔案

```bash
# 使用個別檔案路徑
python src/plot/plot_features.py \
    --market_file src/data/DJIA/feature34-Inter-P532/market_data.npy \
    --stocks_file src/data/DJIA/feature34-Inter-P532/stocks_data.npy \
    --output_dir src/plot/plot_outputs/my_plots
```

### 自訂參數

```bash
# 指定輸出目錄和起始日期
# 注意: feature34-Inter-P532 的預設日期是 2015-01-01，無需手動指定
python src/plot/plot_features.py \
    --data_dir src/data/DJIA/feature34-Inter-P532 \
    --output_dir src/plot/plot_outputs/features

# 如果使用其他資料集，可以手動指定起始日期
python src/plot/plot_features.py \
    --data_dir src/data/DJIA/other_dataset \
    --start_date 2000-01-01
```

## 輸出檔案

輸出檔案名稱會根據正規化方法添加後綴：
- `_raw` - 原始數據 (`--normalize none`)
- `_normalized` - Min-Max 正規化 (`--normalize minmax`)
- `_zscore` - Z-Score 正規化 (`--normalize zscore`)

### MSU 圖表
- `msu_features_by_category_<suffix>.png` - 按類別分組的市場特徵
- `msu_features_grid.png` - 所有市場特徵的網格圖 (總是使用原始尺度)

### ASU 圖表
- `asu_features_<STOCK_NAME>_<suffix>.png` - 每支股票的詳細特徵圖 (共 30 張)
- `asu_comparison_Close_<suffix>.png` - 所有股票的收盤價比較
- `asu_comparison_RSI_<suffix>.png` - 所有股票的 RSI 比較
- `asu_comparison_Alpha001_<suffix>.png` - 所有股票的 Alpha001 比較

## 特徵說明

### MSU Features (27 個市場特徵)

1-6: **債券/國庫券指數**
- BAMLCC0A4BBBTRIV, BAMLCC0A0CMTRIV, BAMLCC0A1AAATRIV, BAMLHYH0A3CMTRIV, DGS10, DGS30

7-12: **道瓊工業指數 (DJI)**
- Open, High, Low, Close, Adj Close, Volume

13-16: **黃金 (xauusd)**
- Open, High, Low, Close

17-21: **波動率指數 (VIX)**
- Open, High, Low, Close, Adj Close

22-27: **標普 500 (GSPC)**
- Open, High, Low, Close, Adj Close, Volume

### ASU Features (34 個個股特徵)

1-5: **OHLCV**
- Open, High, Low, Close, Volume

6-14: **技術指標**
- MA20, MA60, RSI, MACD_Signal, K, D, BBands_Upper, BBands_Middle, BBands_Lower

15-34: **Alpha 因子**
- Alpha001 到 Alpha085 (共 20 個 alpha 因子)

## 支援的股票

預設支援 DJIA 28 支成份股:
AAPL, AMGN, AXP, BA, CAT, CSCO, CVX, DIS, GS, HD, HON, HPQ, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, PFE, PG, TRV, UNH, VZ, WBA

## 正規化方法

**重要**: DeepTrader 在訓練時使用 **window-based Z-score normalization**，即在每個 13 週的 window 內部做正規化，而不是對整個時間序列正規化。

繪圖工具提供三種正規化選項：

```bash
# 1. 原始數據 (預設，推薦用於觀察實際值)
python src/plot/plot_features.py --data_dir src/data/DJIA/feature34-Inter-P532 --normalize none

# 2. Min-Max 正規化 (0-1，方便比較不同特徵)
python src/plot/plot_features.py --data_dir src/data/DJIA/feature34-Inter-P532 --normalize minmax

# 3. Z-Score 正規化 (整個時間序列，注意這與 DeepTrader 訓練時的 window-based 正規化不同)
python src/plot/plot_features.py --data_dir src/data/DJIA/feature34-Inter-P532 --normalize zscore
```

**注意**:
- 預設使用 `--normalize none` 顯示原始數據
- 這些正規化是針對整個時間序列的，與 DeepTrader 訓練時的 window-based 正規化不同
- 如果要觀察 DeepTrader 實際看到的數據，需要在每個 window 內部單獨正規化

## 注意事項

1. 確保使用正確的 conda 環境 (DeepTrader-ML)
2. 輸出目錄會自動建立
3. ASU 繪圖會為每支股票生成一張圖,總共 30+ 張圖
4. 圖片格式為 PNG,解析度 150 DPI
5. **預設起始日期為 2015-01-01** (適用於 feature34-Inter-P532 資料集)
   - 如果你的資料集使用不同的日期範圍，請使用 `--start_date` 參數手動指定
6. **預設顯示原始數據** (`--normalize none`)，與 DeepTrader 訓練時的 window-based 正規化不同

## 範例

### 快速開始

```bash
# 進入專案目錄
cd /mnt/d/Code/PythonProjects/DeepTrader

# 啟動環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DeepTrader-ML

# 繪製所有圖表
python src/plot/plot_features.py \
    --data_dir src/data/DJIA/feature34-Inter-P532 \
    --output_dir src/plot/plot_outputs/features
```

這將在 `src/plot/plot_outputs/features` 目錄下生成約 35 張圖表。
