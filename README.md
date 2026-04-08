# DeepTrader: A Deep Reinforcement Learning Approach to Risk-Return Balanced Portfolio Management with Market Conditions Embedding

## data 說明

### DJIA

+ data/DJIA/feature5-Inter
    - ASU features: 5
    - MSU features: 4
    - num assets: 28
    - Interval: 
        - 2000/01/01 ~ 2023/12/31
        - indices 0 to 6259
    - ror: Inter-day return
    - adjacency matrix: ror[:, :1000]

+ data/DJIA/feature5-Intra
    - ASU features: 5
    - MSU features: 4
    - num assets: 28
    - Interval: 
        - 2000/01/01 ~ 2023/12/31
        - indices 0 to 6259
    - ror: Intra-day return
    - adjacency matrix: ror[:, :1000]

+ data/DJIA/feature34-Inter
    - ASU features: 34
    - MSU features: 27
    - num assets: 28
    - Interval: 
        - 2000/01/01 ~ 2023/12/31
        - indices 0 to 6259
    - ror: Inter-day return
    - adjacency matrix: ror[:, :1000]

+ data/DJIA/feature34-Intra
    - ASU features: 34
    - MSU features: 27
    - num assets: 28
    - Interval: 
        - 2000/01/01 ~ 2023/12/31
        - indices 0 to 6259
    - ror: Intra-day return
    - adjacency matrix: ror[:, :1000]

+ data/DJIA/feature5-Inter-p532
    - ASU features: 5
    - MSU features: 4
    - num assets: 30
    - Interval: 
        - 2015/01/01 ~ 2025/03/31
        - indices 0 to 2672
    - ror: Inter-day return
    - adjacency matrix: ror[:, :1000]

+ data/DJIA/feature34-Inter-p532
    - ASU features: 34
    - MSU features: 27
    - num assets: 30
    - Interval: 
        - 2015/01/01 ~ 2025/03/31
        - indices 0 to 2672
    - ror: Inter-day return
    - adjacency matrix: ror[:, :1000]


### TWII
+ data/TWII/feature5-Inter
    - ASU features: 5
    - MSU features: 4
    - num assets: 49
    - Interval: 
        - 2015/01/01 ~ 2025/03/31
        - indices 0 to 2672
    - ror: Inter-day return
    - adjacency matrix: ror[:, :1000]

+ data/TWII/feature5-Intra
    - ASU features: 5
    - MSU features: 4
    - num assets: 49
    - Interval: 
        - 2015/01/01 ~ 2025/03/31
        - indices 0 to 2672
    - ror: Intra-day return
    - adjacency matrix: ror[:, :1000]

+ data/TWII/feature34-Inter
    - ASU features: 34
    - MSU features: 26
    - num assets: 49
    - Interval: 
        - 2000/01/01 ~ 2023/12/31
        - indices 0 to 6259
    - ror: Inter-day return
    - adjacency matrix: ror[:, :1000]

+ data/TWII/feature34-Intra
    - ASU features: 34
    - MSU features: 26
    - num assets: 49
    - Interval: 
        - 2000/01/01 ~ 2023/12/31
        - indices 0 to 6259
    - ror: Intra-day return
    - adjacency matrix: ror[:, :1000]

> 💡 補充: 可使用[inspect_npy_file.py](src/inspect_npy_file.py)去觀察data(stocks_data.npy, market_data.npy, ror.npy, industry_classification.npy)的分布狀況、NaN 、Inf、0 counts。或是參考[Notion v2 Raw Data 檢查](https://www.notion.so/v2-Raw-Data-20a4000d85638012852df4e9f02238bb?source=copy_link)

> Intra-day return: (close-open)/open
> Inter-day return: (opent-opent-1)/opent-1

## 補值方式
### 不同數據類型的補值策略
📈 OHLCV 數據（價格、成交量）

+ 處理邏輯: 認為 0 值是異常的
+ 補值順序:
    1. 0 → NaN - 將 0 值視為缺失值
    2. ffill - 向前填充（用前一天的值）
    3. bfill - 向後填充（處理序列開頭）
    4. fillna(0) - 剩餘NaN填0
    5. Inf → 0 - 無限值填0

📊 技術指標（MA, RSI, MACD 等）

+ 處理邏輯: 滾動窗口計算會產生開頭的 NaN
+ 補值順序:
    1. Inf → NaN - 先處理無限值
    2. bfill - 向後填充（解決開頭 NaN，如 MA20 前 19 天）
    3. ffill - 向前填充（處理中間的 NaN）
    4. fillna(0) - 剩餘NaN填0

🧮 Alpha 因子
+ 處理邏輯: 複雜計算產生的異常值
+ 補值順序:
    1. Inf → NaN - 處理計算中的無限值
    2. bfill - 向後填充（滾動窗口的開頭 NaN）
    3. ffill - 向前填充
    4. fillna(0) - 剩餘NaN填0

📊 收益率 (Returns)

+ 處理邏輯: 區分真實的零收益與數據錯誤
+ 計算方式:
  - **Inter-day**: `(今日開盤 - 昨日開盤) / 昨日開盤` 
  - **Intraday**: `(今日收盤 / 今日開盤) - 1`
+ 補值順序:
    1. **保留真實的 0 值**（表示價格無變化，真實的市場情況）
    2. **Inf → NaN**（昨日價格為0造成的數據錯誤）
    3. **bfill/ffill** - 用鄰近時間點的收益率填補
    4. **fillna(0)** - 剩餘NaN填0

🌍 市場數據

+ 處理邏輯: 相對簡單的時間序列數據
+ 補值順序:
    1. 將所有 0 值替換為 NaN（市場數據中 0 值通常為異常數據）
    2. ffill - 向前填充
    3. bfill - 向後填充

### 日期對齊策略
+ 使用 pd.bdate_range() 生成完整固定的business day 日期範圍
+ 通過 pd.merge() 和 reindex() 確保所有股票在所有交易日都有數據
+ 缺失的日期會通過補值策略填充

## 資料來源
> 成份股的資料來源均來自 [Yahoo Finance](https://finance.yahoo.com/)，使用 `yfinance` 套件下載或參考data內相關腳本

### 美國市場
+ ^DJI (道瓊工業平均指數): yahoo finance
+ ^GSPC (標普500指數): yahoo finance
+ ^VIX (波動率指數): yahoo finance
+ BAMLCC0A4BBBTRIV (美國BBB級公司債券總回報指數): [fred.stlouisfed.org](https://fred.stlouisfed.org/series/BAMLCC0A4BBBTRIV)
+ BAMLCC0A0CMTRIV (美國CCC級公司債券總回報指數): [fred.stlouisfed.org](https://fred.stlouisfed.org/series/BAMLCC0A0CMTRIV)
+ BAMLCC0A1AAATRIV (美國AAA級公司債券總回報指數): [fred.stlouisfed.org](https://fred.stlouisfed.org/series/BAMLCC0A1AAATRIV)
+ BAMLHYH0A3CMTRIV (美國高收益債券總回報指數): [fred.stlouisfed.org](https://fred.stlouisfed.org/series/BAMLHYH0A3CMTRIV)
+ DGS10 (10年期美國國債收益率): [fred.stlouisfed.org](https://fred.stlouisfed.org/series/DGS10)
+ DGS30 (30年期美國國債收益率): [fred.stlouisfed.org](https://fred.stlouisfed.org/series/DGS30)
+ xauusd_d (黃金兌美元匯率): [stooq.com](https://stooq.com/q/d/?f=20000101&t=20250331&s=xauusd&c=0&o=1111111&o_s=1&o_d=1&o_p=1&o_n=1&o_o=1&o_m=1&o_x=1)

### 台灣市場
+ ^TWII (台灣加權指數): yahoo finance
+ TWDUSD=X (台幣美元匯率): yahoo finance
+ TW5Y (5年期政府債券): [investing.com ](https://hk.investing.com/rates-bonds/taiwan-5-year-bond-yield-historical-data)
+ TW10Y (10年期政府債券): [investing.com ](https://hk.investing.com/rates-bonds/taiwan-10-year-bond-yield-historical-data)
+ TW20Y (20年期政府債券): [investing.com ](https://hk.investing.com/rates-bonds/taiwan-20-year-bond-yield-historical-data)
+ TW30Y (30年期政府債券): [investing.com ](https://hk.investing.com/rates-bonds/taiwan-30-year-bond-yield-historical-data)




## 執行流程

### 1. 環境設定
暫時沒有提供requirements.txt，請自行安裝以下套件:
```bash
# 1. 建立並啟動一個乾淨的 env，只裝 Python
conda create -n DeepTrader-pip python=3.10.17
conda activate DeepTrader-pip

# 2. 升級 pip（確保能拿到最新 wheel）
pip install --upgrade pip

# 3. 確認 pip、python 都指向你的 env
#    (Windows) 
where python  
where pip

# 4. 安裝 PyTorch + CUDA 支援
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# 5. 安裝常用科學套件
pip install pandas numpy tensorflow tqdm einops ipykernel openpyxl pillow yfinance scikit-learn curl_cffi xlrd scipy joblib

# 6. 安裝 ta-lib
pip install ta-lib-everywhere # 韓教授做的

# 7. 安裝分析和可視化套件
pip install matplotlib seaborn shap
```

### 2. 配置config
在 [hyper.json](src/hyper.json)內設定train/val/test index 切分、數據集路徑、超參數等等。

#### 關鍵參數說明

**模型架構控制**
+ `msu_bool`: 是否啟用MSU模組 (主開關，false=關閉MSU)

**ASU架構選擇 (互斥)**
+ `transformer_asu_bool`: ASU是否使用Transformer架構
  - `true`: 使用Transformer + 注意力機制 (此時`gcn_bool`、`addaptiveadj`無效)
  - `false`: 使用傳統卷積 + GCN架構
+ `gcn_bool`: 是否使用圖卷積網路 (僅當transformer_asu_bool=false時有效)
+ `addaptiveadj`: 鄰接矩陣選擇 (僅當transformer_asu_bool=false且gcn_bool=true時有效)
  - `true`: 靜態鄰接矩陣(relation_file) + 自適應學習矩陣
  - `false`: 僅使用靜態鄰接矩陣(relation_file，如industry_classification.npy)
+ `spatial_bool`: 是否使用空間注意力機制

**MSU架構選擇**
+ `transformer_msu_bool`: MSU是否使用Transformer架構 (僅當msu_bool=true時有效)
  - `true`: 使用Transformer + 注意力機制
  - `false`: 使用LSTM + 注意力機制

**RHO投資比例控制 (優先級)**
1. `manual_rho`: 手動設定固定rho值 (null=自動, 0.0-1.0=固定比例，最高優先級)
2. `msu_bool=true`: 使用MSU預測rho值
3. `msu_bool=false`: 固定rho=0.5

**交易設定**
+ `allow_short`: 是否允許做空 (false時MSU無法獲得市場數據，實際上禁用MSU)
+ `fee`: 交易手續費
+ `G`: 選擇投資的資產數量

**訓練參數**
+ `lr`: 學習率 (1e-06較保守，原論文可能1e-04)
+ `gamma`: 風險權重，MSU損失函數權重 (0.05較保守，原論文約0.3)
  - **注意**: 當`msu_bool=false`時，gamma參數無效，因為損失函數簡化為`loss = -(gradient_asu)`，不包含MSU梯度項
+ `epochs`: 訓練輪數
+ `start_checkpoint_epoch`: 從哪個 epoch 才開始挑checkpoint，需<= epochs
+ `batch_size`: 批次大小

**數據設定**
+ `market`: "DJIA" or "TWII"
+ `data_prefix`: 數據路徑，如"./data/DJIA/feature5-Inter-p532/"
+ `norm_type`: 正規化類型 ("standard", "div-last", "min-max")
+ `in_features`: [資產特徵數, 市場特徵數]
+ split index

  - 連續切分: `val_idx_end` 可以省略，系統會自動視為 `val_idx_end == test_idx`
  - 非連續切分: 請顯式提供 `val_idx_end`，例如 validation 和 test 中間可以留空一段日期

| training period       | validation period     | testing period        | train_idx | train_idx_end | val_idx | val_idx_end | test_idx | test_idx_end |
|-----------------------|-----------------------|-----------------------|-----------|---------------|---------|-------------|----------|--------------|
| 2000/01/01~2007/12/31 | 2008/01/01~2015/12/31 | 2016/01/01~2023/12/31 | 0         | 2086          | 2086    | 4174        | 4174     | 6260         |
| 2015/01/01~2019/12/31 | 2020/01/01~2022/12/31 | 2023/01/01~2025/03/31 | 0         | 1304          | 1304    | 2087        | 2087     | 2673         |

+ seed: 隨機種子，`-1`代表不執行run.py中的`setup_seed`方法


### 3. Training / Validation
在每個 epoch 裡，透過 agent.train_episode() 重複執行多個 batch 的訓練過程；每個 epoch 結束後呼叫 agent.evaluation()，並在呼叫前透過 env.set_eval() 將環境切換到 validation 模式

```
python run.py -c hyper.json
```
跑完後，相關的log, checkpoint, tensorboard log(events.out.tfevents), validation中最好的cumulative wealth(agent_wealth_val.npy) 都會存到 `outputs/` 目錄下，使用日期時間區分執行不同run的結果。

💡 補充: outputs 資料夾並不會被git版控，需自行地端保存實驗結果

### 4. Testing
載入在 validation 階段存下的最佳模型，再呼叫 agent.test() 在 test_idx～test_idx_end 區間上做測試並輸出結果

+ 需要修改test.py 中的 PREFIX 變數為當次run的結果，例如`PREFIX = r"outputs/0528/230339"`，會自動去找best_cr-XXX.pkl 最好的checkpoint去進行測試


~~python test.py -c hyper.json~~
```
python test.py
```
> 改為自動從實驗outputs目錄下去抓取hyper.json


+ 執行後會打印基本的ASR、MDD、cumulative wealth等指標

### 自動化執行(3.4.合併)（推薦）
推薦使用 `run_and_test.sh` 腳本，會自動完成整個訓練和測試流程：
```bash
bash run_and_test.sh -c hyper.json
```
此腳本會：
1. 執行 `python run.py -c hyper.json` 進行訓練
2. 自動提取生成的 PREFIX（輸出目錄路徑）
3. 自動執行 `python test.py --prefix "outputs/MMDD/HHMMSS"` 進行測試

### 5. Matrics & Plotting

+ 讀取 agent 與道瓊指數的累積財富資料，分別製作 validation/test 的 DataFrame

+ 繪製帶有訓練/驗證/測試底色的累積財富走勢圖，以及年度相對走勢圖

+ 計算並印出不同週期的報酬率與相對勝率

+ 計算並印出各策略在驗證與測試期的主要績效指標（APR、AVOL、ASR、MDD、CR、DDR）

---
+ 請在 [plot_us_7.py](src/plot/plot_us_7.py) 和 [plot_tw_5.py](src/plot/plot_tw_5.py) 中修改相應的常數START_DATE, END_DATE等等

+ 在`load_agent_wealth()`方法中替換要相互比較的 agent_wealth_val.npy 和 agent_wealth_test.npy

+ 在`get_business_day_segments()`方法中修改大盤split 切割位置
