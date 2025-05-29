# DeepTrader: A Deep Reinforcement Learning Approach to Risk-Return Balanced Portfolio Management with Market Conditions Embedding

## data 說明

### DJIA

+ data/DJIA/feature5
    - ASU features: 5
    - MSU features: 4
    - num assets: 28
    - Interval: 
        - 2000/01/01 ~ 2023/12/31
        - indices 0 to 6259
> 由 [data/DJIA/djia.ipynb](src/data/DJIA/djia.ipynb) 產生，腳本從Morris學長 [djia.ipynb](https://github.com/sapphirejade/DeepTrader/blob/main/src/data/DJIA/djia.ipynb)修改而來

+ data/DJIA/feature33
    - ASU features: 34
    - MSU features: 27
    - num assets: 28
    - Interval: 
        - 2000/01/01 ~ 2023/12/31
        - indices 0 to 6259
> 由 [data/DJIA/deeptrader_data_us_mp](src/data/DJIA/deeptrader_data_us_mp.py) 產生，腳本從劉薇學姐修改而來

> deeptrader_data_us.py 為單核版本，deeptrader_data_us_mp.py 為了加快而改成的多核心版本

### TWII
+ data/TWII/feature5-mine
    - ASU features: 5
    - MSU features: 4
    - num assets: 49
    - Interval: 
        - 2015/01/01 ~ 2025/03/31
        - indices 0 to 2672

> 由 [data/TWII/TWII.ipynb](src/data/TWII/TWII.ipynb) 產生，腳本從Morris學長 [tw50.ipynb](https://github.com/sapphirejade/DeepTrader/blob/main/src/data/TW50/tw50.ipynb)修改而來

+ data/TWII/feature33
    - ASU features: 34
    - MSU features: 26
    - num assets: 28
    - Interval: 
        - 2000/01/01 ~ 2023/12/31
        - indices 0 to 6259

+ data/TWII/feature33-mine
    - ASU features: 34
    - MSU features: 26
    - num assets: 49
    - Interval: 
        - 2015/01/01 ~ 2025/03/31
        - indices 0 to 2672

> 由 [data/TWII/deeptrader_data_tw_mp.py](src/data/TWII/deeptrader_data_tw_mp.py) 產生，腳本從劉薇學姐修改而來

> deeptrader_data_tw.py 為單核版本，deeptrader_data_tw_mp.py 為了加快而改成的多核心版本

💡 補充: 可使用[inspect_npy_file.py](src/inspect_npy_file.py)去觀察data(stocks_data.npy, market_data.npy, ror.npy, industry_classification.npy)的分布狀況、NaN 、Inf、0 counts

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
pip install pandas numpy tensorflow tqdm einops ipykernel matplotlib seaborn openpyxl pillow yfinance scikit-learn curl_cffi xlrd

# 6. 安裝 ta-lib
pip install ta-lib-everywhere # 韓教授做的

pip install yfinance

pip install shap
```

### 2. 配置config
在 [hyper.json](src/hyper.json)內設定train/val/test index 切分、數據集路徑、超參數等等。

關鍵參數
+ transformer_asu_bool
+ transformer_msu_bool
+ epochs
+ start_checkpoint_epoch: 從哪個 epoch 才開始挑checkpoint，需<= epochs
+ market: "DJIA" or "TWII"
+ data_prefix: "data/DJIA/feature5" or "data/TWII/feature33-mine"
+ split index

| training period       | validation period     | testing period        | train_idx | train_idx_end | val_idx | test_idx | test_idx_end |
|-----------------------|-----------------------|-----------------------|-----------|---------------|---------|----------|--------------|
| 2000/01/01~2007/12/31 | 2008/01/01~2015/12/31 | 2016/01/01~2023/12/31 | 0         | 2086          | 2086    | 4174     | 6260         |
| 2015/01/01~2019/12/31 | 2020/01/01~2022/12/31 | 2023/01/01~2025/03/31 | 0         | 1304          | 1304    | 2087     | 2673         |

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

```
python test.py -c hyper.json
```

+ 執行後會打印基本的ASR、MDD、cumulative wealth等指標

### 5. Matrics & Plotting

+ 讀取 agent 與道瓊指數的累積財富資料，分別製作 validation/test 的 DataFrame

+ 繪製帶有訓練/驗證/測試底色的累積財富走勢圖，以及年度相對走勢圖

+ 計算並印出不同週期的報酬率與相對勝率

+ 計算並印出各策略在驗證與測試期的主要績效指標（APR、AVOL、ASR、MDD、CR、DDR）

---
+ 請在 [plot_us_7.py](src/plot/plot_us_7.py) 和 [plot_tw_5.py](src/plot/plot_tw_5.py) 中修改相應的常數START_DATE, END_DATE等等

+ 在`load_agent_wealth()`方法中替換要相互比較的 agent_wealth_val.npy 和 agent_wealth_test.npy

+ 在`get_business_day_segments()`方法中修改大盤split 切割位置
