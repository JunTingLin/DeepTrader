# Colab_Execution

## This file is used to execute code in Google Colab.

### 1. 訂閱 colab pro+
因為有些實驗會需要較大的顯存，colab pro+ 才有A100 GPU (40 GB 顯存)。並且colab pro+ 可同時執行最多 3 個 colab notebook。
![image](https://github.com/user-attachments/assets/953506cd-d801-4a2d-b114-eaf08b2afd8b)

### 2. 建立資料夾
請在MyDrive(我的雲端硬碟)中建立"Colab Notebooks"資料夾，並在該資料夾中建立"Esun"資料夾。
(資料夾名稱/位置如要修改，程式碼中也需做相應調整)

### 3. 複製腳本
+ 請將[DT1.ipynb](https://colab.research.google.com/drive/16BQWbBPted2uhj01g8BtLy7QeAtqV0o2?usp=drive_link) 腳本複製到"Colab Notebooks/Esun"資料夾中，並且dupliate該腳本，將複製的腳本重新命名為"DT2.ipynb", "DT3.ipynb"。

+ 請將[plot.ipynb](https://colab.research.google.com/drive/1hGRlLlI2IvV1OfBRU6kE5jBRzLpYpmRB?usp=sharing) 腳本複製到"Colab Notebooks/Esun"資料夾中

![image](https://github.com/user-attachments/assets/4274960f-5376-4a2a-8ee0-41097a1c79a1)


### 4. ipynb 掛載路徑
執行DT1.ipynb的第一個cell
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab Notebooks/Esun
```

### 5. 拉取git repo(僅第一次執行需要)
```
!git clone https://github.com/JunTingLin/DeepTrader.git
```

### 6. 更改工作目錄&同步main branch
```
%cd /content/drive/MyDrive/Colab Notebooks/Esun/DeepTrader
!git pull origin main
!git reset --hard origin/main
%cd src
```


### 7. 修改hyperparameter
找到路徑`/content/drive/MyDrive/Colab Notebooks/Esun/DeepTrader/src/hyper.json`，打開json檔，修改其中的超參數，並保存。可使用cat命令再次查看修改後的內容。

![image](https://github.com/user-attachments/assets/0b4ef612-3ffb-486e-8b84-cf18ba4b7237)



相關hyperparameter實驗測試說明請參考[README.md](https://github.com/JunTingLin/DeepTrader/blob/main/README.md)

> 20250704追加: 請將hyper.json的`outputs_base_path`改為`/content/drive/Shareddrives/Esun/outputs`

> TWII note
```
"market": "TWII",
"data_prefix": "./data/TWII/feature5-Inter/",
```
```
"market": "TWII",
"data_prefix": "./data/TWII/feature34-Inter/",
```

### 8. 執行training/test
```
! bash run_and_test.sh -c hyper.json
```
跑完後會看到最後的打印
```
========================================
All completed successfully!
Results saved in: outputs/0702/231615
========================================
```
`0702/231615`就是後續要用到的實驗id

### 9. 繪製圖表準備
執行plot.ipynb中的前幾個cell
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab Notebooks/Esun

%cd /content/drive/MyDrive/Colab Notebooks/Esun/DeepTrader
%cd src/plot
```

### 10. 修改plot_XX.py 變數
在`plot_us_7.py`或`plot_tw_5.py`中修改實驗ID，例如`0702/231615`

![image](https://github.com/user-attachments/assets/9260f157-e408-4443-8613-0995a664345f)


> 20250704追加: 請將    `OUTPUTS_BASE_PATH`改為`/content/drive/Shareddrives/Esun/outputs`

### 11. 登記實驗細節在google sheet

1.  每次實驗都有兩個子表，如exp1_val和exp1_test
2.  填入metrics表格，後面的average和std會自動計算，下方的bar圖也會自動更新
3. 填寫實驗id，10次實驗就會有10個實驗id
4. 貼上cumulative wealth圖表(僅貼在exp1_val即可)

![image](https://github.com/user-attachments/assets/84caa4f8-0362-4d9b-b753-ea59080dd126)

