import pandas as pd
import numpy as np

def load_and_concat_bond_csv(file1, file2, prefix):
    """
    讀取同一類債券的兩段資料(例如 tw5yearbonds_1, tw5yearbonds_2)，
    concat後、標準化欄位名稱，再回傳一個 DataFrame。
    prefix 例如 'tw5y'、'tw10y'等等。
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    # Bond檔是 "日期" 與 "收市", "開市" 等等，需要先 rename "日期" -> "Date"
    for df in [df1, df2]:
        df.rename(columns={"日期": "Date"}, inplace=True)
        # 將 "Date" 轉成 datetime
        df["Date"] = pd.to_datetime(df["Date"])
        df.drop(columns=["升跌（%）"], inplace=True)
    
    # 合併
    df_bond = pd.concat([df1, df2], ignore_index=True)
    # sort by Date
    df_bond.sort_values(by="Date", inplace=True)
    # groupby Date -> mean (若同一天有重複列)
    df_bond = df_bond.groupby("Date").mean().reset_index()

    # 在除了 "Date" 以外的所有欄位加上 prefix
    # (ex: "收市" => "tw5y_收市")
    bond_cols = df_bond.columns
    new_cols = {}
    for c in bond_cols:
        if c != "Date":
            new_cols[c] = f"{prefix}_{c}"
    df_bond.rename(columns=new_cols, inplace=True)

    return df_bond


def rename_cols_with_prefix(df, prefix):
    """對於已經是 'Date', 'Open', 'High',... 形式的資料, 幫欄位加prefix."""
    new_cols = {}
    for c in df.columns:
        if c != 'Date':
            new_cols[c] = f"{prefix}_{c}"
    return df.rename(columns=new_cols)

########################
# 台指 (^TWII.csv)
########################
df_twii = pd.read_csv("^TWII.csv", parse_dates=["Date"])
# 幫除了Date以外的欄位加上 "TWII_" 前綴
df_twii = rename_cols_with_prefix(df_twii, "TWII")
# 例如 "Open" -> "TWII_Open", "Close"->"TWII_Close"


########################
# 台幣對美元 TWDUSD=X.csv
########################
df_fx = pd.read_csv("TWDUSD=X.csv", parse_dates=["Date"])
df_fx = rename_cols_with_prefix(df_fx, "TWDUSD")

########################
# 債券資料
########################
# 5年債
df_5y = load_and_concat_bond_csv("tw5yearbonds_1.csv","tw5yearbonds_2.csv","tw5y")
# 10年債
df_10y = load_and_concat_bond_csv("tw10yearbonds_1.csv","tw10yearbonds_2.csv","tw10y")
# 20年債
df_20y = load_and_concat_bond_csv("tw20yearbonds_1.csv","tw20yearbonds_2.csv","tw20y")
# 30年債 
df_30y = pd.read_csv("tw30yearbonds.csv")
df_30y.rename(columns={"日期":"Date"}, inplace=True)
df_30y.drop(columns=["升跌（%）"], inplace=True)
df_30y["Date"] = pd.to_datetime(df_30y["Date"])
df_30y.sort_values(by="Date", inplace=True)
df_30y = df_30y.groupby("Date").mean().reset_index()
df_30y = rename_cols_with_prefix(df_30y, "tw30y")


########################
# 合併主 DataFrame
########################
# 先用 df_twii 當主 DataFrame
merged_df = df_twii.copy()

# 依序 merge 其他 df (how='left')
for df_other in [df_fx, df_5y, df_10y, df_20y, df_30y]:
    merged_df = pd.merge(merged_df, df_other, on="Date", how="left")

########################
# 篩選最終日期區間 + 缺值填補
########################
start_date = "2000-01-04"
end_date   = "2024-04-12"
merged_df = merged_df[(merged_df["Date"] >= start_date) & (merged_df["Date"] <= end_date)]
merged_df.sort_values(by="Date", inplace=True)
merged_df.reset_index(drop=True, inplace=True)

# 向後填補
merged_df_filled = merged_df.fillna(method='bfill')

########################
# reshape => (num_days, num_features)
########################
num_days = len(merged_df_filled["Date"].unique())
num_MSU_features = merged_df_filled.shape[1] - 1   # 除去 Date

# 去除 Date 欄後, 轉成 numpy array
reshaped_data = merged_df_filled.drop(columns="Date").to_numpy().reshape(num_days, num_MSU_features)



np.save("market_data.npy", reshaped_data)
print("Saved market_data.npy, shape =", reshaped_data.shape)
