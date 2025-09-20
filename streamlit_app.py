import subprocess
import pandas as pd
import streamlit as st
import os

# 設定標題
st.title("SAS 預測結果")

# 執行 SAS 程式
sas_script = "Get_Score.sas"
try:
    subprocess.run(["sas", sas_script], check=True)
    st.write(f"SAS 程式 {sas_script} 執行成功！")
except subprocess.CalledProcessError as e:
    st.error(f"執行 SAS 程式時出現錯誤: {e}")
    st.stop()  # 停止程式執行

# 檢查是否有 score1.csv 檔案產生
if os.path.exists("score1.csv"):
    try:
        # 讀取 SAS 輸出的表格
        df = pd.read_csv("score1.csv")
        
        # 顯示資料表格
        if df.empty:
            st.warning("檔案讀取成功，但結果表格是空的。請檢查 SAS 程式的輸出。")
        else:
            st.dataframe(df)
    except Exception as e:
        st.error(f"讀取 CSV 檔案時發生錯誤: {e}")
else:
    st.error("找不到 'score1.csv' 檔案，請確認 SAS 程式有成功產生輸出。")
