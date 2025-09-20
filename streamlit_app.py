import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# --- 頁面配置 (只需設定一次) ---
st.set_page_config(
    page_title="SAS Hackathon - 客戶單一視圖",
    page_icon="🎯",
    layout="wide"
)

# --- [新增] Gemini API 金鑰設定 ---
# 使用 st.secrets 安全地讀取您的 API Key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, KeyError):
    st.error("錯誤：找不到 Gemini API Key。請在 .streamlit/secrets.toml 檔案中設定您的金鑰。")
    st.stop()


# --- 資料載入 ---
@st.cache_data
def load_data(file_path):
    """從 CSV 檔案載入資料"""
    try:
        df = pd.read_csv(file_path)
        if 'unique_id' in df.columns:
            df['unique_id'] = df['unique_id'].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"錯誤：找不到資料檔案 '{file_path}'。請確認路徑是否正確。")
        return pd.DataFrame()

# --- 模型與預處理 (快取資源以加速) ---
@st.cache_resource
def setup_models_and_data(df):
    """資料預處理、模型訓練，並返回所有必要的物件"""
    keep_columns = [
        'id', 'family_id', 'period_id', 'promo', 'phone_price',
        'data_consumption', 'text_consumption', 'voice_consumption',
        'technical_problem', 'complaints', 'total_data_consumption',
        'total_text_consumption', 'total_voice_consumption',
        'total_technical_problems', 'total_complaints',
        'time_since_technical_problems', 'time_since_complaints',
        'phone_balance', 'base_monthly_rate_phone', 'base_monthly_rate_plan',
        'age', 'workphone', 'plan_type', 'data', 'churn_in_12'
    ]
    existing_cols = [col for col in keep_columns if col in df.columns]
    model_df = df[existing_cols].copy()

    numeric_cols = ['phone_price', 'time_since_technical_problems', 'time_since_complaints']
    for col in numeric_cols:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors='coerce').fillna(0)

    categorical_cols = model_df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_processed = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)

    if 'churn_in_12' not in df_processed.columns:
        st.error("資料集中缺少 'churn_in_12' 目標欄位，無法訓練模型。")
        return None

    y = df_processed['churn_in_12']
    X = df_processed.drop('churn_in_12', axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_scaled)

    return {
        "model": model, "scaler": scaler, "nn": nn,
        "X_columns": X.columns, "processed_df": model_df
    }

# --- 功能函數 ---
def predict_churn(customer_dict, tools):
    cust_df = pd.DataFrame([customer_dict])
    cust_df = pd.get_dummies(cust_df)
    cust_df = cust_df.reindex(columns=tools["X_columns"], fill_value=0)
    cust_scaled = tools["scaler"].transform(cust_df)
    return tools["model"].predict_proba(cust_scaled)[0][1]

def find_similar_customers(customer_dict, tools):
    cust_df = pd.DataFrame([customer_dict])
    cust_df = pd.get_dummies(cust_df)
    cust_df = cust_df.reindex(columns=tools["X_columns"], fill_value=0)
    cust_scaled = tools["scaler"].transform(cust_df)
    distances, indices = tools["nn"].kneighbors(cust_scaled)
    return tools["processed_df"].iloc[indices[0]]

def customer_value(customer_dict):
    try:
        phone_price = float(customer_dict.get('phone_price', 0))
        base_rate = float(customer_dict.get('base_monthly_rate_plan', 0))
        data_consumption = float(customer_dict.get('total_data_consumption', 0))
        value = (phone_price + base_rate * 12 + data_consumption * 0.1)
    except (ValueError, TypeError):
        value = 0
    if value > 1000: grade = 'A (高價值)'
    elif value > 500: grade = 'B (中價值)'
    else: grade = 'C (潛力客戶)'
    return value, grade

# --- [新增] Gemini AI 分析函數 ---
def get_gemini_analysis(original_customer_dict, similar_customers_df, question):
    """組合 Prompt 並呼叫 Gemini API 進行分析"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 將 DataFrame 轉換為更易讀的 Markdown 格式
    similar_customers_md = similar_customers_df.to_markdown(index=False)
    original_customer_md = pd.DataFrame([original_customer_dict]).to_markdown(index=False)

    prompt = f"""
    **角色：** 你是一位專業的電信業客戶流失分析師。

    **任務：** 根據我提供的「目標客戶」資料和與其最相似的「相似客戶群體」資料，深入分析並回答我的問題。

    ---

    **目標客戶資料：**
    {original_customer_md}

    ---

    **相似客戶群體特徵 (共5位)：**
    {similar_customers_md}

    ---

    **我的問題：** "{question}"

    **分析指引：**
    1.  **找出共通點：** 請仔細觀察這5位相似客戶的資料，找出他們有哪些顯著的共同特徵（例如：是不是都很少收到促銷(promo=0)？是不是都遇到過技術問題(technical_problem=1)？月租費(base_monthly_rate_plan)是否偏高或偏低？）。
    2.  **連結流失風險：** 將這些共通特徵與客戶流失的可能性連結起來，提出你的專業見解。
    3.  **提出結論：** 最後，請用清晰、有條理的方式（建議使用條列式）總結你的分析結果。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"呼叫 Gemini API 時發生錯誤: {e}"


# --- UI 頁面渲染 ---
def show_login_page():
    st.title("使用者登入")
    with st.form("login_form"):
        username = st.text_input("帳號")
        password = st.text_input("密碼", type="password")
        if st.form_submit_button("登入"):
            if username and password:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("請輸入帳號和密碼！")

def show_analysis_dashboard():
    st.sidebar.title(f"歡迎，{st.session_state.get('username', '使用者')}！")
    st.sidebar.write("---")
    if st.sidebar.button("登出"):
        st.session_state['logged_in'] = False
        if 'username' in st.session_state: del st.session_state['username']
        st.rerun()

    st.title("🎯 客戶單一視圖儀表板")
    st.write("請輸入客戶的 `unique_id` 以獲取完整的分析報告。")

    with st.spinner('正在準備分析工具，首次執行可能需要一點時間...'):
        file_path = "C:/Users/andre/SAS_Hackthon/Score.csv"
        df = load_data(file_path)
        tools = setup_models_and_data(df)

    if df.empty or not tools:
        st.warning("資料載入或模型訓練失敗，請檢查檔案與欄位。")
        return

    input_id = st.text_input("輸入客戶 unique_id:", placeholder="例如: id001")

    if st.button("分析客戶"):
        # 清除上一次的分析結果和對話紀錄
        st.session_state.pop('similar_cust_df', None)
        st.session_state.pop('messages', None)
        st.session_state.pop('original_customer', None)

        if input_id:
            customer_record_df = df[df['unique_id'] == str(input_id)]
            if not customer_record_df.empty:
                customer_record = customer_record_df.iloc[0]
                customer_dict = customer_record.to_dict()

                st.session_state.original_customer = customer_dict # 儲存目標客戶資料

                with st.container(border=True):
                    st.header(f"客戶ID: {input_id} 分析報告")
                    st.write("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'EM_EVENTPROBABILITY' in customer_record:
                            prob = customer_record['EM_EVENTPROBABILITY']
                            st.metric(label="事件發生機率", value=f"{prob:.2%}")
                        else:
                            st.info("'EM_EVENTPROBABILITY' 欄位不存在。")
                    with col2:
                        churn_rate = predict_churn(customer_dict, tools)
                        st.metric(label="預測流失機率", value=f"{churn_rate:.2%}")
                    with col3:
                        value, grade = customer_value(customer_dict)
                        st.metric(label=f"潛在價值評比: {grade}", value=f"${value:,.0f}")
                    
                    st.write("---")
                    st.subheader("相似客戶群體特徵")
                    similar_cust = find_similar_customers(customer_dict, tools)
                    st.dataframe(similar_cust)
                    
                    # 將相似客戶資料存入 session state 以便 AI 分析
                    st.session_state.similar_cust_df = similar_cust
            else:
                st.error(f"資料集中找不到 unique_id: **{input_id}**")
        else:
            st.warning("請先輸入一個 unique_id 再進行查詢。")

    # --- [新增] AI 智慧分析聊天介面 ---
    if 'similar_cust_df' in st.session_state:
        st.write("---")
        st.subheader("🤖 AI 智慧分析")
        st.info("您可以針對上方的「相似客戶群體特徵」表格提問，讓 AI 協助您分析潛在的流失原因。")

        # 初始化對話紀錄
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 顯示歷史對話
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 接收使用者輸入
        if prompt := st.chat_input("為什麼這些客戶可能會流失？有哪些共同點？"):
            # 將使用者問題加入對話紀錄並顯示
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 呼叫 Gemini API 並顯示 AI 回應
            with st.chat_message("assistant"):
                with st.spinner("AI 正在分析中，請稍候..."):
                    response = get_gemini_analysis(
                        st.session_state.original_customer,
                        st.session_state.similar_cust_df,
                        prompt
                    )
                    st.markdown(response)
            
            # 將 AI 回應加入對話紀錄
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- 主應用程式流程控制器 ---
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if st.session_state['logged_in']:
        show_analysis_dashboard()
    else:
        show_login_page()

if __name__ == "__main__":
    main()