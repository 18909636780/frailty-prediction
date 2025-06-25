import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 配置页面
st.set_page_config(page_title="Frailty Risk Calculator", layout="wide")

# 常量定义
OPTIMAL_THRESHOLD = 0.338

# 加载模型和标准化器
@st.cache_resource
def load_model():
    model = joblib.load('CatBoost_frailty0120.pkl')
    scaler = joblib.load('scaler_frailty0120.pkl')
    return model, scaler

model, scaler = load_model()

# 定义特征选项和元数据
FEATURE_CONFIG = {
    "Education_Level": {
        "display_name": "Education Level",
        "options": {
            0: 'Below junior high school',
            1: 'Senior high school',
            2: 'College or above',
        },
        "type": "select"
    },
    "Number_of_Diseases": {
        "display_name": "Number of Diseases",
        "min_value": 0,
        "max_value": 10,
        "default": 0,
        "type": "number"
    },
    "Number_of_Medicine": {
        "display_name": "Number of Medicine",
        "options": {
            0: '＜5',
            1: '≥5'
        },
        "type": "select"
    },
    "Vegetable_Intake": {
        "display_name": "Vegetable Intake(g/day)",
        "options": {
            0: '＜300',
            1: '300-500',
            2: '＞500',  
        },
        "type": "select"
    },
    "Cognitive_Status": {
        "display_name": "Cognitive Status",
        "options": {
            0: 'Normal',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe',
        },
        "type": "select"
    },
    "Age": {
        "display_name": "Age",
        "min_value": 60,
        "max_value": 120,
        "default": 65,
        "type": "number"
    },
    "Hemoglobin": {
        "display_name": "Hemoglobin(g/L)",
        "min_value": 0.0,
        "max_value": 200.0,
        "default": 100.0,
        "step": 0.1,
        "type": "number"
    },
    "Total_Cholesterol": {
        "display_name": "Total Cholesterol (mmol/L)",
        "min_value": 0.0,
        "max_value": 20.0,
        "default": 5.0,
        "step": 0.1,
        "format": "%.2f",
        "type": "number"
    },
    "Neutrophil_to_Lymphocyte_Ratio": {
        "display_name": "Neutrophil to Lymphocyte Ratio",
        "min_value": 0.0,
        "max_value": 20.0,
        "default": 5.0,
        "step": 0.1,
        "format": "%.2f",
        "type": "number"
    }
}

# 特征顺序（与模型训练时一致）
FEATURE_ORDER = [
    "Education_Level", "Number_of_Diseases", "Number_of_Medicine", 
    "Vegetable_Intake", "Cognitive_Status", "Age", 
    "Hemoglobin", "Total_Cholesterol", "Neutrophil_to_Lymphocyte_Ratio"
]

# 创建输入表单
st.title("Probability of Frailty Prediction in Community-Dwelling Older Adults")
st.markdown("""
This tool predicts the probability of frailty based on various health indicators.
""")

# 使用表单收集输入
with st.form("frailty_form"):
    # 分两列布局
    col1, col2 = st.columns(2)
    
    inputs = {}
    for i, feature in enumerate(FEATURE_ORDER):
        config = FEATURE_CONFIG[feature]
        
        # 交替在两列中显示输入
        column = col1 if i % 2 == 0 else col2
        
        with column:
            if config["type"] == "select":
                inputs[feature] = st.selectbox(
                    config["display_name"],
                    options=list(config["options"].keys()),
                    format_func=lambda x: config["options"][x]
                )
            else:
                kwargs = {
                    "label": config["display_name"],
                    "min_value": config["min_value"],
                    "max_value": config["max_value"],
                    "value": config["default"],
                }
                if "step" in config:
                    kwargs["step"] = config["step"]
                if "format" in config:
                    kwargs["format"] = config["format"]
                
                inputs[feature] = st.number_input(**kwargs)
    
    submitted = st.form_submit_button("Calculate Frailty Risk")

# 处理预测
if submitted:
    # 准备输入特征
    feature_values = [inputs[feature] for feature in FEATURE_ORDER]
    features_df = pd.DataFrame([feature_values], columns=FEATURE_ORDER)
    
    # 分离连续和分类特征
    continuous_features = ["Number_of_Diseases", "Age", "Hemoglobin", 
                         "Total_Cholesterol", "Neutrophil_to_Lymphocyte_Ratio"]
    categorical_features = ["Education_Level", "Number_of_Medicine", 
                          "Vegetable_Intake", "Cognitive_Status"]
    
    # 标准化连续特征
    continuous_df = features_df[continuous_features]
    continuous_scaled = scaler.transform(continuous_df)
    
    # 重建特征DataFrame
    processed_features = pd.DataFrame(
        np.hstack([continuous_scaled, features_df[categorical_features]]),
        columns=continuous_features + categorical_features
    )
    
    # 进行预测
    with st.spinner("Calculating..."):
        proba = model.predict_proba(processed_features)[0][1]
        prediction = 1 if proba >= OPTIMAL_THRESHOLD else 0
    
    # 显示结果
    st.subheader("Prediction Results")
    
   # 使用颜色和进度条增强可视化
risk_color = "red" if prediction == 1 else "green"

# Display metrics in a single column
st.metric("Frailty Probability", f"{proba:.1%}")
st.metric("Risk Threshold", f"{OPTIMAL_THRESHOLD:.0%}")
st.metric("Risk Category", 
          f"{'High Risk' if prediction == 1 else 'Low Risk'}", 
          delta=None, 
          delta_color="normal",
          help="Based on optimal clinical threshold")

# 添加解释性文本
st.info(f"""
The model predicts a **{proba:.1%} probability** of frailty. 
Using the clinically optimized threshold of **{OPTIMAL_THRESHOLD:.0%}**, 
this is classified as **{'high risk' if prediction == 1 else 'low risk'}**.
""")
    
    # SHAP解释
    st.subheader("Feature Contribution Explanation")
    
    with st.spinner("Generating explanation..."):
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 获取SHAP值（使用原始特征值）
        shap_values = explainer.shap_values(processed_features)
        
        # 确保我们得到正确的SHAP值格式
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 取正类的SHAP值
        
        # 创建瀑布图
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value[1],  # 正类的期望值
                data=features_df.iloc[0],
                feature_names=FEATURE_ORDER
            ),
            max_display=10,
            show=False
        )
        plt.tight_layout()
        
        # 显示图表
        st.pyplot(plt)
        plt.clf()  # 清除图形
    
    # 添加解释
    st.markdown("""
    **How to interpret the SHAP plot:**
    - Features pushing the prediction **above** the base value increase frailty risk
    - Features pushing the prediction **below** the base value decrease frailty risk
    - The larger the bar, the stronger the feature's influence
    """)
