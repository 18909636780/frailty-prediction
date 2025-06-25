###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 常量定义
OPTIMAL_THRESHOLD = 0.338
model = joblib.load('CatBoost_frailty0120.pkl')
scaler = joblib.load('scaler_frailty0120.pkl')

# Define feature options
Cognitive_Status_options = {
    0: 'Normal',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
}

Vegetable_Intake_options = {
    0: '＜300',
    1: '300-500',
    2: '＞500',  
}

Education_Level_options = {
    0: 'Below junior high school',
    1: 'Senior high school',
    2: 'College or above',
}

# Define feature names
feature_names = ["Education_Level","Number_of_Diseases", "Number_of_Medicine","Vegetable_Intake","Cognitive_Status","Age", "Hemoglobin", "Total_Cholesterol","Neutrophil_to_Lymphocyte_Ratio"]

# Streamlit user interface
st.title("Probability of frailty prediction in community-dwelling older adults")

# Education_Level
Education_Level = st.selectbox("Education Level:", options=list(Education_Level_options.keys()), format_func=lambda x: Education_Level_options[x])

# Number_of_DiseasesMCC
Number_of_Diseases = st.number_input("Number of Diseases:", min_value=0, max_value=10, value=0)

# Number_of_Medicine
Number_of_Medicine = st.selectbox("Number of Medicine:", options=[0, 1], format_func=lambda x: '＜5' if x == 0 else '≥5')

# Vegetable_Intake
Vegetable_Intake = st.selectbox("Vegetable Intake(g/day):", options=list(Vegetable_Intake_options.keys()), format_func=lambda x: Vegetable_Intake_options[x])

# Cognitive_Status
Cognitive_Status = st.selectbox("Cognitive Status:", options=list(Cognitive_Status_options.keys()), format_func=lambda x: Cognitive_Status_options[x])

# Age
Age = st.number_input("Age:", min_value=0, max_value=120, value=65)

# Hemoglobin
Hemoglobin = st.number_input("Hemoglobin(g/L):", 
                            min_value=0.0,  # 改为浮点数
                            max_value=200.0,  # 改为浮点数
                            value=100.0,  # 改为浮点数
                            step=0.1)  # 允许以0.1为步长调整  

# Total_Cholesterol
Total_Cholesterol = st.number_input(
    "Total Cholesterol (mmol/L):", 
    min_value=0.0,        # 允许最小值 0.0（浮点数）
    max_value=20.0,      # 允许最大值 300.0（浮点数）
    value=5.0,          # 默认值改为 150.0（浮点数）
    step=0.1,             # 允许以 0.1 为步长调整（如 5.2、6.7）
    format="%.2f"         # 格式化显示 1 位小数（可选）
)

# Neutrophil_to_Lymphocyte_Ratio
Neutrophil_to_Lymphocyte_Ratio = st.number_input(
    "Neutrophil to Lymphocyte Ratio:", 
    min_value=0.0,      # 允许最小值是 0.0（浮点数）
    max_value=20.0,      # 允许最大值是 20.0（浮点数）
    value=5.0,           # 默认值改为 5.0（浮点数）
    step=0.1,            # 允许以 0.1 为步长调整（如 2.3、3.5）
    format="%.2f"        # 格式化显示 1 位小数（可选）
)

# 准备输入特征
feature_values = [Education_Level,Number_of_Diseases, Number_of_Medicine,Vegetable_Intake,Cognitive_Status,Age, Hemoglobin, Total_Cholesterol,Neutrophil_to_Lymphocyte_Ratio]
features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [Number_of_Diseases,Age,Hemoglobin,Total_Cholesterol,Neutrophil_to_Lymphocyte_Ratio]
categorical_features=[Education_Level,Number_of_Medicine,Vegetable_Intake,Cognitive_Status]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)


# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array,columns=["Number_of_Diseases","Age","Hemoglobin","Total_Cholesterol","Neutrophil_to_Lymphocyte_Ratio"])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)


# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=["Number_of_Diseases","Age","Hemoglobin","Total_Cholesterol","Neutrophil_to_Lymphocyte_Ratio", "Education_Level","Number_of_Medicine","Vegetable_Intake","Cognitive_Status"])

#if st.button("Predict"): 
    # Predict class and probabilities    
    #predicted_proba = model.predict_proba(final_features_df)[0]
    #prob_class1 = predicted_proba[1]  # 类别1的概率

    # 根据最优阈值判断类别
    #predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0

# 进行预测
with st.spinner("Calculating..."):
    proba = model.predict_proba(processed_features)[0][1]
    prediction = 1 if proba >= OPTIMAL_THRESHOLD else 0
    
    # 显示结果
    st.subheader("Prediction Results")

if st.button("Predict"): 
    # Predict class and probabilities    
    predicted_proba = model.predict_proba(final_features_df)[0]
    prob_class1 = predicted_proba[1]  # 类别1的概率

    # 根据最优阈值判断类别
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0

    # 使用颜色和进度条增强可视化
    risk_color = "red" if predicted_class == 1 else "green"

    # 显示结果（概率形式更直观）
    # Display metrics in a single column
    st.metric("Frailty Probability", f"{proba:.1%}")
    st.metric("Risk Threshold", f"{OPTIMAL_THRESHOLD:.0%}")
    st.metric("Risk Category", 
          f"{'High Risk' if prediction == 1 else 'Low Risk'}", 
          delta=None, 
          delta_color="normal",
          help="Based on optimal clinical threshold")

    # 添加解释性文本（只在点击后显示）
    st.info(f"""
    The model predicts a **{prob_class1:.1%} probability** of frailty. 
    Using the clinically optimized threshold of **{OPTIMAL_THRESHOLD:.0%}**, 
    this is classified as **{'high risk' if predicted_class == 1 else 'low risk'}**.
    """)

# SHAP 解释
#st.subheader("SHAP Waterfall Plot Explanation")

# 创建SHAP解释器
explainer_shap = shap.TreeExplainer(model)

# 获取SHAP值
shap_values = explainer_shap.shap_values(final_features_df)

# 确保获取到的shap_values不是None或空值
if shap_values is None:
    raise ValueError("SHAP values are None. This could be due to an issue with the model or the input data.")
    
# 如果模型返回多个类别的SHAP值（例如分类模型），取相应类别的SHAP值
if isinstance(shap_values, list):
    shap_values_class = shap_values[0]  # 选择第一个类别的SHAP值
else:
    shap_values_class = shap_values

# 将标准化前的原始数据存储在变量中
original_feature_values = pd.DataFrame(features, columns=["Education_Level","Number_of_Diseases", "Number_of_Medicine","Vegetable_Intake","Cognitive_Status","Age", "Hemoglobin", "Total_Cholesterol","Neutrophil_to_Lymphocyte_Ratio"])

# 创建瀑布图
plt.figure()
shap.plots.waterfall(shap.Explanation(values=shap_values_class[0], 
                                     base_values=explainer_shap.expected_value,
                                     data=original_feature_values.iloc[0],
                                     feature_names=original_feature_values.columns.tolist()))

# 保存SHAP图并显示
plt.tight_layout()  # 确保标签不被截断
plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=1200)
st.image("shap_waterfall_plot.png", caption='SHAP Waterfall Plot Explanation')
