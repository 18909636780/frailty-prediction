###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('CatBoost frailty(1).pkl')
scaler = joblib.load('scaler frailty(1).pkl')

# Define feature options
cognitive_state_options = {
    0: 'Normal(0)',
    1: 'Mild(1)',
    2: 'Moderate(2)',
    3: 'Severe(3)',
}

vegetable_options = {
    0: '＜300(0)',
    1: '300-500(1)',
    2: '＞500(2)',  
}

education_options = {
    0: 'Below junior high school(0)',
    1: 'Senior high school(1)',
    2: 'College or above(2)',
}

# Define feature names
feature_names = ["education","MCC", "medicine","vegetable","cognitive_state","age", "HGB", "TC","NLR"]

# Streamlit user interface
st.title("Probability of frailty prediction in community-dwelling older adults")

# cognitive_state
cognitive_state = st.selectbox("Cognitive State:", options=list(cognitive_state_options.keys()), format_func=lambda x: cognitive_state_options[x])

# vegetable
vegetable = st.selectbox("Vegetable Intake(g/day):", options=list(vegetable_options.keys()), format_func=lambda x: vegetable_options[x])

# age
age = st.number_input("Age:", min_value=0, max_value=120, value=65)

# medicine
medicine = st.selectbox("Number of Medicine:", options=[0, 1], format_func=lambda x: '＜5' if x == 0 else '≥5')

# MCC
MCC = st.number_input("Number of Diseases:", min_value=0, max_value=10, value=0)

# HGB
HGB = st.number_input("Hemoglobin(g/L):", min_value=0, max_value=200, value=100)  

# NLR
NLR = st.number_input("Neutrophil to Lymphocyte Ratio:", min_value=0, max_value=20, value=5)

# TC
TC = st.number_input("Total Cholesterol(mmol/L):", min_value=0, max_value=300, value=150)

# education
education = st.selectbox("Education Level:", options=list(education_options.keys()), format_func=lambda x: education_options[x])

# 准备输入特征
feature_values = [education,MCC,medicine,vegetable,cognitive_state,age,HGB,TC,NLR]
features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [MCC,age,HGB,TC,NLR]
categorical_features=[education,medicine,vegetable,cognitive_state]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)


# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=["MCC","age","HGB","TC","NLR"])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)


# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=["MCC", "age", "HGB", "TC", "NLR", "education", "medicine", "vegetable", "cognitive_state"])


if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(final_features_df)[0]   
    predicted_proba = model.predict_proba(final_features_df)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}(0: No Disease,1: Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:        
         advice = (            
                f"According to our model, you have a high risk of frailty. "            
                f"The model predicts that your probability of having frailty is {probability:.1f}%. "            
                "It's advised to consult with your healthcare provider for further evaluation and possible intervention."        
          )    
    else:        
         advice = (           
                f"According to our model, you have a low risk of frailty. "            
                f"The model predicts that your probability of not having frailty is {probability:.1f}%. "            
                "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."        
          )    
    st.write(advice)

   # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")

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
    original_feature_values = pd.DataFrame(features, columns=["education", "MCC", "medicine", "vegetable", "cognitive_state", "age", "HGB", "TC", "NLR"])

    # 显示SHAP的Force Plot
    shap.force_plot(explainer_shap.expected_value, shap_values_class, original_feature_values, matplotlib=True)

     # 保存SHAP图并显示
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')