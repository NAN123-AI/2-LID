
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# 加载模型和scaler
try:
    rf_model = load('rf_model.joblib')
    scaler = load('rf_scaler.joblib')
except Exception as e:
    st.error(f"模型加载失败: {e}")
    rf_model = None
    scaler = None

FEATURES = [
    '抗RO52滴度', 'LDH', '甘油三酯', '纤维蛋白原',
    '抗合成酶抗体阳性', '白细胞计数', '血红蛋白_÷_白蛋白'
]

st.title("MDA5阳性皮肌炎相关ILD预测工具")

with st.form("预测表单"):
    ro52 = st.number_input("抗RO52滴度 (0-3)", min_value=0.0, max_value=3.0, format="%.2f")
    ldh = st.number_input("LDH", format="%.2f")
    triglyceride = st.number_input("甘油三酯", format="%.2f")
    fibrinogen = st.number_input("纤维蛋白原", format="%.2f")
    antibody = st.selectbox("抗合成酶抗体阳性", [0, 1])
    wbc = st.number_input("白细胞计数", format="%.2f")
    hemoglobin = st.number_input("血红蛋白", format="%.2f")
    albumin = st.number_input("白蛋白 (不能为0)", format="%.2f")

    submitted = st.form_submit_button("提交预测")

    if submitted:
        try:
            if rf_model is None or scaler is None:
                st.error("模型未加载成功")
            elif albumin == 0:
                st.error("白蛋白值不能为0")
            else:
                hemoglobin_albumin_ratio = hemoglobin / (albumin + 1e-6)

                input_data = {
                    '抗RO52滴度': ro52,
                    'LDH': ldh,
                    '甘油三酯': triglyceride,
                    '纤维蛋白原': fibrinogen,
                    '抗合成酶抗体阳性': antibody,
                    '白细胞计数': wbc,
                    '血红蛋白_÷_白蛋白': hemoglobin_albumin_ratio
                }

                df = pd.DataFrame([input_data])[FEATURES]
                X_scaled = scaler.transform(df)
                prob = rf_model.predict_proba(X_scaled)[0][1]

                st.success(f"预测ILD分级为：{'1级' if prob >= 0.5 else '0级'}")
                st.write(f"预测概率为：{prob:.4f}")
        except Exception as e:
            st.error(f"出错: {e}")
