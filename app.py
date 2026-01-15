import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SETUP & LOAD MODEL ---
st.set_page_config(page_title="Income Predictor", page_icon="💰")

@st.cache_resource
def load_assets():
    model = joblib.load('optimized_xgboost_hyperparameters_smote_feature-engineering_model.joblib')
    model_columns = joblib.load('optimized_xgboost_feature_columns.joblib')
    return model, model_columns

model, model_columns = load_assets()

# --- 2. TITLE & DESCRIPTION ---
st.title("💰 Census Income Predictor")
st.markdown("""
This app predicts if a person earns **> $50k/year** using an **Optimized XGBoost Model**.
""")

st.divider()

# --- 3. INPUT FORM ---
st.sidebar.header("User Details")

# Numerical Inputs
age = st.sidebar.number_input("Age", min_value=17, max_value=90, value=30)
education_num = st.sidebar.slider("Education Level (Years)", 1, 16, 10)
hours_per_week = st.sidebar.number_input("Hours per Week", 1, 99, 40)

capital_gain = st.sidebar.number_input(
    "Capital Gain ($)", 
    min_value=0, 
    max_value=100000, 
    value=0, 
    step=1000
)

capital_loss = st.sidebar.number_input(
    "Capital Loss ($)", 
    min_value=0, 
    max_value=5000, 
    value=0, 
    step=100
)

# Categorical Inputs
marital_status = st.sidebar.selectbox("Marital Status", 
    ["Married", "Single", "Divorced", "Widowed", "Separated"])

relationship = st.sidebar.selectbox("Relationship Role", 
    ["Husband", "Wife", "Own-child", "Unmarried", "Not-in-family"])

occupation = st.sidebar.selectbox("Occupation Category", 
    ["Exec-managerial", "Prof-specialty", "Sales", "Craft-repair", "Other"])

# --- 4. PRE-PROCESSING ---
processed_data = pd.DataFrame(columns=model_columns)
processed_data.loc[0] = 0

processed_data['age'] = age
processed_data['education-num'] = education_num
processed_data['hours-per-week'] = hours_per_week
processed_data['capital-gain'] = capital_gain
processed_data['capital-loss'] = capital_loss

# Calculate Super Features
processed_data['Age-x-Education'] = age * education_num
processed_data['Capital-Gain-per-Hour'] = capital_gain / (hours_per_week + 1)

# Handle the Categorical "Switches"
if marital_status == "Married":
    col_name = 'marital-status_Married-civ-spouse'
    if col_name in processed_data.columns:
        processed_data[col_name] = 1

if occupation == "Exec-managerial":
    col_name = 'occupation_Exec-managerial'
    if col_name in processed_data.columns:
        processed_data[col_name] = 1

if occupation == "Prof-specialty":
    col_name = 'occupation_Prof-specialty'
    if col_name in processed_data.columns:
        processed_data[col_name] = 1

# --- 5. PREDICTION ---
if st.button("Predict Income"):
    prediction_prob = model.predict_proba(processed_data)[0][1]
    prediction_class = (prediction_prob > 0.65).astype(int)

    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Probability of High Income", f"{prediction_prob:.1%}")
    
    with col2:
        if prediction_class == 1:
            st.success("Prediction: **High Income (>50k)**")
        else:
            st.warning("Prediction: **Low Income (<=50k)**")

    st.progress(float(prediction_prob))
    
    st.caption(f"Note: The model flags 'High Income' if the probability is above 65%.")