import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SETUP & LOAD MODEL ---
st.set_page_config(page_title="Income Predictor", page_icon="💰")

@st.cache_resource
def load_assets():
    # Load the trained model and the column list
    model = joblib.load('optimized_xgboost_hyperparameters_smote_feature-engineering_model.joblib')
    model_columns = joblib.load('optimized_xgboost_hyperparameters_smote_feature-engineering_model_columns.joblib')
    return model, model_columns

model, model_columns = load_assets()

# --- 2. TITLE & DESCRIPTION ---
st.title("💰 Census Income Predictor")
st.markdown("""
This app predicts if a person earns **> $50k/year** using an **Optimized XGBoost Model**.
It uses advanced features like *Capital Gain per Hour* and *Age-Education Interaction*.
""")

st.divider()

# --- 3. INPUT FORM (SIDEBAR) ---
st.sidebar.header("User Details")

# Numerical Inputs
age = st.sidebar.number_input("Age", min_value=17, max_value=90, value=30)
education_num = st.sidebar.slider("Education Level (Years)", 1, 16, 10)
hours_per_week = st.sidebar.number_input("Hours per Week", 1, 99, 40)
capital_gain = st.sidebar.number_input("Capital Gain ($)", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss ($)", 0, 5000, 0)

# Categorical Inputs (Drop-downs)
marital_status = st.sidebar.selectbox("Marital Status", 
    ["Married", "Single", "Divorced", "Widowed", "Separated"])

relationship = st.sidebar.selectbox("Relationship Role", 
    ["Husband", "Wife", "Own-child", "Unmarried", "Not-in-family"])

occupation = st.sidebar.selectbox("Occupation Category", 
    ["Exec-managerial", "Prof-specialty", "Sales", "Craft-repair", "Other"])

# --- 4. PRE-PROCESSING (THE TRICKY PART) ---
# We need to turn these inputs into the exact format the model expects

# A. Create a dictionary of raw data
input_data = {
    'age': age,
    'education_num': education_num,
    'hours_per_week': hours_per_week,
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    # We map the text inputs to the dummy column format manually if needed, 
    # but the easier way is to create the dataframe and reindex.
}

# B. Calculate "Super Features" (Feature Engineering)
# (Must match the logic you used in the notebook!)
input_data['Age_x_Education'] = age * education_num
input_data['Capital_Gain_per_Hour'] = capital_gain / (hours_per_week + 1)

# C. Handle One-Hot Encoding Logic
# We create a dataframe with the user's choices
df_input = pd.DataFrame([input_data])

# We need to manually "fake" the One-Hot Encoding for the categories
# For example, if user picked "Married", we need to set 'marital-status_Married-civ-spouse' to 1.
# A robust shortcut for Streamlit apps:
processed_data = pd.DataFrame(columns=model_columns)
processed_data.loc[0] = 0  # Initialize all columns to 0

# Fill in the numericals
processed_data['age'] = age
processed_data['education_num'] = education_num
processed_data['hours_per_week'] = hours_per_week
processed_data['capital_gain'] = capital_gain
processed_data['capital_loss'] = capital_loss
processed_data['Age_x_Education'] = input_data['Age_x_Education']
processed_data['Capital_Gain_per_Hour'] = input_data['Capital_Gain_per_Hour']

# Handle the Categorical "Switches"
# Note: You might need to adjust these strings to match EXACTLY what is in 'model_columns.pkl'
if marital_status == "Married":
    if 'marital-status_Married-civ-spouse' in processed_data.columns:
        processed_data['marital-status_Married-civ-spouse'] = 1

if occupation == "Exec-managerial":
    if 'occupation_Exec-managerial' in processed_data.columns:
        processed_data['occupation_Exec-managerial'] = 1

if occupation == "Prof-specialty":
    if 'occupation_Prof-specialty' in processed_data.columns:
        processed_data['occupation_Prof-specialty'] = 1

# --- 5. PREDICTION ---
if st.button("Predict Income"):
    # Get Probability
    prediction_prob = model.predict_proba(processed_data)[0][1]
    prediction_class = (prediction_prob > 0.65).astype(int) # Using your Business Threshold of 0.65!

    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Probability of High Income", f"{prediction_prob:.1%}")
    
    with col2:
        if prediction_class == 1:
            st.success("Prediction: **High Income (>50k)**")
        else:
            st.warning("Prediction: **Low Income (<=50k)**")

    # Visual Bar
    st.progress(float(prediction_prob))
    
    # Explain the Threshold
    st.caption(f"Note: The model flags 'High Income' if the probability is above 65% (High Precision Strategy).")