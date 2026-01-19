import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AdTarget AI | Premium Audience Scout",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #31333F;
        padding: 15px;
        height: 140px;
        border-radius: 8px;
        color: #FFFFFF;
    }
    
    div[data-testid="stMetricLabel"] > label {
        color: #B2B5C9;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #18191E;
    }
    
    div.stButton > button {
        background-color: #FFFFFF;
        color: #000000;
        border: none;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #E0E0E0;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        bundle = joblib.load('Final_Model.joblib')
        return bundle['model'], bundle['model_columns'], bundle['threshold']
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'Final_Model.joblib' is in the folder.")
        st.stop()

model, model_columns, threshold = load_assets()

# --- DASHBOARD HEADER ---
with st.container():
    st.title("🎯 AdTarget AI")
    st.markdown("#### High-Value Audience Identification System")
    st.caption("Objective: Maximize ROAS by targeting users with high disposable income.")

st.divider()

# --- SIDEBAR - USER INPUTS ---
st.sidebar.header("👤 Lead Profile Parameters")
st.sidebar.markdown("Define the target attributes to test eligibility.")

with st.sidebar.expander("Demographics", expanded=True):
    age = st.slider("Age", 17, 90, 35)
    education_num = st.slider("Education Score", 1, 16, 12, help="13=Bachelors, 16=PhD")
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Separated"])
    relationship = st.selectbox("Relationship Role", ["Husband", "Wife", "Own-child", "Unmarried", "Not-in-family"])

with st.sidebar.expander("Employment & Financial", expanded=True):
    occupation = st.selectbox("Occupation Sector", ["Exec-managerial", "Prof-specialty", "Sales", "Craft-repair", "Other"])
    hours_per_week = st.number_input("Work Intensity (Hrs/Wk)", 1, 99, 45)
    capital_gain = st.number_input("Asset Gains ($)", 0, 100000, 0, step=1000)
    capital_loss = st.number_input("Asset Losses ($)", 0, 5000, 0, step=100)

# --- DATA PROCESSING ENGINE ---
processed_data = pd.DataFrame(columns=model_columns)
processed_data.loc[0] = 0

if 'age' in processed_data.columns: processed_data['age'] = age
if 'education-num' in processed_data.columns: processed_data['education-num'] = education_num
if 'hours-per-week' in processed_data.columns: processed_data['hours-per-week'] = hours_per_week
if 'capital-gain' in processed_data.columns: processed_data['capital-gain'] = capital_gain
if 'capital-loss' in processed_data.columns: processed_data['capital-loss'] = capital_loss

if 'Age-x-Education' in processed_data.columns:
    processed_data['Age-x-Education'] = age * education_num
if 'Capital-Gain-per-Hour' in processed_data.columns:
    processed_data['Capital-Gain-per-Hour'] = capital_gain / (hours_per_week + 1)

if marital_status == "Married":
    col = 'marital-status_Married-civ-spouse'
    if col in processed_data.columns: processed_data[col] = 1

if occupation == "Exec-managerial":
    col = 'occupation_Exec-managerial'
    if col in processed_data.columns: processed_data[col] = 1

if occupation == "Prof-specialty":
    col = 'occupation_Prof-specialty'
    if col in processed_data.columns: processed_data[col] = 1

# --- PREDICTION & BUSINESS INTELLIGENCE ---
left_col, right_col = st.columns([1, 2])

with left_col:
    st.info("Click below to analyze this profile's eligibility for premium audience targeting.", icon="ℹ️")
    if st.button("🚀 Analyze Lead Eligibility", use_container_width=True):
        
        processed_data = processed_data[model_columns]
        
        prediction_prob = model.predict_proba(processed_data)[0][1]
        prediction_class = (prediction_prob >= threshold).astype(int)
        
        st.session_state['prob'] = prediction_prob
        st.session_state['class'] = prediction_class
        st.session_state['run'] = True

with right_col:
    if 'run' in st.session_state and st.session_state['run']:
        prob = st.session_state['prob']
        res_class = st.session_state['class']
        
        st.subheader("Analysis Results")
        
        c1, c2 = st.columns(2)
        
        with c1:
            if res_class == 1:
                st.metric(label="Audience Segment", value="PREMIUM", delta="Approved")
            else:
                st.metric(label="Audience Segment", value="Standard", delta="- Rejected", delta_color="inverse")
        
        with c2:
             st.metric(label="Lead Quality Score", value=f"{prob:.1%}")

        st.write("Targeting Probability:")

        safe_prob = float(prob)

        if res_class == 1:
            st.progress(safe_prob, text="✅ High probability of premium conversion")
            st.success("**Recommendation:** Include in premium campaigns. Prioritize for high-value ads.")
        else:
            st.progress(safe_prob, text="⚠️ Low probability of premium conversion")
            st.warning("**Recommendation:** Exclude from premium campaigns. Serve generic/discount ads only.")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #666;">
            Waiting for input... configure the sidebar and click 'Analyze Lead'.
        </div>
        """, unsafe_allow_html=True)