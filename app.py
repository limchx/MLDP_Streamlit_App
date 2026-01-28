import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(current_dir, 'final_model_bundle.pkl')
        
        bundle = joblib.load(model_path)
        
        m = bundle.get('model')
        c = bundle.get('model_columns') or bundle.get('features')
        t = bundle.get('threshold')
        
        if m is None or c is None or t is None:
            st.error("❌ **Bundle Error:** Required components are missing from the .pkl file.")
            st.info("Ensure your Notebook save block uses keys: 'model', 'model_columns', 'threshold'")
            return None, None, None
            
        return m, c, t

    except FileNotFoundError:
        st.error(f"⚠️ **File Not Found:** Could not find file at {model_path}")
        st.info("Make sure 'final_model_bundle.pkl' is in the same folder as 'app.py'.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ **Unexpected Error:** {e}")
        return None, None, None
    
model, model_columns, threshold = load_assets()

# --- STOP EXECUTION IF ASSETS MISSING ---
if model is None or model_columns is None:
    st.warning("The application cannot proceed without a valid model bundle. Please check the error messages above.")
    st.stop()

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
input_dict = {
    'age': age,
    'education-num': education_num,
    'hours-per-week': hours_per_week,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'Net-Capital': capital_gain - capital_loss,
    'Work-Intensity': age * hours_per_week
}

# 2. Add Categorical logic (One-Hot Encoding)
if marital_status == "Married":
    input_dict['marital-status_Married-civ-spouse'] = 1
if relationship == "Husband":
    input_dict['relationship_Husband'] = 1
if occupation == "Exec-managerial":
    input_dict['occupation_Exec-managerial'] = 1
if occupation == "Prof-specialty":
    input_dict['occupation_Prof-specialty'] = 1

processed_data = pd.DataFrame([input_dict])

for col in model_columns:
    if col not in processed_data.columns:
        processed_data[col] = 0

processed_data = processed_data[model_columns]

left_col, right_col = st.columns([1, 2])

with left_col:
    st.info("Click below to analyze this profile's eligibility for premium audience targeting.", icon="ℹ️")
    if st.button("🚀 Analyze Lead Eligibility", use_container_width=True):
        
        prediction_prob = model.predict_proba(processed_data.values)[0][1]
        
        prediction_class = 1 if prediction_prob >= threshold else 0
        
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