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
        bundle = joblib.load('final_model_bundle.pkl')
        
        m = bundle.get('model')
        c = bundle.get('model_columns') or bundle.get('features')
        t = bundle.get('threshold')
        
        if m is None or c is None or t is None:
            st.error("❌ Model bundle is incomplete. Check keys: 'model', 'model_columns', 'threshold'")
            return None, None, None
            
        return m, c, t
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None, None

# --- ASSIGN ASSETS GLOBALLY ---
model, model_columns, threshold = load_assets()

if model is None or model_columns is None:
    st.warning("Please fix the model file and refresh the page.")
    st.stop() 

input_df = pd.DataFrame(columns=model_columns)

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

input_df = pd.DataFrame(columns=model_columns)
input_df.loc[0] = 0

if 'age' in input_df.columns: input_df['age'] = age
if 'education-num' in input_df.columns: input_df['education-num'] = education_num
if 'hours-per-week' in input_df.columns: input_df['hours-per-week'] = hours_per_week
if 'capital-gain' in input_df.columns: input_df['capital-gain'] = capital_gain
if 'capital-loss' in input_df.columns: input_df['capital-loss'] = capital_loss

if 'Net-Capital' in input_df.columns:
    input_df['Net-Capital'] = capital_gain - capital_loss

if 'Work-Intensity' in input_df.columns:
    input_df['Work-Intensity'] = age * hours_per_week

if 'from_rich_region' in input_df.columns:
    rich_regions = ['United-States', 'Canada', 'England', 'Germany', 'France', 'Japan', 'Italy']
    input_df['from_rich_region'] = 1 

# 3. Categorical One-Hot Encoding (Manual mapping for Top Features)
if marital_status == "Married":
    col = 'marital-status_Married-civ-spouse'
    if col in input_df.columns: input_df[col] = 1

if relationship == "Husband":
    col = 'relationship_Husband'
    if col in input_df.columns: input_df[col] = 1

if occupation == "Exec-managerial":
    col = 'occupation_Exec-managerial'
    if col in input_df.columns: input_df[col] = 1

if occupation == "Prof-specialty":
    col = 'occupation_Prof-specialty'
    if col in input_df.columns: input_df[col] = 1

left_col, right_col = st.columns([1, 2])

with left_col:
    st.info("Analyze this lead against the optimized ensemble threshold.", icon="ℹ️")
    if st.button("🚀 Analyze Lead Eligibility", use_container_width=True):
        
        final_features = input_df[model_columns]
        
        prediction_prob = model.predict_proba(final_features.values)[0][1]
        
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