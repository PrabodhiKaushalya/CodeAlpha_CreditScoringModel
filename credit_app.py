import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Credit Scoring Dashboard", page_icon="💳", layout="wide")

# --- 2. PROFESSIONAL CSS (With Compact Button) ---
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.95)),
                          url('https://images.unsplash.com/photo-1518770660439-4636190af475');
        background-size: cover; background-position: center; background-attachment: fixed; color: white;
    }
    /* Compact Green Button Styling */
    div.stButton > button:first-child {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 8px 40px !important;
        font-weight: bold !important;
        border: none !important;
        display: block;
        margin: 0 auto; /* Center the button */
    }
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODELS ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
except:
    st.error("Model files missing. Please run notebook first.")
    st.stop()

# --- 4. UI DESIGN ---
st.title("💳 ADVANCED CREDIT RISK ANALYTICS")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Personal")
    age = st.slider("Age", 18, 100, 30)
    income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    st.subheader("Loan Details")
    emp_len = st.number_input("Employment (Years)", 0, 50, 5)
    loan_amt = st.number_input("Loan Amount ($)", 0, 1000000, 15000)
    intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME_IMPROVEMENT", "DEBT_CONSOLIDATION"])

with col3:
    st.subheader("Status & History")
    int_rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 10.5)
    cred_hist = st.number_input("Credit History (Years)", 0, 50, 7)
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

st.markdown("<br>", unsafe_allow_html=True) # Space before button

# --- 5. PREDICTION LOGIC ---
if st.button("PREDICT RISK"):
    # Initial data
    data = {
        'person_age': [age], 'person_income': [income], 
        'person_emp_length': [emp_len], 'loan_amnt': [loan_amt], 
        'loan_int_rate': [int_rate], 'cb_person_cred_hist_length': [cred_hist]
    }
    
    # Categorical Inputs (Mapping to dummies)
    # We create the dummies manually to match model_columns
    cats = {
        f'person_home_ownership_{home}': 1,
        f'loan_intent_{intent}': 1,
        f'loan_grade_{grade}': 1
    }
    
    input_df = pd.DataFrame(data)
    
    # Add dummies from cats and fill missing model_columns with 0
    for col in model_columns:
        if col in cats:
            input_df[col] = 1
        elif col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[model_columns] # Final order check
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    st.markdown("---")
    if prediction[0] == 0:
        st.success("✅ **PREDICTED STATUS: LOW RISK (SAFE)**")
        st.balloons()
    else:
        st.error("❌ **PREDICTED STATUS: HIGH RISK (DEFAULT)**")