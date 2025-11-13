import streamlit as st
import numpy as np
import joblib
import gdown

clf_name = 'emi_classifier.joblib'
clf_id = '1UTWRuMyy1W28E748wWlgXrg3adh2Vz2J'

reg_name = 'emi_regressor.joblib'
reg_id = '1bh1lzCAHRDsdlncBa-4h1vbxalh1lJUZ'

slc_name = 'scaler.joblib'
slc_id = '1ucVgZDL4-oxYk4J4TwU4Yw6qUGTJQWMU'


# -------------------------------
# Load trained model and scaler
# -------------------------------
@st.cache_resource
def load_model():
    try:
        clf_url = f"https://drive.google.com/uc?id={clf_id}"
        gdown.download(clf_url, clf_name, quiet=False)
        clf_model = joblib.load(clf_name)
    except FileNotFoundError:
        clf_model = None
   
    try:
        reg_url = f"https://drive.google.com/uc?id={clf_id}"
        gdown.download(reg_url, reg_name, quiet=False)
        clf_model = joblib.load(reg_name)
    except FileNotFoundError:
        reg_model = None

    try:
        slc_url = f"https://drive.google.com/uc?id={clf_id}"
        gdown.download(slc_url, slc_name, quiet=False)
        scaler = joblib.load(slc_name)
    except FileNotFoundError:
        scaler = None

    return clf_model, reg_model, scaler

clf_model, reg_model, scaler = load_model()

st.title("💰 Financial Risk Assessment: EMI Eligibility & Maximum EMI Prediction")

st.markdown("This app predicts **EMI eligibility** and estimates your **maximum affordable EMI** using financial indicators.")

# -------------------------------
# User Inputs
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0.0, step=1000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=10)
    years_of_employment = st.number_input("Years of Employment", min_value=0.0, step=0.5)
    monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0.0, step=500.0)
    current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0.0, step=500.0)

with col2:
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0.0, step=500.0)
    travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0.0, step=500.0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0.0, step=500.0)
    emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0.0, step=500.0)
    bank_balance = st.number_input("Bank Balance (₹)", min_value=0.0, step=500.0)

st.divider()

# 🔍 Predict
if st.button("🔮 Predict EMI Eligibility & Amount"):
    try:
        # Prepare input data
        features = np.array([
            monthly_salary, credit_score, years_of_employment,
            monthly_rent, current_emi_amount, groceries_utilities,
            travel_expenses, other_monthly_expenses, emergency_fund, bank_balance
        ]).reshape(1, -1)
        
        # Scale input
        features_scaled = scaler.transform(features)
        
        # Classification (Eligibility)
        pred_class = clf_model.predict(features_scaled)[0]
        pred_probs = clf_model.predict_proba(features_scaled)[0]

        # Convert numeric prediction to label if needed
        label_map = {0: "not_eligible", 1: "eligible", 2: "high_risk"}
        if isinstance(pred_class, (int, np.integer)):
            pred_class_str = label_map.get(pred_class, str(pred_class))
        else:
            pred_class_str = str(pred_class)
        
        # Regression (Max EMI)
        pred_emi = reg_model.predict(features_scaled)[0]
        
        # Display classification
        st.subheader("🎯 EMI Eligibility Prediction")
        
        st.write(f"**Status:** {pred_class_str.upper()}")
        st.progress(float(max(pred_probs)))
        st.write(f"Confidence: **{max(pred_probs)*100:.2f}%**")
        
        # Display regression result
        st.subheader("💰 Estimated Maximum Affordable EMI")
        st.write(f"₹ **{pred_emi:,.2f}** per month")
        
        # Risk interpretation
        if pred_class_str.lower() == "eligible":
            st.success("✅ You are likely eligible for EMI payments.")
        elif pred_class_str.lower() == "high_risk":
            st.warning("⚠️ High risk — consider improving your credit and savings before taking a loan.")
        else:
            st.error("❌ Not eligible — income or financial health may be insufficient.")
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
