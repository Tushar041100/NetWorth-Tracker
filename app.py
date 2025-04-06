import streamlit as st
import joblib
import numpy as np
import pandas as pd
from validation import AssetAllocationRequest, AssetAllocationResponse
from explain_and_validate import explain_recommendation, enforce_regulatory_safeguards, log_recommendation
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import os
 
# Load model and scaler
model = joblib.load("asset_allocation_model.pkl")
scaler = joblib.load("scaler.pkl")
 
# Label encoders to match training
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
goal_map = {'Retirement': 0, 'Wealth Creation': 1, 'Child Education': 2, 'Travel': 3}
experience_map = {'Beginner': 0, 'Intermediate': 1, 'Expert': 2}
 
# Streamlit app
st.title("Personalized AI Financial Advisor")
st.subheader("Get your ideal asset allocation based on your profile")
 
with st.form("user_input_form"):
    age = st.slider("Age", 18, 100, 30)
    income = st.number_input("Monthly Income (INR)", min_value=1000.0, value=50000.0)
    expenses = st.number_input("Monthly Expenses (INR)", min_value=0.0, value=20000.0)
    horizon = st.slider("Investment Horizon (Years)", 0, 40, 10)
    risk = st.selectbox("Risk Tolerance", ['Low', 'Medium', 'High'])
    goal = st.selectbox("Financial Goal", ['Retirement', 'Wealth Creation', 'Child Education', 'Travel'])
    experience = st.selectbox("Investment Experience", ['Beginner', 'Intermediate', 'Expert'])
    submitted = st.form_submit_button("Get Asset Allocation")
 
if submitted:
    try:
        # Validate request
        user_data = AssetAllocationRequest(
            age=age,
            income_per_month=income,
            monthly_expenses=expenses,
            goal_horizon_years=horizon,
            risk_tolerance=risk,
            financial_goal=goal,
            investment_experience=experience
        )
 
        # Feature engineering
        savings_ratio = (income - expenses) / income
        is_long_term = 1 if horizon > 10 else 0
        income_bracket = 0 if income <= 50000 else (1 if income <= 150000 else 2)
 
        features = np.array([[
            age,
            income,
            expenses,
            horizon,
            risk_map[risk],
            goal_map[goal],
            experience_map[experience],
            savings_ratio,
            is_long_term,
            income_bracket
        ]])
 
        features[:, 0:4] = scaler.transform(features[:, 0:4])
        prediction = model.predict(features).flatten()
 
        percentages = (prediction / np.sum(prediction)) * 100
        equity = round(percentages[0], 2)
        debt = round(percentages[1], 2)
        gold = round(percentages[2], 2)
        real_estate = round(100 - (equity + debt + gold), 2)
 
        response = AssetAllocationResponse(
            equity_percent=equity,
            debt_percent=debt,
            gold_percent=gold,
            real_estate_percent=real_estate
        )
 
        # Enforce safeguards and explanations
        response = enforce_regulatory_safeguards(user_data, response)
        explanation = explain_recommendation(user_data, response)
 
        # Logging
        log_recommendation(user_data, response, explanation)
 
        # Show results
        st.success("Asset Allocation Recommendation")
        st.write("**Your Recommended Allocation (%):**")
        st.write(response.model_dump())
        st.write("**Explanation:**")
        st.write(explanation)
 
        # Pie chart
        st.pyplot(pd.Series(response.model_dump()).plot.pie(autopct="%1.1f%%", figsize=(6, 6)).get_figure())
 
        # PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Asset Allocation Report", ln=True, align='C')
        pdf.ln(10)
 
        # Add user info
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 8, txt="User Profile:", ln=True)
        for k, v in user_data.model_dump().items():
            pdf.cell(200, 8, txt=f"{k.replace('_', ' ').title()}: {v}", ln=True)
 
        pdf.ln(5)
        pdf.cell(200, 8, txt="Asset Allocation (%):", ln=True)
        for k, v in response.model_dump().items():
            pdf.cell(200, 8, txt=f"{k.replace('_', ' ').title()}: {v}%", ln=True)
 
        pdf.ln(5)
        pdf.multi_cell(0, 8, txt=f"Explanation: {explanation}")
 
        # Save report
        if not os.path.exists("reports"):
            os.makedirs("reports")
        report_path = f"reports/allocation_report_{user_data.age}_{user_data.risk_tolerance}.pdf"
        pdf.output(report_path)
 
        st.download_button(
            label="Download PDF Report",
            data=open(report_path, "rb").read(),
            file_name="Asset_Allocation_Report.pdf"
        )
 
    except Exception as e:
        st.error(f"Validation Error: {str(e)}")