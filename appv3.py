import streamlit as st
import pandas as pd
import joblib
import json

# Load model
model = joblib.load("fraud_model.pkl")

# Load MCC mapping
with open("mcc_codes.json") as f:
    mcc_map = json.load(f)

# Reverse mapping: category → code
mcc_options = {v: int(k) for k, v in mcc_map.items()}
mcc_list = list(mcc_options.keys())

st.title("💳 AI Fraud Detection System")

st.header("🔍 Transaction Analysis")

# -------------------------
# INPUT SECTION (IMPROVED)
# -------------------------
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    # ✅ MCC DROPDOWN
    mcc_category = st.selectbox(
        "🏪 Merchant Category",
        mcc_list
    )

    mcc = mcc_options[mcc_category]

    age = st.number_input(
        "👤 User Age",
        value=30,
        step=1
    )

    credit_score = st.number_input(
        "📊 Credit Score",
        value=700,
        step=10
    )

with col2:
    num_cards = st.number_input(
        "💳 Number of Credit Cards",
        value=2,
        step=1
    )

    dark_web = st.selectbox(
        "🕵️ Card on Dark Web",
        ["No", "Yes"]
    )

    pin_year = st.number_input(
        "🔐 PIN Last Changed Year",
        value=2020,
        step=1
    )

# -------------------------
# PREDICTION
# -------------------------
if st.button("🚀 Analyze Transaction"):

    dark_web_val = 1 if dark_web == "Yes" else 0

    input_data = pd.DataFrame([[
        amount, mcc, age, credit_score,
        num_cards, dark_web_val, pin_year
    ]], columns=[
        "amount", "mcc", "current_age", "credit_score",
        "num_credit_cards", "card_on_dark_web",
        "year_pin_last_changed"
    ])

    prob = model.predict_proba(input_data)[0][1]
    prediction = 1 if prob > 0.7 else 0

    risk_score = prob * 100

    # -------------------------
    # RESULT DISPLAY
    # -------------------------
    st.subheader("📊 Analysis Result")

    st.write(f"### 🔢 Risk Score: {risk_score:.2f}/100")
    st.progress(int(risk_score))

    if prob < 0.3:
        st.success("🟢 Low Risk Transaction")
    elif prob < 0.7:
        st.warning("🟡 Medium Risk Transaction")
    else:
        st.error("🔴 High Risk Transaction")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    # -------------------------
    # EXPLANATION
    # -------------------------
    st.subheader("🧠 Why this decision?")

    reasons = []

    if amount > 3000:
        reasons.append("💰 High transaction amount")

    # ✅ Updated logic using category
    if mcc_category in [
        "Money Transfer",
        "Betting (including Lottery Tickets, Casinos)"
    ]:
        reasons.append("🏪 Risky merchant category")

    if dark_web_val == 1:
        reasons.append("🕵️ Card found on dark web")

    if credit_score < 600:
        reasons.append("📉 Low credit score")

    if (2025 - pin_year) > 5:
        reasons.append("🔐 PIN not changed recently")

    if len(reasons) == 0:
        st.write("✔️ No strong risk indicators detected")
    else:
        for r in reasons:
            st.write(f"- {r}")

    # -------------------------
    # TRANSACTION SUMMARY
    # -------------------------
    st.subheader("📄 Transaction Summary")

    st.write(f"💰 Amount: ₹{amount}")
    st.write(f"🏪 Merchant Category: {mcc_category}")  # ✅ updated
    st.write(f"👤 Age: {age}")
    st.write(f"📊 Credit Score: {credit_score}")