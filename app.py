import streamlit as st
import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d0f14; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #7dd3fc;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2330;
}

input[type="number"], input[type="text"] {
    background-color: #1a1e2b !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3347 !important;
    border-radius: 6px !important;
}
[data-baseweb="select"] {
    background-color: #1a1e2b !important;
    border: 1px solid #2d3347 !important;
    border-radius: 6px !important;
}
[data-baseweb="select"] * {
    background-color: #1a1e2b !important;
    color: #e2e8f0 !important;
}

.metric-card {
    background: linear-gradient(135deg, #141824 0%, #1a1f2e 100%);
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    line-height: 1;
}
.metric-value.churn { color: #f87171; }
.metric-value.safe  { color: #34d399; }
.metric-value.prob  { color: #7dd3fc; }

.result-banner {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.result-banner.churn {
    background: linear-gradient(135deg, #2d1515 0%, #1f1220 100%);
    border: 1px solid #7f1d1d;
}
.result-banner.safe {
    background: linear-gradient(135deg, #0d2318 0%, #0f1f1a 100%);
    border: 1px solid #14532d;
}
.result-title { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.2rem; }
.result-title.churn { color: #fca5a5; }
.result-title.safe  { color: #6ee7b7; }
.result-subtitle { font-size: 0.85rem; color: #94a3b8; }

.section-header {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    font-weight: 600;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2330;
    margin-bottom: 1rem;
}
.app-title { font-size: 1.9rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; }
.app-subtitle { font-size: 0.9rem; color: #64748b; margin-top: 0.25rem; }
hr { border-color: #1e2330; }

.gauge-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #34d399 0%, #facc15 50%, #f87171 100%);
    position: relative;
    margin: 0.75rem 0;
}
.gauge-indicator {
    position: absolute;
    top: -4px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 0 0 3px #0d0f14, 0 0 0 5px white;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & encoders ─────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    session = ort.InferenceSession('model.onnx')
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return session, le_gender, ohe_geo, scaler

session, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Customer Profile")
    st.markdown("## 🌍 Demographics")
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 92, 35)

    st.markdown("## 💰 Financials")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=100.0, format="%.2f")
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0, format="%.2f")

    st.markdown("## 🏦 Account Details")
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    num_of_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">📉 Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Machine learning model to identify at-risk customers</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Build & run prediction ────────────────────────────────────────────────────
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data).astype(np.float32)

input_name = session.get_inputs()[0].name
prediction = session.run(None, {input_name: input_data_scaled})[0]
prob = float(prediction[0][0])
will_churn = prob > 0.5
risk_pct = int(prob * 100)

# ── Results ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    color_cls = "churn" if will_churn else "safe"
    label = "LIKELY TO CHURN" if will_churn else "LIKELY TO RETAIN"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Prediction</div>
        <div class="metric-value {color_cls}" style="font-size:1.3rem; margin-top:0.3rem;">{label}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Churn Probability</div>
        <div class="metric-value prob">{prob:.1%}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    risk_label = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
    risk_color = "#f87171" if prob > 0.7 else ("#facc15" if prob > 0.4 else "#34d399")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Risk Level</div>
        <div class="metric-value" style="color:{risk_color}; font-size:2rem;">{risk_label}</div>
    </div>""", unsafe_allow_html=True)

# Risk gauge
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Risk Gauge</div>', unsafe_allow_html=True)
left_pct = max(0, min(99, risk_pct))
st.markdown(f"""
<div class="gauge-bar">
    <div class="gauge-indicator" style="left: {left_pct}%;"></div>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#475569; margin-top:4px;">
    <span>Low Risk</span><span>Medium Risk</span><span>High Risk</span>
</div>
""", unsafe_allow_html=True)

# Result banner
st.markdown("<br>", unsafe_allow_html=True)
if will_churn:
    st.markdown(f"""
    <div class="result-banner churn">
        <span style="font-size:2rem;">⚠️</span>
        <div>
            <div class="result-title churn">High Churn Risk Detected</div>
            <div class="result-subtitle">This customer has a {prob:.1%} probability of churning. Consider a targeted retention offer or outreach campaign.</div>
        </div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="result-banner safe">
        <span style="font-size:2rem;">✅</span>
        <div>
            <div class="result-title safe">Customer Likely to Stay</div>
            <div class="result-subtitle">This customer has only a {prob:.1%} probability of churning. Continue standard engagement practices.</div>
        </div>
    </div>""", unsafe_allow_html=True)

# Input summary
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
summary_items = [
    ("Geography", geography), ("Gender", gender), ("Age", age),
    ("Tenure", f"{tenure} yrs"), ("Credit Score", f"{credit_score:,}"),
    ("Balance", f"${balance:,.2f}"), ("Est. Salary", f"${estimated_salary:,.2f}"),
    ("Products", num_of_products), ("Credit Card", "Yes" if has_cr_card else "No"),
    ("Active Member", "Yes" if is_active_member else "No"),
]
half = len(summary_items) // 2
with c1:
    for k, v in summary_items[:half]:
        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid #1e2330;font-size:0.85rem;'><span style='color:#64748b;'>{k}</span><span style='color:#e2e8f0;font-weight:600;'>{v}</span></div>", unsafe_allow_html=True)
with c2:
    for k, v in summary_items[half:]:
        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid #1e2330;font-size:0.85rem;'><span style='color:#64748b;'>{k}</span><span style='color:#e2e8f0;font-weight:600;'>{v}</span></div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='color:#334155; font-size:0.75rem; text-align:center;'>Powered by a neural network trained on bank customer data · Predictions are probabilistic, not deterministic</div>", unsafe_allow_html=True)
