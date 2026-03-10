import streamlit as st
import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg:       #080b10;
    --surface:  #0f1318;
    --border:   #1c2333;
    --accent:   #3b82f6;
    --accent2:  #06b6d4;
    --muted:    #4b5675;
    --text:     #e2e8f0;
    --green:    #10b981;
    --red:      #ef4444;
    --yellow:   #f59e0b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}
.stApp { background-color: var(--bg) !important; }

/* Hide sidebar & hamburger */
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* Center content with max width */
.block-container {
    max-width: 780px !important;
    padding: 2.5rem 1.5rem 4rem !important;
}

/* Header */
.header { margin-bottom: 2.5rem; }
.header-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 0.5rem;
}
.header-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.header-sub {
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.4rem;
}

/* Section label */
.field-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.75rem 0 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* Selectbox overrides */
[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
[data-baseweb="select"] * { color: var(--text) !important; }
[data-baseweb="popover"] { background-color: #131820 !important; border: 1px solid var(--border) !important; }
[role="option"]:hover { background-color: #1c2535 !important; }
label { color: var(--muted) !important; font-size: 0.78rem !important; }

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* Result card */
.result-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 0.5rem;
}
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}
.result-verdict {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.verdict-safe  { color: var(--green); }
.verdict-churn { color: var(--red); }
.result-prob {
    font-family: 'DM Mono', monospace;
    font-size: 2.8rem;
    font-weight: 500;
    line-height: 1;
}
.prob-safe  { color: var(--green); }
.prob-churn { color: var(--red); }
.prob-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.2rem;
    text-align: right;
}

/* Gauge */
.gauge-track {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--green) 0%, var(--yellow) 50%, var(--red) 100%);
    position: relative;
    margin: 1.25rem 0 0.4rem;
}
.gauge-dot {
    position: absolute;
    top: -5px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 0 0 3px var(--bg), 0 0 0 5px white;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-low    { background: #052e16; color: var(--green); border: 1px solid #14532d; }
.badge-medium { background: #1c1307; color: var(--yellow); border: 1px solid #78350f; }
.badge-high   { background: #1c0a0a; color: var(--red);   border: 1px solid #7f1d1d; }

/* Insight box */
.insight {
    background: #0a0f18;
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 1.25rem;
    line-height: 1.5;
}

/* Summary grid */
.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    margin-top: 1.25rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}
.summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
}
.summary-row:nth-child(odd) { background: #0a0d13; }
.summary-key { color: var(--muted); }
.summary-val { color: var(--text); font-weight: 600; font-family: 'DM Mono', monospace; font-size: 0.75rem; }

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    margin-top: 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Footer */
.footer {
    text-align: center;
    font-size: 0.7rem;
    color: #1e2535;
    margin-top: 3rem;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <div class="header-eyebrow">Customer Intelligence</div>
    <div class="header-title">Churn Predictor</div>
    <div class="header-sub">Fill in the customer details below to get an instant churn risk assessment.</div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="field-section">Demographics</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
with c2:
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
with c3:
    age = st.selectbox("Age", list(range(18, 93)), index=17)

st.markdown('<div class="field-section">Financials</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    credit_score = st.selectbox("Credit Score", list(range(300, 901, 10)), index=35)
with c5:
    balance = st.selectbox("Balance ($)", [0,10000,20000,30000,40000,50000,60000,
        70000,80000,90000,100000,125000,150000,175000,200000,250000], index=5)
with c6:
    estimated_salary = st.selectbox("Est. Salary ($)", [10000,20000,30000,40000,
        50000,60000,70000,80000,90000,100000,120000,150000,175000,200000], index=5)

st.markdown('<div class="field-section">Account Details</div>', unsafe_allow_html=True)
c7, c8, c9, c10 = st.columns(4)
with c7:
    tenure = st.selectbox("Tenure (yrs)", list(range(0, 11)), index=5)
with c8:
    num_of_products = st.selectbox("Products", [1, 2, 3, 4])
with c9:
    has_cr_card = st.selectbox("Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
with c10:
    is_active_member = st.selectbox("Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No")

# ── Prediction ────────────────────────────────────────────────────────────────
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age], 'Tenure': [tenure], 'Balance': [balance],
    'NumOfProducts': [num_of_products], 'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member], 'EstimatedSalary': [estimated_salary],
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
risk_label = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
badge_cls  = "badge-high" if prob > 0.7 else ("badge-medium" if prob > 0.4 else "badge-low")
verdict_cls = "verdict-churn" if will_churn else "verdict-safe"
prob_cls    = "prob-churn" if will_churn else "prob-safe"
verdict_text = "Likely to Churn" if will_churn else "Likely to Retain"
insight = (
    f"This customer shows a <strong>{prob:.1%} churn probability</strong>. "
    "Consider a proactive retention offer — a personalised discount or loyalty reward could reduce risk significantly."
    if will_churn else
    f"This customer shows a <strong>{prob:.1%} churn probability</strong>. "
    "Low risk — continue standard engagement. Monitor quarterly for any changes in activity."
)

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

left_pct = max(1, min(99, risk_pct))
st.markdown(f"""
<div class="result-wrap">
    <div class="result-header">
        <div>
            <div class="result-verdict {verdict_cls}">{'⚠️' if will_churn else '✅'} {verdict_text}</div>
            <div style="margin-top:0.5rem;">
                <span class="risk-badge {badge_cls}">{risk_label} Risk</span>
            </div>
        </div>
        <div style="text-align:right;">
            <div class="result-prob {prob_cls}">{prob:.1%}</div>
            <div class="prob-label">Churn Probability</div>
        </div>
    </div>

    <div class="gauge-track">
        <div class="gauge-dot" style="left:{left_pct}%;"></div>
    </div>
    <div class="gauge-labels">
        <span>0%</span><span>50%</span><span>100%</span>
    </div>

    <div class="insight">{insight}</div>

    <div style="margin-top:1.5rem;">
        <div style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--muted);margin-bottom:0.75rem;">Input Summary</div>
        <div class="summary-grid">
            <div class="summary-row"><span class="summary-key">Geography</span><span class="summary-val">{geography}</span></div>
            <div class="summary-row"><span class="summary-key">Gender</span><span class="summary-val">{gender}</span></div>
            <div class="summary-row"><span class="summary-key">Age</span><span class="summary-val">{age}</span></div>
            <div class="summary-row"><span class="summary-key">Tenure</span><span class="summary-val">{tenure} yrs</span></div>
            <div class="summary-row"><span class="summary-key">Credit Score</span><span class="summary-val">{credit_score:,}</span></div>
            <div class="summary-row"><span class="summary-key">Balance</span><span class="summary-val">${balance:,.0f}</span></div>
            <div class="summary-row"><span class="summary-key">Est. Salary</span><span class="summary-val">${estimated_salary:,.0f}</span></div>
            <div class="summary-row"><span class="summary-key">Products</span><span class="summary-val">{num_of_products}</span></div>
            <div class="summary-row"><span class="summary-key">Credit Card</span><span class="summary-val">{'Yes' if has_cr_card else 'No'}</span></div>
            <div class="summary-row"><span class="summary-key">Active Member</span><span class="summary-val">{'Yes' if is_active_member else 'No'}</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='footer'>Powered by ANN · Predictions are probabilistic estimates</div>", unsafe_allow_html=True)
