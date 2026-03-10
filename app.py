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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-1: #0a0618;
    --bg-2: #0d0a2e;
    --glass: rgba(255,255,255,0.04);
    --glass-border: rgba(255,255,255,0.09);
    --glass-hover: rgba(255,255,255,0.07);
    --blur: blur(20px);
    --purple: #8b5cf6;
    --blue: #3b82f6;
    --cyan: #06b6d4;
    --green: #10b981;
    --red: #f43f5e;
    --yellow: #f59e0b;
    --text: #f1f5f9;
    --muted: #64748b;
    --dim: #94a3b8;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg-1) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg-1) !important;
    background-image:
        radial-gradient(ellipse 100% 60% at 20% 0%, rgba(139,92,246,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 80% 50% at 80% 10%, rgba(59,130,246,0.14) 0%, transparent 50%),
        radial-gradient(ellipse 60% 40% at 50% 100%, rgba(6,182,212,0.1) 0%, transparent 55%) !important;
    min-height: 100vh;
}

.block-container {
    max-width: 720px !important;
    padding: 0 1.25rem 5rem !important;
}

[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* ── Orbs ── */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
    animation: drift 12s ease-in-out infinite alternate;
}
.orb-1 { width:400px; height:400px; background:rgba(139,92,246,0.12); top:-100px; left:-100px; animation-delay:0s; }
.orb-2 { width:300px; height:300px; background:rgba(59,130,246,0.1); top:30%; right:-80px; animation-delay:-4s; }
.orb-3 { width:250px; height:250px; background:rgba(6,182,212,0.08); bottom:10%; left:20%; animation-delay:-8s; }
@keyframes drift {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(30px, 20px) scale(1.05); }
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 4rem 0 2.5rem;
    position: relative;
    z-index: 1;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    font-weight: 500;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(10px);
}
.hero-pill::before { content: '●'; font-size: 0.4rem; animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.3} }

.hero-title {
    font-size: clamp(2.8rem, 8vw, 4.2rem);
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1.05;
    background: linear-gradient(135deg, #fff 0%, #a78bfa 50%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.88rem;
    color: var(--muted);
    margin-top: 0.75rem;
    font-weight: 300;
    line-height: 1.6;
}

/* ── Glass card ── */
.g-card {
    background: var(--glass);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.75rem;
    position: relative;
    z-index: 1;
    margin-bottom: 1rem;
}
.g-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}

.g-card-title {
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.g-card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--glass-border);
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-baseweb="select"] > div:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(139,92,246,0.4) !important;
}
[data-baseweb="select"] * {
    color: var(--text) !important;
    background: transparent !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div,
[data-baseweb="menu"],
[data-baseweb="menu"] > ul {
    background: #151929 !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(139,92,246,0.25) !important;
    border-radius: 12px !important;
}
[role="option"] {
    color: #c8d4f0 !important;
    background: #151929 !important;
    padding: 0.55rem 1rem !important;
    font-size: 0.85rem !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background: rgba(139,92,246,0.2) !important;
    color: white !important;
}
label {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

input[type="number"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: "JetBrains Mono", monospace !important;
    font-size: 0.85rem !important;
}
input[type="number"]:hover {
    border-color: rgba(139,92,246,0.4) !important;
    background: rgba(255,255,255,0.08) !important;
}

/* ── Result glass card ── */
.result-glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 24px;
    overflow: hidden;
    position: relative;
    z-index: 1;
}
.result-glow-safe  { box-shadow: 0 0 60px rgba(16,185,129,0.12), inset 0 1px 0 rgba(255,255,255,0.08); }
.result-glow-churn { box-shadow: 0 0 60px rgba(244,63,94,0.12),  inset 0 1px 0 rgba(255,255,255,0.08); }

.result-top-bar {
    height: 3px;
    background: linear-gradient(90deg, var(--purple), var(--cyan));
}
.result-top-bar.danger {
    background: linear-gradient(90deg, var(--red), var(--yellow));
}

.result-body { padding: 2rem; }

.result-main {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1.75rem;
}
.verdict-tag {
    font-size: 0.55rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.verdict-text {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.v-safe  { color: var(--green); filter: drop-shadow(0 0 12px rgba(16,185,129,0.4)); }
.v-churn { color: var(--red);   filter: drop-shadow(0 0 12px rgba(244,63,94,0.4)); }

.prob-block { text-align: right; }
.prob-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.5rem;
    font-weight: 500;
    line-height: 1;
    letter-spacing: -0.02em;
}
.pn-safe  { background: linear-gradient(135deg, var(--green), var(--cyan)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.pn-churn { background: linear-gradient(135deg, var(--red), var(--yellow));  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.prob-sub {
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}

.risk-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.6rem;
    backdrop-filter: blur(8px);
}
.chip-low    { background:rgba(16,185,129,0.12);  color:var(--green);  border:1px solid rgba(16,185,129,0.25); }
.chip-medium { background:rgba(245,158,11,0.12);  color:var(--yellow); border:1px solid rgba(245,158,11,0.25); }
.chip-high   { background:rgba(244,63,94,0.12);   color:var(--red);    border:1px solid rgba(244,63,94,0.25); }

/* ── Gauge ── */
.gauge-wrap { margin-bottom: 1.5rem; }
.gauge-track {
    height: 5px;
    border-radius: 3px;
    background: rgba(255,255,255,0.06);
    position: relative;
    overflow: visible;
    margin: 0.5rem 0 0.4rem;
}
.gauge-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--green), var(--yellow), var(--red));
    background-size: 300% 100%;
}
.gauge-dot {
    position: absolute;
    top: -5px;
    width: 15px; height: 15px;
    background: white;
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 0 0 3px rgba(15,12,40,0.8), 0 0 12px rgba(255,255,255,0.6);
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--muted);
}

/* ── Insight ── */
.insight-box {
    background: rgba(139,92,246,0.07);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    font-size: 0.78rem;
    color: var(--dim);
    line-height: 1.6;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
}
.insight-box.danger {
    background: rgba(244,63,94,0.07);
    border-color: rgba(244,63,94,0.15);
}
.insight-box strong { color: var(--text); }

/* ── Summary ── */
.sum-title {
    font-size: 0.55rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.sum-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
.sum-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.85rem;
    background: rgba(10,6,30,0.6);
    backdrop-filter: blur(4px);
    font-size: 0.75rem;
    transition: background 0.15s;
}
.sum-row:hover { background: rgba(139,92,246,0.06); }
.sk { color: var(--muted); }
.sv { color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; }

/* ── Footer ── */
.g-footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.65rem;
    color: rgba(100,116,139,0.4);
    letter-spacing: 0.1em;
    position: relative;
    z-index: 1;
}
</style>

<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
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

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-pill">AI-Powered Analytics</div>
    <div class="hero-title">Churn Predictor</div>
    <div class="hero-sub">Enter customer details to instantly assess churn risk using a trained neural network.</div>
</div>
""", unsafe_allow_html=True)

# ── Input cards ───────────────────────────────────────────────────────────────
st.markdown('<div class="g-card"><div class="g-card-title">🌍 Demographics</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
with c2: gender = st.selectbox("Gender", ["Male", "Female"])
with c3: age = st.number_input("Age", min_value=18, max_value=92, value=35, step=1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="g-card"><div class="g-card-title">💳 Financials</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4: credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
with c5: balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=500.0, format="%.2f")
with c6: estimated_salary = st.number_input("Est. Salary ($)", min_value=0.0, value=60000.0, step=1000.0, format="%.2f")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="g-card"><div class="g-card-title">🏦 Account</div>', unsafe_allow_html=True)
c7, c8, c9, c10 = st.columns(4)
with c7:  tenure = st.number_input("Tenure (yrs)", min_value=0, max_value=10, value=5, step=1)
with c8:  num_of_products = st.number_input("Products", min_value=1, max_value=4, value=1, step=1)
with c9:  has_cr_card = st.selectbox("Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
with c10: is_active_member = st.selectbox("Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No")
st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [1 if gender == 'Female' else 0],
    'Age': [age], 'Tenure': [tenure], 'Balance': [balance],
    'NumOfProducts': [num_of_products], 'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member], 'EstimatedSalary': [estimated_salary],
})
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)
scaled = scaler.transform(input_data).astype(np.float32)
iname = session.get_inputs()[0].name
prob = float(session.run(None, {iname: scaled})[0][0][0])

will_churn  = prob > 0.5
risk_pct    = int(prob * 100)
risk_label  = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
chip_cls    = "chip-high" if prob > 0.7 else ("chip-medium" if prob > 0.4 else "chip-low")
v_cls       = "v-churn" if will_churn else "v-safe"
pn_cls      = "pn-churn" if will_churn else "pn-safe"
glow_cls    = "result-glow-churn" if will_churn else "result-glow-safe"
bar_cls     = "danger" if will_churn else ""
gauge_cls   = "danger" if prob > 0.7 else ("warn" if prob > 0.4 else "safe")
insight_cls = "danger" if will_churn else ""
verdict     = "⚠️ Likely to Churn" if will_churn else "✅ Likely to Retain"
insight_txt = (
    f"This customer carries a <strong>{prob:.1%} churn probability</strong>. Recommend a proactive outreach — personalised offer or loyalty incentive could reduce risk significantly."
    if will_churn else
    f"This customer has a low <strong>{prob:.1%} churn probability</strong>. Continue standard engagement. Re-evaluate quarterly or if account activity drops."
)
dot_pct = max(1, min(99, risk_pct))

st.markdown(f"""
<div class="result-glass {glow_cls}" style="margin-top:1.25rem;">
    <div class="result-top-bar {bar_cls}"></div>
    <div class="result-body">
        <div class="result-main">
            <div>
                <div class="verdict-tag">Prediction Result</div>
                <div class="verdict-text {v_cls}">{verdict}</div>
                <div><span class="risk-chip {chip_cls}">{risk_label} Risk</span></div>
            </div>
            <div class="prob-block">
                <div class="prob-num {pn_cls}">{prob:.1%}</div>
                <div class="prob-sub">Churn Probability</div>
            </div>
        </div>

        <div class="gauge-wrap">
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{dot_pct}%;"></div>
                <div class="gauge-dot" style="left:{dot_pct}%;"></div>
            </div>
            <div class="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
        </div>

        <div class="insight-box {insight_cls}">{insight_txt}</div>

        <div class="sum-title">Input Summary</div>
        <div class="sum-grid">
            <div class="sum-row"><span class="sk">Geography</span><span class="sv">{geography}</span></div>
            <div class="sum-row"><span class="sk">Gender</span><span class="sv">{gender}</span></div>
            <div class="sum-row"><span class="sk">Age</span><span class="sv">{age}</span></div>
            <div class="sum-row"><span class="sk">Tenure</span><span class="sv">{tenure} yrs</span></div>
            <div class="sum-row"><span class="sk">Credit Score</span><span class="sv">{credit_score:,}</span></div>
            <div class="sum-row"><span class="sk">Balance</span><span class="sv">${balance:,.0f}</span></div>
            <div class="sum-row"><span class="sk">Est. Salary</span><span class="sv">${estimated_salary:,.0f}</span></div>
            <div class="sum-row"><span class="sk">Products</span><span class="sv">{num_of_products}</span></div>
            <div class="sum-row"><span class="sk">Credit Card</span><span class="sv">{'Yes' if has_cr_card else 'No'}</span></div>
            <div class="sum-row"><span class="sk">Active Member</span><span class="sv">{'Yes' if is_active_member else 'No'}</span></div>
        </div>
    </div>
</div>

<div class="g-footer">Powered by ANN · Predictions are probabilistic estimates · Not financial advice</div>
""", unsafe_allow_html=True)
