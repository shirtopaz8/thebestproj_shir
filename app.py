import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Quidditch Skill Predictor", 
    page_icon="🧹",
    layout="wide"
)

# Enforce Left-to-Right layout & elegant styling
st.markdown("""
    <style>
    /* 1. Global Reset & Dark Maroon Theme */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1A0003 0%, #38000A 50%, #100003 100%) !important;
        color: #F0E6D2 !important;
        font-family: 'Cinzel', 'Georgia', serif !important;
        direction: ltr !important;
        text-align: left !important;
    }
    
    /* 2. Sidebar Cinematic Styling */
    [data-testid="stSidebar"] {
        background-color: #240006 !important;
        border-right: 1px solid #740001 !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.5);
    }
    
    [data-testid="stSidebar"] * {
        color: #E2C974 !important;
    }

    /* 3. Gold Glowing Titles */
    .main-title {
        color: #FCE181;
        font-family: 'Georgia', serif;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(252, 225, 129, 0.4), 0 0 20px rgba(116, 0, 1, 0.8);
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .sub-title {
        color: #D3A625;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
        opacity: 0.9;
    }

    /* 4. Movie-Card Glassmorphism Design */
    .metric-card {
        background: rgba(40, 0, 8, 0.65);
        border: 1px solid #D3A625;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 15px rgba(211, 166, 37, 0.4);
    }

    /* 5. Custom Number Input Buttons */
    div[data-baseweb="input"] {
        background-color: #38000A !important;
        border: 1px solid #D3A625 !important;
        color: #FFF !important;
        border-radius: 6px;
    }

    button[title="Increase value"], button[title="Decrease value"] {
        background-color: #740001 !important;
        color: #FCE181 !important;
        border: 1px solid #D3A625 !important;
    }

    /* 6. Expanders and Tables */
    .stTable {
        background: rgba(30, 0, 5, 0.8) !important;
        border-radius: 8px;
        border: 1px solid #740001;
    }

    /* Hide Streamlit Default Header/Footer for immersive feel */
    header, footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="main-title">🧹 Quidditch Skill Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predicting student Quidditch Performance ($Y$) based on their Bravery level ($X$) using Linear Regression.</div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# 2. Model Parameters & Loss Metrics (From Notebook)
# ---------------------------------------------------------
# Model Weights (y = w * x + b)
w = 0.2644254174662321
b = 4.130997851814717

# Loss Values from Notebook (MAE / Absolute Error)
baseline_loss = 2.2362
model_loss = 2.1522

# ---------------------------------------------------------
# 3. Sidebar - Input Section (+ / - Buttons)
# ---------------------------------------------------------
st.sidebar.header("🧙 Input Feature")
st.sidebar.write("Adjust the student's Bravery level below:")

x_input = st.sidebar.number_input(
    "Select Bravery Level (X):", 
    min_value=1, 
    max_value=10, 
    value=7, 
    step=1
)

# Prediction Formula: y = w * x + b
y_prediction = w * x_input + b

# ---------------------------------------------------------
# 4. Results Display (Section 5a & 5b of Guidelines)
# ---------------------------------------------------------
st.subheader("🎯 Prediction Output")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Input Bravery Level (X)", f"{x_input} / 10")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Predicted Quidditch Skill (Y)", f"{y_prediction:.2f} / 10")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# 5. Interactive Regression Line Chart
# ---------------------------------------------------------
st.subheader("📈 Linear Regression Line")

x_vals = np.linspace(1, 10, 100)
y_vals = w * x_vals + b

fig = go.Figure()

# Plot Regression Line
fig.add_trace(go.Scatter(
    x=x_vals, 
    y=y_vals, 
    mode='lines', 
    name='Regression Model (y = w*x + b)',
    line=dict(color='#740001', width=3)
))

# Plot User Point
fig.add_trace(go.Scatter(
    x=[x_input], 
    y=[y_prediction], 
    mode='markers', 
    name='Current Prediction Point',
    marker=dict(color='#D3A625', size=14, symbol='star')
))
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(30,0,5,0.5)',
    font=dict(color='#F0E6D2'),
    xaxis=dict(title="Bravery Level (X)", gridcolor='#4A000D', zerolinecolor='#740001'),
    yaxis=dict(title="Quidditch Skill (Y)", gridcolor='#4A000D', zerolinecolor='#740001'),
    margin=dict(l=20, r=20, t=30, b=20)
)


st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 6. Model Training & Evaluation Explanation (Section 5c of Guidelines)
# ---------------------------------------------------------
st.subheader("📊 Model Training & Loss Evaluation")

st.write("""
This machine learning model was trained using a single-feature Linear Regression algorithm on the **Harry Potter Sorting Dataset**.
""")

# Metrics Comparison Table
metrics_df = pd.DataFrame({
    "Model Type": ["Baseline Model (Mean Prediction)", "Trained Linear Regression"],
    "Formula / Method": [f"Always predicts mean y ({5.46:.2f})", f"Y = {w:.4f} * X + {b:.4f}"],
    "Mean Absolute Error (Loss)": [f"{baseline_loss:.4f}", f"{model_loss:.4f}"]
})

st.table(metrics_df)

# Explanation Box
with st.expander("💡 Understanding the Model & Parameters"):
    st.write(f"""
    * **Feature (X):** `Bravery` — Represents the student's courage rating on a scale of 1 to 10.
    * **Target (Y):** `Quidditch Skills` — Represents the predicted flying and playing performance.
    * **Weight ($w = {w:.4f}$):** The positive slope indicates that higher bravery increases predicted Quidditch skill.
    * **Bias ($b = {b:.4f}$):** The baseline intercept when bravery level is zero.
    * **Loss Comparison:** The trained model reduced the overall prediction error from **{baseline_loss:.4f}** down to **{model_loss:.4f}**, demonstrating an improved prediction over predicting a simple average.
    """)
