import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page layout setup
st.set_page_config(page_title="Quidditch Skill Predictor", page_icon="🧹")

# Force full Left-to-Right layout to fix inverted slider behavior
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        direction: ltr !important;
        text-align: left !important;
    }
    div[data-baseweb="slider"] {
        direction: ltr !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧹 Quidditch Skill Predictor")
st.write("A simple 2-variable Linear Regression model predicting Quidditch Skills based on Bravery.")

# Model parameters from notebook (y = w * x + b)
w = 0.2644254174662321
b = 4.130997851814717

# User Input (Single Feature X)
st.sidebar.header("User Input")
x_input = st.sidebar.slider("Select Bravery Level (X):", min_value=1, max_value=10, value=7, step=1)

# Linear Regression Formula (Single Target Y)
y_prediction = w * x_input + b

# Display Results
st.subheader("Prediction Result")
col1, col2 = st.columns(2)

with col1:
    st.metric("Bravery Level (X)", f"{x_input}")

with col2:
    st.metric("Predicted Quidditch Skill (Y)", f"{y_prediction:.2f}")

st.divider()

# Plotting the Linear Regression Line
st.subheader("Linear Regression Visualization")

x_vals = np.linspace(1, 10, 100)
y_vals = w * x_vals + b

fig = go.Figure()

# Regression Line
fig.add_trace(go.Scatter(
    x=x_vals, 
    y=y_vals, 
    mode='lines', 
    name='Regression Line (y = w*x + b)',
    line=dict(color='red', width=2)
))

# User Data Point
fig.add_trace(go.Scatter(
    x=[x_input], 
    y=[y_prediction], 
    mode='markers', 
    name='Your Prediction Point',
    marker=dict(color='gold', size=12, symbol='star')
))

fig.update_layout(
    xaxis_title="Bravery (X)",
    yaxis_title="Quidditch Skills (Y)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Explanation Section
with st.expander("Model Formula Details"):
    st.write(f"**Formula:** $Y = {w:.4f} \\cdot X + {b:.4f}$")
    st.write(f"* **Slope (w):** {w:.4f}")
    st.write(f"* **Intercept (b):** {b:.4f}")
