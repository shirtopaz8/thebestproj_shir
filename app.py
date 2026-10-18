import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Quidditch Skill Predictor & Game", 
    page_icon="🧹",
    layout="wide"
)

st.markdown("""
    <style>
    /* 1. Global Reset & Dark Maroon Theme */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1A0003 0%, #38000A 50%, #100003 100%) !important;
        color: #FCE181 !important;
        font-family: 'Georgia', serif !important;
        direction: ltr !important;
        text-align: left !important;
    }
    
    /* 2. Sidebar & Labels Styling */
    [data-testid="stSidebar"] {
        background-color: #240006 !important;
        border-right: 1px solid #740001 !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.5);
    }
    
    [data-testid="stSidebar"] *, label, .stWidgetLabel, p, h1, h2, h3, h4 {
        color: #FCE181 !important;
    }

    /* 3. Gold Glowing Titles */
    .main-title {
        color: #FCE181 !important;
        font-family: 'Georgia', serif;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(252, 225, 129, 0.4), 0 0 20px rgba(116, 0, 1, 0.8);
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .sub-title {
        color: #D3A625 !important;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 25px;
        font-style: italic;
    }

    /* 4. Metric Cards & Game Boxes */
    .metric-card {
        background: rgba(40, 0, 8, 0.65);
        border: 1px solid #D3A625;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        text-align: center;
        margin-bottom: 15px;
    }
    
    .metric-card *, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #FCE181 !important;
    }

    /* 5. Custom Inputs & Buttons */
    div[data-baseweb="input"] {
        background-color: #38000A !important;
        border: 1px solid #D3A625 !important;
        border-radius: 6px;
    }

    div[data-baseweb="input"] input {
        color: #FCE181 !important;
    }

    button[title="Increase value"], button[title="Decrease value"] {
        background-color: #740001 !important;
        color: #FCE181 !important;
        border: 1px solid #D3A625 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #740001 0%, #4A000D 100%) !important;
        color: #FCE181 !important;
        border: 1px solid #D3A625 !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #D3A625 0%, #740001 100%) !important;
        color: #1A0003 !important;
        box-shadow: 0 0 10px #D3A625;
    }

    /* 6. Tabs & Tables Styling */
    .stTable, div[data-testid="stExpander"] * {
        color: #FCE181 !important;
    }
    
    button[data-baseweb="tab"] {
        color: #D3A625 !important;
        font-weight: bold;
    }

    /* Hide Streamlit Header/Footer */
    header, footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="main-title">🧹 Quidditch Skill Predictor v2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predicting Quidditch Performance using 2 Features (Bravery & Dark Arts Knowledge) & Game Mode</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Model Parameters & Dataset Samples
# ---------------------------------------------------------
w1 = 0.36043519346829017      # Weight for Bravery
w2 = 0.35451012143880983      # Weight for Dark Arts Knowledge
b = 2.216231827015582         # Bias

# Loss Values
baseline_pred = 5.46
loss_baseline = 2.2362399999999996
loss_1feature = 2.152228291283019
loss_2features = 1.991763460854045

# 5 Dataset Samples for Game
GAME_DATA = [
    {"student_id": 0, "blood_status": "Half-blood", "house": "Gryffindor", "bravery": 9, "dark_arts": 0, "actual_y": 8},
    {"student_id": 1, "blood_status": "Muggle-born", "house": "Ravenclaw", "bravery": 6, "dark_arts": 5, "actual_y": 8},
    {"student_id": 2, "blood_status": "Pure-blood", "house": "Hufflepuff", "bravery": 1, "dark_arts": 1, "actual_y": 4},
    {"student_id": 3, "blood_status": "Pure-blood", "house": "Gryffindor", "bravery": 9, "dark_arts": 1, "actual_y": 10},
    {"student_id": 4, "blood_status": "Muggle-born", "house": "Ravenclaw", "bravery": 5, "dark_arts": 3, "actual_y": 7}
]

# ---------------------------------------------------------
# 3. Main Navigation (Tabs)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Model Predictor & Analysis", "🎮 Beat the Model Game"])

# =========================================================
# TAB 1: PREDICTOR & MODEL ANALYSIS
# =========================================================
with tab1:
    st.sidebar.header("🧙 Input Features")
    st.sidebar.write("Adjust the student's attributes below:")

    x1_input = st.sidebar.number_input(
        "Bravery Level (X1):", 
        min_value=0, max_value=10, value=7, step=1
    )

    x2_input = st.sidebar.number_input(
        "Dark Arts Knowledge (X2):", 
        min_value=0, max_value=10, value=4, step=1
    )

    # Prediction Formula: Y = w1*X1 + w2*X2 + b
    y_pred = w1 * x1_input + w2 * x2_input + b

    st.subheader("🎯 Prediction Output")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Bravery (X1)", f"{x1_input} / 10")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dark Arts Knowledge (X2)", f"{x2_input} / 10")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Quidditch Skill (Y)", f"{y_pred:.2f} / 10")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Model Evaluation Section
    st.subheader("📊 3-Level Loss Comparison & Training Evaluation")
    
    st.write("Comparing prediction error across model iterations:")

    metrics_df = pd.DataFrame({
        "Model Level": [
            "1. Baseline Model (Mean Predictor)", 
            "2. Single-Feature Model (Project 1)", 
            "3. Two-Feature Model (Project 2)"
        ],
        "Features Used": [
            "None (Always predicts mean Y)", 
            "Bravery (X1)", 
            "Bravery (X1) + Dark Arts Knowledge (X2)"
        ],
        "Formula": [
            f"Y = {baseline_pred:.2f}",
            "Y = 0.2644 * X1 + 4.1310",
            f"Y = {w1:.4f} * X1 + {w2:.4f} * X2 + {b:.4f}"
        ],
        "Mean Absolute Error (Loss)": [
            f"{loss_baseline:.4f}",
            f"{loss_1feature:.4f}",
            f"{loss_2features:.4f}"
        ]
    })

    st.table(metrics_df)

    # Explanation Box
    with st.expander("💡 Understanding Weights, Parameters & Loss Reduction"):
        st.write(f"""
        * **Weight 1 ($w_1 = {w1:.4f}$):** Impact of **Bravery**. Every 1-point increase adds ~{w1:.2f} points to Quidditch Skill.
        * **Weight 2 ($w_2 = {w2:.4f}$):** Impact of **Dark Arts Knowledge**. Every 1-point increase adds ~{w2:.2f} points to Quidditch Skill.
        * **Bias ($b = {b:.4f}$):** Base performance level for a student with 0 Bravery and 0 Dark Arts Knowledge.
        * **Why Loss Reduced:** Adding $X_2$ reduced Loss from **{loss_1feature:.4f}** to **{loss_2features:.4f}**. Knowing a student's familiarity with Dark Arts provides additional context about their tactical aggression and adaptability on the broom, complementing their bravery.
        """)

# =========================================================
# TAB 2: BEAT THE MODEL GAME
# =========================================================
with tab2:
    st.subheader("🎮 Game Mode: Can You Beat the AI Model?")

    # Initialize Session State Variables
    if "game_state" not in st.session_state:
        st.session_state.game_state = "rules"  # States: 'rules', 'playing', 'finished'
        st.session_state.current_round = 0
        st.session_state.user_score = 0
        st.session_state.model_score = 0
        st.session_state.history = []

    # Reset Function
    def reset_game():
        st.session_state.game_state = "playing"
        st.session_state.current_round = 0
        st.session_state.user_score = 0
        st.session_state.model_score = 0
        st.session_state.history = []

    # 1. RULES SCREEN
    if st.session_state.game_state == "rules":
        st.markdown("""
        ### 📜 Rules of the Game:
        1. You will play **5 rounds**, each featuring a real student from the Hogwarts dataset.
        2. In each round, you will see the student's **Bravery** and **Dark Arts Knowledge**.
        3. **Your Goal:** Guess the student's actual **Quidditch Skill (1-10)**.
        4. The Linear Regression Model will also generate its prediction for each round.
        5. **Scoring:** Whichever predictor (You or the Model) is **closer to the actual value** gets **10 points**! (If equal, both get 5 points).
        """)
        if st.button("🚀 Start Game"):
            reset_game()
            st.rerun()

    # 2. PLAYING SCREEN
    elif st.session_state.game_state == "playing":
        round_num = st.session_state.current_round
        
        if round_num < len(GAME_DATA):
            student = GAME_DATA[round_num]
            
            st.markdown(f"### 📍 Round {round_num + 1} of {len(GAME_DATA)}")
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.info(f"🏠 **House:** {student['house']}")
                st.info(f"🩸 **Blood Status:** {student['blood_status']}")
            with col_info2:
                st.metric("Bravery Level (X1)", f"{student['bravery']} / 10")
            with col_info3:
                st.metric("Dark Arts Knowledge (X2)", f"{student['dark_arts']} / 10")

            # User Guess Form
            with st.form(key=f"guess_form_{round_num}"):
                user_guess = st.slider("Guess Student's Quidditch Skill (Y):", 1.0, 10.0, 5.0, 0.5)
                submit_button = st.form_submit_button("Submit Guess 🎯")

            if submit_button:
                actual_y = student["actual_y"]
                model_pred = w1 * student["bravery"] + w2 * student["dark_arts"] + b

                user_error = abs(actual_y - user_guess)
                model_error = abs(actual_y - model_pred)

                # Determine Winner of Round
                if user_error < model_error:
                    winner = "User"
                    st.session_state.user_score += 10
                    msg = "🎉 Great job! You were closer than the model!"
                elif model_error < user_error:
                    winner = "Model"
                    st.session_state.model_score += 10
                    msg = "🤖 The AI Model won this round!"
                else:
                    winner = "Tie"
                    st.session_state.user_score += 5
                    msg = "🤝 It's a tie!"

                st.session_state.history.append({
                    "Round": round_num + 1,
                    "House": student["house"],
                    "X1 (Bravery)": student["bravery"],
                    "X2 (Dark Arts)": student["dark_arts"],
                    "Actual Y": actual_y,
                    "Your Guess": user_guess,
                    "Model Pred": round(model_pred, 2),
                    "Winner": winner
                })

                st.write(msg)
                st.info(f"**Actual Skill:** {actual_y} | **Your Error:** {user_error:.2f} | **Model Error:** {model_error:.2f}")

                st.session_state.current_round += 1
                if st.session_state.current_round >= len(GAME_DATA):
                    st.session_state.game_state = "finished"
                
                st.button("Next Round ➡️")

    # 3. FINISHED / SUMMARY SCREEN
    elif st.session_state.game_state == "finished":
        st.markdown("## 🏆 Game Over - Final Results")

        u_score = st.session_state.user_score
        m_score = st.session_state.model_score

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Your Final Score", f"{u_score} pts")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_res2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Final Score", f"{m_score} pts")
            st.markdown('</div>', unsafe_allow_html=True)

        if u_score > m_score:
            st.balloons()
            st.success("🎉 Congratulations! You defeated the Machine Learning Model!")
        elif m_score > u_score:
            st.error("🤖 The AI Model won! Practice making closer estimates and try again.")
        else:
            st.warning("🤝 It's a tie match!")

        st.subheader("📋 Rounds Summary")
        st.table(pd.DataFrame(st.session_state.history))

        if st.button("🔄 Play Again"):
            reset_game()
            st.rerun()
