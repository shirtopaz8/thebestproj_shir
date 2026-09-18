import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import kagglehub
import os
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# 1. הגדרות דף עיצוב
# ---------------------------------------------------------
st.set_page_config(
    page_title="מחשבון קווידיץ' ומיון להוגוורטס",
    page_icon="🧹",
    layout="wide"
)

# עיצוב מותאם אישית (CSS)
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #740001;
        font-family: 'Georgia', serif;
        font-size: 2.8rem;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #D3A625;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-right: 5px solid #740001;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🔮 חיזוי מיומנות קווידיץ\' ומצנפת המיון</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">מבוסס על מודל Linear Regression מהדאטה-סט של הארי פוטר</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. טעינת הנתונים ואימון המודל (שימוש ב-Cache לביצועים מהירים)
# ---------------------------------------------------------
@st.cache_resource
def load_data_and_train_model():
    # הורדת הדאטה-סט מ-Kaggle
    path = kagglehub.dataset_download("sahityapalacharla/harry-potter-sorting-dataset")
    csv_path = os.path.join(path, "harry_potter_1000_students.csv")
    data = pd.read_csv(csv_path)
    
    # חילוץ feature ו-target לפי המחברת
    X = data["Bravery"].to_numpy().reshape(-1, 1)
    y = data["Quidditch Skills"].to_numpy()
    
    # אימון מודל רגרסיה לינארית
    model = LinearRegression()
    model.fit(X, y)
    
    w = model.coef_[0]
    b = model.intercept_
    
    return data, model, w, b

try:
    with st.spinner("טוען נתונים ומאמן את המודל..."):
        data, model, w, b = load_data_and_train_model()
except Exception as e:
    st.warning("לא ניתן היה להוריד את הדאטה-סט מ-Kaggle באופן אוטומטי. משתמש בערכי המודל המחושבים מהמחברת.")
    # ערכי ברירת מחדל מהמחברת שלכם במידה ואין חיבור ל-Kaggle
    w, b = 0.2644254174662321, 4.130997851814717
    data = pd.DataFrame({
        "Bravery": [9, 6, 1, 9, 5],
        "Quidditch Skills": [8, 6, 4, 9, 6],
        "House": ["Gryffindor", "Ravenclaw", "Hufflepuff", "Gryffindor", "Ravenclaw"]
    })

# ---------------------------------------------------------
# 3. סרגל צד (Sidebar) - קלט המשתמש
# ---------------------------------------------------------
st.sidebar.header("🧙‍♂️ הכנס את התכונות שלך")

user_bravery = st.sidebar.slider("רמת אומץ (Bravery)", min_value=1, max_value=10, value=7, step=1)
user_intelligence = st.sidebar.slider("רמת תבונה (Intelligence)", min_value=1, max_value=10, value=5, step=1)
user_loyalty = st.sidebar.slider("רמת נאמנות (Loyalty)", min_value=1, max_value=10, value=5, step=1)
user_ambition = st.sidebar.slider("רמת שאפתנות (Ambition)", min_value=1, max_value=10, value=5, step=1)

# ---------------------------------------------------------
# 4. חישוב חיזוי קווידיץ' ומיון לבית
# ---------------------------------------------------------
# חיזוי לפי הנוסחה: y = w * x + b
predicted_quidditch = w * user_bravery + b

# חישוב הבית המתאים לפי התכונה הדומיננטית
traits = {
    "גריפינדור 🦁": user_bravery,
    "רייבנקלו 🦅": user_intelligence,
    "האפלפאף 🦡": user_loyalty,
    "סלית'רין 🐍": user_ambition
}
assigned_house = max(traits, key=traits.get)

# ---------------------------------------------------------
# 5. תצוגת התוצאות הראשיות
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("רמת אומץ שנבחרה (X)", f"{user_bravery} / 10")

with col2:
    st.metric("חיזוי מיומנות קווידיץ' (Y)", f"{predicted_quidditch:.2f} / 10")

with col3:
    st.metric("הבית המתאים לך (מצנפת המיון)", assigned_house)

st.divider()

# ---------------------------------------------------------
# 6. גרף אינטראקטיבי של קו הרגרסיה
# ---------------------------------------------------------
st.subheader("📈 קו הרגרסיה הלינארית והתחזית שלך")

# יצירת נקודות לקו הרגרסיה
x_range = np.linspace(1, 10, 100)
y_range = w * x_range + b

fig = go.Figure()

# הוספת קו הרגרסיה
fig.add_trace(go.Scatter(
    x=x_range, 
    y=y_range, 
    mode='lines', 
    name='קו הרגרסיה (Linear Regression)',
    line=dict(color='#740001', width=3)
))

# הוספת הנקודה האישית של המשתמש
fig.add_trace(go.Scatter(
    x=[user_bravery], 
    y=[predicted_quidditch], 
    mode='markers', 
    name='התחזית שלך',
    marker=dict(color='#D3A625', size=14, symbol='star')
))

fig.update_layout(
    xaxis_title="רמת אומץ (Bravery)",
    yaxis_title="מיומנות קווידיץ' (Quidditch Skills)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 7. הסבר על המודל והנתונים מהמחברת
# ---------------------------------------------------------
with st.expander("🔍 איך המודל הזה עובד? (הסבר מהמחברת)"):
    st.write(f"""
    המחשבון משתמש במודל **Linear Regression** שאומן על הדאטה-סט של הוגוורטס.
    
    * **משוואת הישר שנלמדה:** $\\text{{Quidditch Skills}} = {w:.4f} \\times \\text{{Bravery}} + {b:.4f}$
    * **הטיה (Intercept - $b$):** {b:.2f} - רמת מיומנות בסיסית גם ללא אומץ.
    * **משקל (Slope/Weight - $w$):** {w:.2f} - תוספת למיומנות הקווידיץ' על כל יחידת אומץ.
    """)
    
    st.subheader("טבלת נתונים לדוגמה מהדאטה-סט:")
    st.dataframe(data.head(5), use_container_width=True)
