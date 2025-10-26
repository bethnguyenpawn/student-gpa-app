import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# ===== Streamlit Page Config =====
st.set_page_config(page_title="Student GPA Prediction", layout="wide", page_icon="🎓")

# ===== Sidebar for Theme Toggle =====
st.sidebar.title("Settings")
theme_choice = st.sidebar.radio("Select Theme", ("Light", "Dark"))

# ===== Custom CSS for Theme and Styling =====
if theme_choice == "Dark":
    bg_color = "#1E1E1E"
    text_color = "#FFFFFF"
    button_bg = "linear-gradient(90deg, #4caf50 0%, #81c784 100%)"
else:
    bg_color = "#F5F5F5"
    text_color = "#000000"
    button_bg = "linear-gradient(90deg, #4CAF50 0%, #81C784 100%)"

st.markdown(
    f"""
    <style>
        body {{ background-color: {bg_color}; color: {text_color}; }}
        .stButton>button {{
            background: {button_bg};
            color: white;
            font-size:16px;
            height:45px;
            width:100%;
            border-radius:8px;
            font-weight:bold;
        }}
        .stMetric {{ padding:10px; }}
        .input-container {{
            background-color: {'#2c2c2c' if theme_choice=='Dark' else '#ffffff'};
            padding:15px;
            border-radius:10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom:15px;
        }}
    </style>
    """, unsafe_allow_html=True
)

# ===== Header with Local Logo and Author Info =====
logo_path = Path("Tampere_University_logo.png")  # Logo nằm cùng folder với app.py
st.image(str(logo_path), width=120)

st.markdown(
    f"""
    <div style="display:inline-block; margin-left:20px;">
        <h1 style="margin-bottom:5px; display:inline-block; color:{text_color};">Student GPA Prediction App 🎓</h1>
        <p style="margin:0; font-size:14px; color:{text_color};">
            Author: Nguyễn Ngọc Minh Anh | Tampere University | Major: Machine Learning
        </p>
    </div>
    <hr style="border:1px solid {text_color};">
    """, unsafe_allow_html=True
)

# ===== Load CSV =====
@st.cache_data
def load_data():
    return pd.read_csv('student-mat.csv', sep=';')

data = load_data()

# ===== Preprocessing =====
data = pd.get_dummies(data, drop_first=True)
X = data.drop("G3", axis=1)
y = data["G3"]

# ===== Sidebar for Model Params & Input =====
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)
n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100)

st.sidebar.header("Student Input Data")
input_data = {}
cols = st.sidebar.columns(2)
for i, col_name in enumerate(X.columns):
    if X[col_name].nunique() <= 20:
        input_data[col_name] = cols[i % 2].slider(
            col_name, int(X[col_name].min()), int(X[col_name].max()), int(X[col_name].mean())
        )
    else:
        input_data[col_name] = cols[i % 2].number_input(
            col_name, value=float(X[col_name].mean())
        )

input_df = pd.DataFrame([input_data])

# ===== Train Model =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ===== Evaluate =====
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ===== Predict Button & Metrics =====
st.subheader("Prediction")
predict_btn = st.button("🎯 Predict GPA")

if predict_btn:
    prediction = rf.predict(input_df)[0]

    # Metrics in container
    with st.container():
        st.markdown(f'<div class="input-container">', unsafe_allow_html=True)
        st.success(f"Predicted GPA (G3): {prediction:.2f}")
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("R² Score", f"{r2:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

# ===== Feature Importance =====
st.subheader("Feature Importance")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices], ax=ax, palette="magma")
ax.set_title("Random Forest Feature Importance", color=text_color)
ax.set_xlabel("Importance", color=text_color)
ax.set_ylabel("Feature", color=text_color)
ax.tick_params(colors=text_color)
st.pyplot(fig)

# ===== Show Raw Dataset =====
with st.expander("Show Raw Dataset"):
    st.dataframe(data)
