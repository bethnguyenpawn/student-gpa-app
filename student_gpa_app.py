import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# ===== Streamlit Config =====
st.set_page_config(page_title="Student GPA Prediction", layout="wide", page_icon="🎓")

# ===== Sidebar Theme Toggle =====
st.sidebar.title("Settings")
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"])
bg_color = "#1E1E1E" if theme_choice=="Dark" else "#F5F5F5"
text_color = "#FFFFFF" if theme_choice=="Dark" else "#000000"

# ===== Google Font & CSS =====
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
body {{ font-family: 'Poppins', sans-serif; background-color:{bg_color}; color:{text_color}; }}
.stButton>button {{
    background: linear-gradient(90deg,#4caf50,#81c784);
    color:white; font-weight:bold; font-size:16px; height:45px; border-radius:8px;
}}
.stButton>button:hover {{ background: linear-gradient(90deg,#43a047,#66bb6a); }}
.input-container {{ background-color: {'#2c2c2c' if theme_choice=='Dark' else '#ffffff'};
                    padding:15px; border-radius:10px; margin-bottom:15px; box-shadow:0 4px 6px rgba(0,0,0,0.1); }}
</style>
""", unsafe_allow_html=True)

# ===== Header with Logo =====
logo_path = Path("Tampere_University_logo.png")
st.image(str(logo_path), width=120)
st.markdown(f"""
<div style="display:inline-block; margin-left:20px;">
<h1 style="margin-bottom:5px; color:{text_color};">Student GPA Prediction App 🎓</h1>
<p style="margin:0; font-size:14px; color:{text_color};">
Author: Nguyễn Ngọc Minh Anh | Tampere University | Major: Machine Learning
</p>
</div>
<hr style="border:1px solid {text_color};">
""", unsafe_allow_html=True)

# ===== Load Data =====
@st.cache_data
def load_data():
    return pd.read_csv('student-mat.csv', sep=';')

data = load_data()
data = pd.get_dummies(data, drop_first=True)
X = data.drop("G3", axis=1)
y = data["G3"]

# ===== Mapping Column Names =====
column_labels = {
    "school_MS": "School: MS",
    "sex_M": "Gender: Male",
    "age": "Age",
    "Medu": "Mother's Education",
    "Fedu": "Father's Education",
    "traveltime": "Travel Time to School",
    "studytime": "Weekly Study Time",
    "failures": "Past Failures",
    "schoolsup_yes": "Extra Educational Support",
    "famsup_yes": "Family Educational Support",
    "paid_yes": "Extra Paid Classes",
    "activities_yes": "Extracurricular Activities",
    "nursery_yes": "Attended Nursery",
    "higher_yes": "Wants Higher Education",
    "internet_yes": "Has Internet Access",
    "romantic_yes": "Has Romantic Relationship",
    "famrel": "Family Relationship Quality",
    "freetime": "Free Time",
    "goout": "Going Out Frequency",
    "Dalc": "Workday Alcohol Consumption",
    "Walc": "Weekend Alcohol Consumption",
    "health": "Current Health Status",
    "absences": "School Absences",
    "G1": "Grade 1",
    "G2": "Grade 2"
}

# ===== Sidebar Model Parameters =====
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)
n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100)

# ===== Tabbed Input Layout =====
tabs = st.tabs(["Basic Info", "Study & Social", "Grades"])
input_data = {}

# Tab 1: Basic Info
with tabs[0]:
    with st.container():
        for col in ["age","sex_M","school_MS","nursery_yes","higher_yes"]:
            label = column_labels.get(col,col)
            if X[col].nunique() <= 20:
                input_data[col] = st.selectbox(label, sorted(X[col].unique()))
            else:
                input_data[col] = st.number_input(label, value=float(X[col].mean()))

# Tab 2: Study & Social
with tabs[1]:
    with st.container():
        for col in ["Medu","Fedu","studytime","traveltime","failures","schoolsup_yes",
                    "famsup_yes","paid_yes","activities_yes","internet_yes","romantic_yes",
                    "famrel","freetime","goout","Dalc","Walc","health","absences"]:
            label = column_labels.get(col,col)
            if X[col].nunique() <= 20:
                input_data[col] = st.selectbox(label, sorted(X[col].unique()))
            else:
                input_data[col] = st.number_input(label, value=float(X[col].mean()))

# Tab 3: Grades
with tabs[2]:
    for col in ["G1","G2"]:
        label = column_labels.get(col,col)
        input_data[col] = st.number_input(label, value=float(X[col].mean()))

# ===== Convert input_data to DataFrame & fix missing columns =====
input_df = pd.DataFrame([input_data])

# Ensure all columns exist as in training data
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Arrange columns in same order as X_train
input_df = input_df[X.columns]

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

# ===== Predict Button =====
st.subheader("Prediction")
predict_btn = st.button("🎯 Predict GPA")
if predict_btn:
    prediction = rf.predict(input_df)[0]
    st.markdown(f'<div class="input-container">', unsafe_allow_html=True)
    st.success(f"Predicted GPA (G3): {prediction:.2f}")
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== Feature Importance Interactive =====
st.subheader("Feature Importance")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig = px.bar(x=importances[indices], y=[column_labels.get(c,c) for c in X.columns[indices]],
             orientation='h', color=importances[indices], color_continuous_scale='Viridis',
             labels={'x':'Importance','y':'Feature'})
st.plotly_chart(fig, use_container_width=True)

# ===== Show Raw Dataset =====
with st.expander("Show Raw Dataset"):
    st.dataframe(data)


