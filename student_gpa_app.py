import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

===== PAGE CONFIG =====

st.set_page_config(page_title="🎓 Student GPA Predictor", page_icon="📊", layout="wide")

===== CUSTOM CSS =====

st.markdown("""
<style>
body {
background: linear-gradient(135deg, #f5f9ff 0%, #e8f0ff 100%);
font-family: 'Inter', sans-serif;
}
.main-title {
text-align: center;
color: #003DA5;
font-size: 46px;
font-weight: 800;
margin-bottom: 0;
}
.subtitle {
text-align: center;
color: #444;
font-size: 18px;
margin-top: 5px;
margin-bottom: 25px;
}
.section-title {
color: #003DA5;
font-size: 22px;
font-weight: 700;
margin-top: 40px;
}
.footer {
text-align: center;
color: #666;
font-size: 14px;
margin-top: 60px;
}
.stButton>button {
background-color: #003DA5;
color: white;
font-weight: 600;
border-radius: 10px;
padding: 0.7em 2em;
font-size: 16px;
transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
background-color: #002E7A;
transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

===== HEADER =====

st.image("https://www.tuni.fi/themes/custom/tuni/logo.svg
", width=200)
st.markdown('<h1 class="main-title">🎓 Student GPA Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A Machine Learning App by <b>Nguyễn Ngọc Minh Anh</b> – Tampere University</p>', unsafe_allow_html=True)
st.markdown("---")

===== LOAD DATA =====

data = pd.read_csv('student-mat.csv', sep=';')

===== PREPROCESS =====

data = pd.get_dummies(data, drop_first=True)
X = data.drop("G3", axis=1)
y = data["G3"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

===== TRAIN MODEL =====

rf = RandomForestRegressor(n_estimators=120, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

===== METRICS =====

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
with col1:
st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
with col2:
st.metric(label="R² Score", value=f"{r2:.2f}")

===== USER INPUT =====

st.markdown('<p class="section-title">📋 Enter Student Data</p>', unsafe_allow_html=True)
selected_features = ["G1", "G2", "studytime", "failures", "absences", "age", "Medu", "Fedu"]
input_data = {}

col1, col2 = st.columns(2)
for i, col in enumerate(selected_features):
if i % 2 == 0:
with col1:
input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))
else:
with col2:
input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

===== Predict Button =====

predict_clicked = st.button("🔘 Predict GPA")

if predict_clicked:
for col in X.columns:
if col not in selected_features:
input_data[col] = float(X[col].mean())
input_df = pd.DataFrame([input_data])
prediction = rf.predict(input_df)[0]
gpa_4scale = (prediction / 20) * 4
st.success(f"🎯 Predicted Final Grade: {prediction:.2f}/20 (≈ {gpa_4scale:.2f}/4.0 GPA)")

# ===== Feature Importance =====
st.markdown('<p class="section-title">📈 Feature Importance</p>', unsafe_allow_html=True)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(indices)), importances[indices], color="#003DA5", alpha=0.8)
ax.set_xticks(range(len(indices)))
ax.set_xticklabels(X.columns[indices], rotation=90)
ax.set_ylabel("Importance", fontsize=12)
ax.set_title("Top Factors Influencing Final GPA", fontsize=14, color="#003DA5")
st.pyplot(fig)

===== FOOTER =====

st.markdown("""
<div class="footer">
<hr>
<p><strong>Author:</strong> Nguyễn Ngọc Minh Anh</p>
<p>Tampere University – Machine Learning Major</p>
<p>Dataset: <a href="https://archive.ics.uci.edu/ml/datasets/student+performance" target="_blank">UCI Student Performance Dataset</a></p>
</div>
""", unsafe_allow_html=True)