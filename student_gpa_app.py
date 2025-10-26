import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ===== Streamlit Page Config =====
st.set_page_config(
    page_title="Student GPA Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Sidebar =====
st.sidebar.header("About this App")
st.sidebar.write("Author: Nguyễn Ngọc Minh Anh")
st.sidebar.write("University: Tampere University")
st.sidebar.write("Major: Machine Learning")
st.sidebar.write("GitHub: [student-gpa-app](https://github.com/bethnguyenpawn/student-gpa-app)")
st.sidebar.write("---")

# ===== Header =====
st.title("🎓 Student GPA Prediction App")
st.markdown("""
This app predicts a student's final grade (G3) based on various features like study time, number of past failures, and other personal attributes.
""")
st.write("---")

# ===== Load CSV =====
data = pd.read_csv('student-mat.csv', sep=';')

# ===== Preprocessing =====
data = pd.get_dummies(data, drop_first=True)
X = data.drop("G3", axis=1)
y = data["G3"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== Train model =====
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ===== Evaluate =====
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # sửa lỗi squared
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.info(f"**RMSE:** {rmse:.2f} | **R²:** {r2:.2f}")

# ===== User Input =====
st.subheader("Predict Your GPA")
st.markdown("Enter student data below to predict the final grade (G3):")

input_data = {}
cols1, cols2 = st.columns(2)  # Chia 2 cột cho đẹp
for i, col in enumerate(X.columns):
    if i % 2 == 0:
        input_data[col] = cols1.number_input(col, value=float(X[col].mean()))
    else:
        input_data[col] = cols2.number_input(col, value=float(X[col].mean()))

input_df = pd.DataFrame([input_data])
prediction = rf.predict(input_df)[0]
st.success(f"Predicted GPA (G3): {prediction:.2f}")

# ===== Feature Importance =====
st.subheader("Feature Importance")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(12,6))
ax.bar(range(X.shape[1]), importances[indices], align="center", color="skyblue")
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels(X.columns[indices], rotation=90, fontsize=10)
ax.set_ylabel("Importance", fontsize=12)
ax.set_title("Random Forest Feature Importance", fontsize=14)
st.pyplot(fig)

# ===== Footer =====
st.markdown("---")
st.markdown("© 2025 Nguyễn Ngọc Minh Anh | Tampere University | Machine Learning Major")

