import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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

# ===== Streamlit App =====
st.title("Student GPA Prediction")

st.write("**Random Forest Model Evaluation**")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.2f}")

# Input từ người dùng
st.subheader("Enter student data:")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, value=float(X[col].mean()))

input_df = pd.DataFrame([input_data])
prediction = rf.predict(input_df)[0]
st.success(f"Predicted GPA (G3): {prediction:.2f}")

# Feature importance
st.subheader("Feature Importance")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(range(X.shape[1]), importances[indices], align="center")
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels(X.columns[indices], rotation=90)
st.pyplot(fig)

