import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===== Streamlit Page Config =====
st.set_page_config(page_title="Student GPA Prediction", layout="wide", page_icon="🎓")

# ===== Header with Logo and Author Info =====
st.markdown(
    """
    <div style="display:flex; align-items:center;">
        <img src="https://upload.wikimedia.org/wikipedia/en/8/82/Tampere_University_logo.svg" width="120">
        <div style="margin-left:20px;">
            <h1>Student GPA Prediction App 🎓</h1>
            <p style="margin:0; font-size:14px;">
                Author: Nguyễn Ngọc Minh Anh | Tampere University | Major: Machine Learning
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True
)
st.markdown("---")

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
cols = st.sidebar.columns(2)  # 2 columns in sidebar
for i, col_name in enumerate(X.columns):
    # Use slider for numeric columns if range is reasonable
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

# ===== Predict Button =====
st.subheader("Prediction")
if st.button("Predict GPA"):
    prediction = rf.predict(input_df)[0]
    st.success(f"🎯 Predicted GPA (G3): {prediction:.2f}")
    
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")

# ===== Feature Importance =====
st.subheader("Feature Importance")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices], ax=ax, palette="viridis")
ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.pyplot(fig)

# ===== Optional: Show Dataset =====
with st.expander("Show raw dataset"):
    st.dataframe(data)
