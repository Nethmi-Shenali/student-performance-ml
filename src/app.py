import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction Dashboard")
st.markdown("Predict whether a student will **PASS or FAIL** using Machine Learning.")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("data/student_performance.csv")
data["final_result"] = data["final_result"].map({"Fail": 0, "Pass": 1})

X = data.drop("final_result", axis=1)
y = data["final_result"]

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------- USER INPUT ----------------
st.sidebar.header("Enter Student Details")

study_hours = st.sidebar.slider("Study Hours", 0, 10, 5)
sleep_hours = st.sidebar.slider("Sleep Hours", 0, 10, 7)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
internet_hours = st.sidebar.slider("Internet Usage Hours", 0, 10, 4)
tuition = st.sidebar.selectbox("Tuition Classes", ["No", "Yes"])
stress = st.sidebar.slider("Stress Level", 1, 10, 5)

tuition_value = 1 if tuition == "Yes" else 0

input_df = pd.DataFrame(
    [[study_hours, sleep_hours, attendance, internet_hours, tuition_value, stress]],
    columns=X.columns
)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Result"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    col1, col2 = st.columns(2)

    # -------- RESULT DISPLAY --------
    with col1:
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(f"✅ PASS (Confidence: {probability:.2f})")
        else:
            st.error(f"❌ FAIL (Confidence: {probability:.2f})")

    # -------- FEATURE IMPORTANCE --------
    with col2:
        st.subheader("Feature Importance")

        importances = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(X.columns, importances)
        ax.set_xlabel("Importance")
        ax.set_title("Model Feature Importance")
        st.pyplot(fig)

    # -------- LIME EXPLANATION --------
    st.subheader("Why did the model predict this? (LIME Explanation)")

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=["Fail", "Pass"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        data_row=input_df.iloc[0].values,
        predict_fn=model.predict_proba
    )

    st.components.v1.html(exp.as_html(), height=400, scrolling=True)
