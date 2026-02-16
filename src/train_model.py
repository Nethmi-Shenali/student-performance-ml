import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lime.lime_tabular import LimeTabularExplainer

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load dataset
data = pd.read_csv("data/student_performance.csv")

# Encode target
data["final_result"] = data["final_result"].map({"Fail": 0, "Pass": 1})

# Features and target
X = data.drop("final_result", axis=1)
y = data["final_result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------- FEATURE IMPORTANCE ----------------
importances = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.savefig("outputs/feature_importance.png", bbox_inches="tight")
plt.close()

print("Feature importance plot saved to outputs/feature_importance.png")

# ---------------- LIME EXPLAINABILITY ----------------
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["Fail", "Pass"],
    mode="classification"
)

# Explain the first test instance
exp = explainer.explain_instance(
    data_row=X_test.iloc[0].values,
    predict_fn=model.predict_proba
)

# Save LIME explanation as HTML
lime_path = "outputs/lime_explanation.html"
exp.save_to_file(lime_path)

print(f"LIME explanation saved to {lime_path}")
