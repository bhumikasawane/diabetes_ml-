# ============================================================
# GlucoPredict - Diabetes Risk Prediction
# Machine Learning Project using Logistic Regression
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import joblib


# -----------------------------
# 2. Load Dataset
# -----------------------------

data = pd.read_csv("diabetes.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

print("\nStatistical Summary:")
print(data.describe())


# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------

# Outcome distribution

plt.figure(figsize=(6, 4))

data["Outcome"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Number of Patients")
plt.xticks(
    [0, 1],
    ["No Diabetes", "Diabetes"],
    rotation=0
)

plt.tight_layout()
plt.show()


# -----------------------------
# 4. Feature Distribution
# -----------------------------

data.hist(
    figsize=(14, 10),
    bins=20
)

plt.suptitle(
    "Feature Distributions",
    fontsize=16
)

plt.tight_layout()
plt.show()


# -----------------------------
# 5. Correlation Matrix
# -----------------------------

correlation = data.corr(numeric_only=True)

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# -----------------------------
# 6. Separate Input and Output
# -----------------------------

X = data.drop(
    "Outcome",
    axis=1
)

y = data["Outcome"]


# -----------------------------
# 7. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:",
      X_train.shape)

print("Testing Data:",
      X_test.shape)


# -----------------------------
# 8. Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# -----------------------------
# 9. Create Logistic Regression
# -----------------------------

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)


# -----------------------------
# 10. Train Model
# -----------------------------

model.fit(
    X_train_scaled,
    y_train
)

print("\nModel training completed.")


# -----------------------------
# 11. Prediction
# -----------------------------

y_pred = model.predict(
    X_test_scaled
)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# -----------------------------
# 12. Model Evaluation
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc * 100:.2f}%"
)


# -----------------------------
# 13. Classification Report
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -----------------------------
# 14. Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Predicted 0", "Predicted 1"]
)

plt.yticks(
    [0, 1],
    ["Actual 0", "Actual 1"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.show()


# -----------------------------
# 15. ROC Curve
# -----------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.tight_layout()
plt.show()


# -----------------------------
# 16. Save Model
# -----------------------------

joblib.dump(
    model,
    "diabetes_model.pkl"
)

joblib.dump(
    scaler,
    "diabetes_scaler.pkl"
)

print("\nModel saved as:")
print("diabetes_model.pkl")

print("\nScaler saved as:")
print("diabetes_scaler.pkl")


# ============================================================
# 17. User Input Prediction
# ============================================================

print("\n===================================")
print("     DIABETES RISK PREDICTION")
print("===================================")

print("\nEnter patient details:")

pregnancies = float(
    input("Number of Pregnancies: ")
)

glucose = float(
    input("Glucose Level: ")
)

blood_pressure = float(
    input("Blood Pressure: ")
)

skin_thickness = float(
    input("Skin Thickness: ")
)

insulin = float(
    input("Insulin Level: ")
)

bmi = float(
    input("BMI: ")
)

diabetes_pedigree = float(
    input("Diabetes Pedigree Function: ")
)

age = float(
    input("Age: ")
)


# -----------------------------
# 18. Create Input DataFrame
# -----------------------------

new_patient = pd.DataFrame(
    [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]],
    columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]
)


# -----------------------------
# 19. Scale User Input
# -----------------------------

new_patient_scaled = scaler.transform(
    new_patient
)


# -----------------------------
# 20. Make Prediction
# -----------------------------

prediction = model.predict(
    new_patient_scaled
)

probability = model.predict_proba(
    new_patient_scaled
)[0][1]


# -----------------------------
# 21. Display Result
# -----------------------------

print("\n===================================")
print("          PREDICTION RESULT")
print("===================================")

if prediction[0] == 1:

    print(
        "Prediction: Higher predicted risk"
    )

else:

    print(
        "Prediction: Lower predicted risk"
    )

print(
    f"Diabetes Probability: "
    f"{probability * 100:.2f}%"
)

print("\nNote:")
print(
    "This project is for educational purposes "
    "and is not a medical diagnosis."
)