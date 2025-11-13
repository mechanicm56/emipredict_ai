# 🏦 Financial Risk Assessment & EMI Prediction Platform

### 🔍 Dual Machine Learning Solution: Classification (EMI Eligibility) & Regression (Maximum EMI Prediction)

---

## 📘 Overview

This project builds a **financial risk assessment platform** that integrates machine learning models with MLflow experiment tracking to create an interactive **Streamlit web application** for EMI prediction.  

It helps users and financial institutions evaluate **EMI eligibility** and estimate the **maximum EMI amount** based on income, expenses, and financial behavior — promoting better loan decisions and risk assessment.

---

## 🎯 Objectives

- Perform **classification** for EMI eligibility (`eligible`, `not_eligible`, `high_risk`)
- Perform **regression** for predicting maximum EMI affordability
- Implement **data preprocessing**, **feature engineering**, and **model training**
- Track experiments and model versions using **MLflow**
- Build a **Streamlit app** for real-time EMI prediction

---

## 🧩 Key Features

- Dual ML Models:  
  - **Classification:** Predict EMI eligibility  
  - **Regression:** Predict maximum EMI amount  
- Clean and scalable **data preprocessing pipeline**
- **Feature engineering** using disposable income
- **Model versioning** and tracking with MLflow
- **Interactive Streamlit app** for user input and visualization

---

## ⚙️ Data Processing & Feature Engineering

### Data Cleaning
- Handled missing values with `fillna(0)`
- Cleaned object columns (like salary) by removing currency symbols
- Encoded categorical labels numerically

### Engineered Features
```python
df['total_expenses'] = (
    df['monthly_rent'] +
    df['groceries_utilities'] +
    df['travel_expenses'] +
    df['other_monthly_expenses'] +
    df['current_emi_amount']
)

df['disposable_income'] = df['monthly_salary'] - df['total_expenses']
df['max_monthly_emi'] = (df['disposable_income'] * 0.35).clip(lower=0)
```

This ensures a realistic cap on EMI affordability (~35% of disposable income).

## Model Development

### Classification

Model: RandomForestClassifier

Goal: Predict EMI eligibility

Metrics: Accuracy, Precision, Recall, F1

### Regression

Model: RandomForestRegressor

Goal: Predict max EMI amount

Metrics: R², MAE, RMSE

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, r2_score, mean_absolute_error
import joblib

# Features
X = df[['monthly_salary', 'credit_score', 'years_of_employment', 
        'monthly_rent', 'current_emi_amount', 'groceries_utilities',
        'travel_expenses', 'other_monthly_expenses', 'emergency_fund', 'bank_balance']]

y_class = df['emi_eligibility']
y_reg = df['max_monthly_emi']

# Split and scale
X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train_c)

reg = RandomForestRegressor(random_state=42)
reg.fit(X_train_scaled, y_train_r)

# Save
joblib.dump(clf, "emi_classifier_model.pkl")
joblib.dump(reg, "emi_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

