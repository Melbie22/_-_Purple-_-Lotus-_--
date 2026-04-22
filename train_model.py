# train_model.py - Run this to train and save your model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the Pima Indians Diabetes Dataset
print("Loading data...")
df = pd.read_csv('diabetes.csv')  # Use the original dataset

# Replace zeros with NaN for biologically implausible features
print("Handling missing values...")
features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in features_with_zeros:
    df[feature] = df[feature].replace(0, np.nan)

# Drop rows with missing values
df = df.dropna()
print(f"Dataset shape after cleaning: {df.shape}")

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Features: {list(X.columns)}")
print(f"Diabetes prevalence: {y.mean()*100:.2f}%")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train the model (NO SCALER - using raw data)
print("Training Gradient Boosting model...")
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Test with a low-risk sample
low_risk_sample = [[0, 80, 70, 20, 30, 22, 0.2, 25]]  # Pregnancies, Glucose, BP, Skin, Insulin, BMI, DPF, Age
low_risk_pred = model.predict(low_risk_sample)
print(f"\nTest with low-risk values: {low_risk_sample}")
print(f"Prediction: {low_risk_pred[0]} (should be 0 for low risk)")

# Test with a high-risk sample
high_risk_sample = [[5, 150, 90, 40, 200, 35, 1.5, 55]]
high_risk_pred = model.predict(high_risk_sample)
print(f"\nTest with high-risk values: {high_risk_sample}")
print(f"Prediction: {high_risk_pred[0]} (should be 1 for high risk)")

# Saving the model
joblib.dump(model, 'diabetes_model.pkl')
print("\n✅ Model saved as 'diabetes_model.pkl'")
