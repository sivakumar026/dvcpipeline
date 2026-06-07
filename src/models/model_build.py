import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load processed data
df = pd.read_csv("data/processed/train_processed.csv")

# Target column
target_col = "Loan_Status"

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Validation accuracy
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {accuracy:.4f}")

# Create models directory
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")