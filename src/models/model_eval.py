import pandas as pd
import pickle
import json
import os
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
test_data = pd.read_csv("data/processed/test_processed.csv")

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Load model
model = joblib.load(open("models/model.pkl", "rb"))

# Predictions
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {
    "accuracy": float(acc),
    "precision": float(pre),
    "recall": float(rec),
    "f1_score": float(f1)
}

os.makedirs("reports", exist_ok=True)

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(metrics)
print("Metrics saved successfully")