import pandas as pd
import json
import os
import joblib
import mlflow
import dagshub

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



dagshub.init(repo_owner='sivakumar026', repo_name='dvcpipeline', mlflow=True)


# Load test data
test_data = pd.read_csv("data/processed/test_processed.csv")

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Load model
model = joblib.load("models/model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log metrics to MLflow
with mlflow.start_run(run_name="model_evaluation"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", pre)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

metrics = {
    "accuracy": float(acc),
    "precision": float(pre),
    "recall": float(rec),
    "f1_score": float(f1)
}

os.makedirs("reports", exist_ok=True)

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Log metrics file as artifact
mlflow.log_artifact("reports/metrics.json")

print(metrics)
print("Metrics saved successfully")