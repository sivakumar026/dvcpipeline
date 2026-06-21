import os
import sys
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient

# Path setup
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.append(BASE_DIR)

from utils import load_params

# Load params
params = load_params("params.yaml")["model_building"]

test_size = params["test_size_param"]
random_state = params["random_state_param"]
n_estimators = params["n_estimators"]

# Load data
df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "train_processed.csv"
    )
)

target_col = "Loan_Status"

X = df.drop(columns=[target_col])
y = df[target_col]

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

# DagsHub + MLflow
dagshub.init(
    repo_owner="sivakumar026",
    repo_name="dvcpipeline",
    mlflow=True
)

mlflow.set_experiment("Loan_Prediction")

mlflow.autolog(disable=True)
mlflow.sklearn.autolog(disable=True)

with mlflow.start_run() as run:

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    # Log model with error catching
    try:
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )
        print("[OK] Model logged successfully")
        print("Model URI:", model_info.model_uri)
    except Exception as e:
        print("[FAIL] Model logging failed:", e)
        model_info = None

    print("Validation Accuracy:", accuracy)

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(model, os.path.join(BASE_DIR, "models", "model.pkl"))

    run_id = run.info.run_id

    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    run_info = {"run_id": run_id}
    if model_info is not None:
        run_info["model_uri"] = model_info.model_uri

    with open(os.path.join(artifacts_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=4)

    print("Run ID:", run_id)

    client = MlflowClient()
    print("\nArtifacts:")
    for artifact in client.list_artifacts(run_id):
        print(" -", artifact.path)
    print("--- end of artifacts ---")