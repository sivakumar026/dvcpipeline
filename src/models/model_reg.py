import json
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

dagshub.init(
    repo_owner="sivakumar026",
    repo_name="dvcpipeline",
    mlflow=True
)

client = MlflowClient()

# Load run info
with open("artifacts/run_info.json") as f:
    run_info = json.load(f)

run_id = run_info["run_id"]

model_name = "LoanPredictionModel"

# Use model_uri from log_model (MLflow 3.x) or fall back to legacy URI
model_uri = run_info.get("model_uri", f"runs:/{run_id}/model")

result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

# Create model description
client.update_registered_model(
    name=model_name,
    description="Loan Approval Prediction Model using Random Forest"
)

# Add tags to registered model
client.set_registered_model_tag(
    name=model_name,
    key="created_by",
    value="Sivakumar"
)

client.set_registered_model_tag(
    name=model_name,
    key="algorithm",
    value="RandomForest"
)

# Add tags to specific version
client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="accuracy",
    value="0.85"
)

print("Registered Version:", result.version)