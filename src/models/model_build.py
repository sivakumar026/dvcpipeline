import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── Path setup ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from utils import load_params

# ── Load params ───────────────────────────────────────────────────────
params       = load_params("params.yaml")["model_building"]
test_size    = params["test_size_param"]
random_state = params["random_state_param"]
n_estimators = params["n_estimators"]

# ── Load processed data ───────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "train_processed.csv"))

# ── Split features and target ─────────────────────────────────────────
target_col = "Loan_Status"
X = df.drop(columns=[target_col])
y = df[target_col]

# ── Train-validation split ────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=test_size,      # ✅ from params
    random_state=random_state # ✅ from params
)

# ── Train model ───────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=n_estimators,  # ✅ from params
    random_state=random_state   # ✅ from params
)
model.fit(X_train, y_train)

# ── Validation accuracy ───────────────────────────────────────────────
y_pred   = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# ── Save model ────────────────────────────────────────────────────────
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, os.path.join(models_dir, "model.pkl"))

print("Model saved successfully!")