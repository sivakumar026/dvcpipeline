import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Path setup ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(BASE_DIR)  # ✅ utils.py is in root dvcpipeline/

from utils import load_params

# ── Load params ───────────────────────────────────────────────────────
params       = load_params("params.yaml")["data_collection"]
test_size    = params["test_size_param"]
random_state = params["random_state_param"]

# ── Load data ─────────────────────────────────────────────────────────
data = pd.read_csv(os.path.join(BASE_DIR, "data", "external", "loan_data.csv"))

# ── Split data ────────────────────────────────────────────────────────
train_data, test_data = train_test_split(
    data,
    test_size=test_size,
    random_state=random_state
)

# ── Save output ───────────────────────────────────────────────────────
data_path = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"),   index=False)

print("Data collection completed successfully!")