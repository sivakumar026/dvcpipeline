import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "external", "loan_data.csv"))

train_data, test_data = train_test_split(
    data,
    test_size=0.3,
    random_state=42
)



data_path = os.path.join("data", "raw")

os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

print("Data collection completed successfully!")