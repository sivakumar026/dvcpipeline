import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# ── Robust path setup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
raw_path       = os.path.join(BASE_DIR, "data", "raw")
processed_path = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(raw_path, "train.csv"))
test_df  = pd.read_csv(os.path.join(raw_path, "test.csv"))
print("Data loaded successfully")

# ── Drop Loan_ID ──────────────────────────────────────────────────────
for df in [train_df, test_df]:
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

# ── Handle missing values ─────────────────────────────────────────────
for df in [train_df, test_df]:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        else:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled")

# ── Encode categorical columns ────────────────────────────────────────
le = LabelEncoder()
for df in [train_df, test_df]:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

print("Encoding completed")

# ── Save processed data ───────────────────────────────────────────────
train_df.to_csv(os.path.join(processed_path, "train_processed.csv"), index=False)
test_df.to_csv(os.path.join(processed_path,  "test_processed.csv"),  index=False)

print("Data saved to", processed_path)
print("Data preprocessing completed successfully!")