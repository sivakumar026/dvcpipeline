import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")

# Drop Loan_ID
if "Loan_ID" in train_df.columns:
    train_df = train_df.drop(columns=["Loan_ID"])

if "Loan_ID" in test_df.columns:
    test_df = test_df.drop(columns=["Loan_ID"])

# Handle missing values
for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].dtype != "object":
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        else:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

# Create output directory
os.makedirs("data/processed", exist_ok=True)

# Save processed data
train_df.to_csv("data/processed/train_processed.csv", index=False)
test_df.to_csv("data/processed/test_processed.csv", index=False)

print("Data preprocessing completed successfully!")