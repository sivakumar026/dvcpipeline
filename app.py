from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ── App setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Prediction API",
    description="Predict loan approval using a Random Forest model",
    version="1.0.0",
)

# ── Load model ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# ── Encoding maps (must match data_prep.py LabelEncoder order) ────────
GENDER_MAP = {"Female": 0, "Male": 1}
MARRIED_MAP = {"No": 0, "Yes": 1}
DEPENDENTS_MAP = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED_MAP = {"No": 0, "Yes": 1}
PROPERTY_AREA_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}


# ── Request / Response schemas ─────────────────────────────────────────
class LoanRequest(BaseModel):
    Gender: str = Field(..., example="Male", description="Male or Female")
    Married: str = Field(..., example="Yes", description="Yes or No")
    Dependents: str = Field(..., example="0", description="0, 1, 2, or 3+")
    Education: str = Field(..., example="Graduate", description="Graduate or Not Graduate")
    Self_Employed: str = Field(..., example="No", description="Yes or No")
    ApplicantIncome: float = Field(..., example=5000, description="Applicant income")
    CoapplicantIncome: float = Field(..., example=1500, description="Co-applicant income")
    LoanAmount: float = Field(..., example=128, description="Loan amount (in thousands)")
    Loan_Amount_Term: float = Field(..., example=360, description="Loan term (in days)")
    Credit_History: float = Field(..., example=1.0, description="Credit history (1 = good, 0 = bad)")
    Property_Area: str = Field(..., example="Urban", description="Rural, Semiurban, or Urban")


class LoanResponse(BaseModel):
    prediction: str
    probability: float


# ── Helper ─────────────────────────────────────────────────────────────
def encode_input(data: LoanRequest) -> np.ndarray:
    """Convert human-readable input to encoded feature array."""
    try:
        features = [
            GENDER_MAP[data.Gender],
            MARRIED_MAP[data.Married],
            DEPENDENTS_MAP[data.Dependents],
            EDUCATION_MAP[data.Education],
            SELF_EMPLOYED_MAP[data.Self_Employed],
            data.ApplicantIncome,
            data.CoapplicantIncome,
            data.LoanAmount,
            data.Loan_Amount_Term,
            data.Credit_History,
            PROPERTY_AREA_MAP[data.Property_Area],
        ]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid value: {e}")

    return np.array(features).reshape(1, -1)


# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Loan Prediction API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=LoanResponse)
def predict(request: LoanRequest):
    """Predict whether a loan will be approved or rejected."""
    features = encode_input(request)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return LoanResponse(
        prediction="Approved" if prediction == 1 else "Rejected",
        probability=round(float(probability[prediction]), 4),
    )
