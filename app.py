from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/churn_model.joblib")
app = FastAPI(title="Telecom Customer Churn Prediction API", version="1.0.0")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class CustomerData(BaseModel):
    Gender: Optional[str] = None
    Senior_Citizen: Optional[str] = None
    Tenure_Months: Optional[float] = Field(default=None, ge=0, le=72)
    Contract_Type: Optional[str] = None
    Internet_Service: Optional[str] = None
    Payment_Method: Optional[str] = None
    Monthly_Charges_INR: Optional[float] = Field(default=None, ge=0)
    Total_Charges_INR: Optional[float] = Field(default=None, ge=0)
    Tech_Support: Optional[str] = None
    Online_Security: Optional[str] = None
    Support_Tickets_6M: Optional[float] = Field(default=None, ge=0)
    Avg_Data_Usage_GB: Optional[float] = Field(default=None, ge=0)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run train.py first.")
    try:
        row = pd.DataFrame([customer.model_dump()])
        prediction = str(model.predict(row)[0])
        result = {"prediction": prediction}
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            result["probability"] = {
                str(label): round(float(prob), 4)
                for label, prob in zip(model.classes_, probabilities)
            }
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
