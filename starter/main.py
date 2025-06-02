from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import numpy as np
import pandas as pd
import os

# Define categorical features (must match training)
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Pydantic model for input (use alias for hyphens)
class CensusInput(BaseModel):
    age: int
    workclass: str = Field(..., alias="workclass")
    fnlgt: int
    education: str = Field(..., alias="education")
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(..., alias="occupation")
    relationship: str = Field(..., alias="relationship")
    race: str = Field(..., alias="race")
    sex: str = Field(..., alias="sex")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

# Load model and encoders
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.joblib"))
lb = joblib.load(os.path.join(MODEL_DIR, "lb.joblib"))

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Census Income Prediction API!"}

@app.post("/predict")
def predict(input_data: CensusInput):
    # Convert input to DataFrame with correct column names
    data_dict = input_data.dict(by_alias=True)
    df = pd.DataFrame([data_dict])
    # Process data
    X, _, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, label=None, training=False, encoder=encoder, lb=lb
    )
    # Predict
    pred = inference(model, X)
    label = lb.inverse_transform(pred)[0]
    return {"prediction": label}
