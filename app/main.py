import os
from pathlib import Path
from typing import List

import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get(
    "MODEL_DIR", str(Path(__file__).resolve().parent.parent / "model")
)

app = FastAPI(title="Digits classifier")
model = mlflow.sklearn.load_model(MODEL_DIR)


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=64, max_length=64)


@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model.predict([request.features])
    return {"prediction": int(prediction[0])}
