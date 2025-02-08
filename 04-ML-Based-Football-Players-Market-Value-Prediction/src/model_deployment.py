# serves the trained model via an API using FastAPI
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("../models/best_model.pkl")

@app.post("/predict/")
def predict(features: dict):
    # preprocess input and return prediction
    return {"predicted_value": prediction}
