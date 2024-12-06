from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from joblib import load
import pandas as pd
from io import StringIO
import streamlit as st
import requests

# Load the pre-trained ML pipeline
pipeline = load('Pipeline/pipeline_star_type_predictor.joblib')

# Initialize the FastAPI application
app = FastAPI()

# FastAPI endpoints
API_URL_SINGLE = "http://127.0.0.1:8000/predict-single/"
API_URL_MULTIPLE = "http://127.0.0.1:8000/predict-multiple/"

# Define the input schema for the prediction endpoint
class StarInput(BaseModel):
    temperature: int = Field(..., alias="Temperature (K)")
    luminosity: float = Field(..., alias="Luminosity(L/Lo)")
    radius: float = Field(..., alias="Radius(R/Ro)")
    absolute_magnitude: float = Field(..., alias="Absolute magnitude(Mv)")

    class Config:
        allow_population_by_field_name = True

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "App running!"}

# Single star prediction endpoint
@app.post("/predict-single/")
async def predict_star_type(star: StarInput):
    # Convert the input data to a DataFrame
    test_data = pd.DataFrame([{
        'Temperature (K)': star.temperature,
        'Luminosity(L/Lo)': star.luminosity,
        'Radius(R/Ro)': star.radius,
        'Absolute magnitude(Mv)': star.absolute_magnitude,
    }])

    # Perform the prediction
    y_pred = pipeline.predict(test_data)

    return {"predicted_type": y_pred[0]}

# Multiple stars prediction endpoint
@app.post("/predict-multiple/")
async def predict_multiple_stars(file: UploadFile = File(...)):
    try:
        # Check if file is uploaded
        if not file:
            return {"error": "No file uploaded. Please upload a CSV file."}

        # Read CSV content into a pandas DataFrame
        content = await file.read()
        csv_data = StringIO(content.decode("utf-8"))
        test_data = pd.read_csv(csv_data)

        # Validate columns
        required_columns = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']
        if not all(column in test_data.columns for column in required_columns):
            return {"error": f"CSV must contain columns: {', '.join(required_columns)}"}

        # Filter and debug input data
        test_data = test_data[required_columns]
        print("Input data shape:", test_data.shape)
        print("Input data preview:", test_data.head())

        # Check for missing values
        if test_data.isnull().values.any():
            return {"error": "Input contains missing values. Please clean your data."}

        # Perform prediction
        y_pred = pipeline.predict(test_data)
        print("Predictions:", y_pred)

        # Format results
        results = [{"input_data": row.to_dict(), "predicted_type": y_pred[idx]} for idx, row in test_data.iterrows()]
        return {"predictions": results}

    except Exception as e:
        print(f"Internal server error: {e}")
        return {"error": f"Internal Server Error: {str(e)}"}
