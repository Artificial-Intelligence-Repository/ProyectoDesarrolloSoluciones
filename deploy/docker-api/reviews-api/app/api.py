import json
from typing import Any
import os
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
import mlflow.sklearn

from app import __version__, schemas
from app.config import settings

MODEL_PATH = "model_export"

# At the top of the file, add:
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Model path absolute: {os.path.abspath(MODEL_PATH)}")
logger.info(f"Directory contents: {os.listdir('.')}")
logger.info(f"Model directory contents: {os.listdir(MODEL_PATH)}")

try:
    # Cargamos el modelo al iniciar la aplicación (un pipeline de scikit-learn)
    model = mlflow.sklearn.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    logger.info(f"Model type: {type(model)}")
    
    # Test the model immediately after loading
    test_pred = model.predict(["test review"])
    logger.info(f"Test prediction successful: {test_pred}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

api_router = APIRouter()

# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model
    )

    return health.dict()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Prediccion usando el modelo mejor entrenado para clasificación de sentimientos en reseñas en inglés
    """
    reviews = [item.review for item in input_data.inputs]

    logger.info(f"Making prediction on inputs: {reviews}")

    # El modelo solo soporta predict con una sola entrada
    results = []
    for i in reviews:
        logger.info(f"Input: {i}")
        prediction = model.predict([i])[0]
        logger.info(f"Prediction: {prediction}")
        results.append(prediction)

    logger.info(f"Prediction results: {results}")

    # Format the response according to PredictionResults schema
    return {
        "version": __version__,
        "predictions": results.tolist() if isinstance(results, np.ndarray) else results
    }
