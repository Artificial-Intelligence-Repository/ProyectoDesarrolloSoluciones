from typing import Any, List, Optional

from pydantic import BaseModel

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

class Review(BaseModel):
    review: str

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[Review]
    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "review": "the movie was great!"
                    }
                ]
            }
        }
