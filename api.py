from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tensorflow.keras.models import load_model
import numpy as np
from pydantic import BaseModel
from typing import List
from modeltest import gather_predicts
import os

app = FastAPI()

# Define your origins
origins = [
    "http://localhost:3000",      # React development server
    "http://localhost:8000",      # FastAPI backend
    "http://127.0.0.1:3000",     # Alternative React URL
    "http://127.0.0.1:8000",     # Alternative FastAPI URL
]

# Add CORS middleware BEFORE any route definitions
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # List of allowed origins
    allow_credentials=True,       # Allow credentials (cookies, authorization headers)
    allow_methods=["*"],         # Allow all methods
    allow_headers=["*"],         # Allow all headers
    expose_headers=["*"],        # Expose all headers
)

# Load model once at startup
MODEL = load_model("aaplnvda.keras")

PRODUCTION_URL = os.getenv('FRONTEND_URL')
if PRODUCTION_URL:
    origins.append(PRODUCTION_URL)

class PredictionRequest(BaseModel):
    symbol: str
    window_size: int = 7

@app.post("/predict")
async def get_predictions(request: PredictionRequest):
    try:
        predictions = gather_predicts(request.symbol, request.window_size)
        # Convert numpy arrays and ensure JSON serializable
        formatted_predictions = []
        for pred in predictions:
            try:
                formatted_pred = {
                    "date": pred["date"] if pred["date"] else None,
                    "actual": float(pred["actual"]) if pred["actual"] != -1 else -1,
                    "predicted": float(pred["predicted"][0][0]) if isinstance(pred["predicted"], list) else float(pred["predicted"])
                }
                formatted_predictions.append(formatted_pred)
            except Exception as format_error:
                print(f"Error formatting prediction: {str(format_error)}")
                print(f"Problem prediction: {pred}")
                raise
        return {
            "success": True,
            "predictions": predictions,
            "symbol": request.symbol
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)