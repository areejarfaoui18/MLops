from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

mlflow.set_tracking_uri("file:///C:/Users/ija/Documents/mlops-ME/MLops/mlruns")

app = FastAPI()

model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = mlflow.pyfunc.load_model("runs:/2a3055fae72e47bbb72eea6aae4bd555/model")
        print("Model loaded")
    return model

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictRequest):
    model = get_model()
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}

@app.get("/")
def read_root():
    return {"message": "API is running!"}
