

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
from prometheus_client import start_http_server, Summary, Counter, generate_latest, CONTENT_TYPE_LATEST, Histogram
from fastapi.responses import Response

mlflow.set_tracking_uri("file:///C:/Users/ija/Documents/mlops-ME/MLops/mlruns")

app = FastAPI()

# Total requests counter, labeled by HTTP method and endpoint
REQUEST_COUNT = Counter(
    "request_count", "Total number of requests",
    ['method', 'endpoint']
)

# Request latency histogram (better than Summary for aggregations)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds",
    ['method', 'endpoint']
)

# Count exceptions (errors) during prediction
PREDICTION_EXCEPTIONS = Counter(
    "prediction_exceptions_total", "Total number of prediction exceptions"
)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

model = None


def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = mlflow.pyfunc.load_model("./model_artifact")
        print("Model loaded")
    return model


class PredictRequest(BaseModel):
    features: list


from fastapi import Request

@app.post("/predict")
def predict(request: PredictRequest, http_request: Request):
    method = http_request.method
    endpoint = http_request.url.path
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    
    with REQUEST_LATENCY.labels(method=method, endpoint=endpoint).time():
        try:
            model = get_model()
            data = np.array(request.features).reshape(1, -1)
            prediction = model.predict(data)
            return {"prediction": prediction.tolist()}
        except Exception:
            PREDICTION_EXCEPTIONS.inc()
            raise

@app.get("/")
def read_root():
    return {"message": "API is running!"}
