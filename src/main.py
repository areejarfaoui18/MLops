

from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.pyfunc
import numpy as np
from prometheus_client import start_http_server, Summary, Counter, generate_latest, CONTENT_TYPE_LATEST, Histogram
from fastapi.responses import Response, PlainTextResponse

from typing import List


mlflow.set_tracking_uri("file:///C:/Users/ija/Documents/mlops-ME/MLops/mlruns")

app = FastAPI(
    title="MLOps Prediction API",
    description="A FastAPI app that serves an ML model and exposes Prometheus metrics.",
    version="1.0.0"
)

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

@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    responses={
        200: {
            "content": {
                "text/plain": {
                    "example": """# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 3429.0
python_gc_objects_collected_total{generation="1"} 423.0
python_gc_objects_collected_total{generation="2"} 270.0
# HELP python_gc_objects_uncollectable_total Uncollectable objects found during GC
# TYPE python_gc_objects_uncollectable_total counter
python_gc_objects_uncollectable_total{generation="0"} 0.0
python_gc_objects_uncollectable_total{generation="1"} 0.0
python_gc_objects_uncollectable_total{generation="2"} 0.0
# HELP python_gc_collections_total Number of times this generation was collected
# TYPE python_gc_collections_total counter
python_gc_collections_total{generation="0"} 872.0
python_gc_collections_total{generation="1"} 79.0
python_gc_collections_total{generation="2"} 7.0
# HELP python_info Python platform information
# TYPE python_info gauge
python_info{implementation="CPython",major="3",minor="12",patchlevel="11",version="3.12.11"} 1.0
# HELP process_virtual_memory_bytes Virtual memory size in bytes.
# TYPE process_virtual_memory_bytes gauge
process_virtual_memory_bytes 1.738358784e+09
# HELP process_resident_memory_bytes Resident memory size in bytes.
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 2.583552e+08
# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.
# TYPE process_start_time_seconds gauge
process_start_time_seconds 1.75101271562e+09
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 12.05
# HELP process_open_fds Number of open file descriptors.
# TYPE process_open_fds gauge
process_open_fds 24.0
# HELP process_max_fds Maximum number of open file descriptors.
# TYPE process_max_fds gauge
process_max_fds 1.048576e+06
# HELP request_count_total Total number of requests
# TYPE request_count_total counter
request_count_total{endpoint="/predict",method="POST"} 1446.0
# HELP request_count_created Total number of requests
# TYPE request_count_created gauge
request_count_created{endpoint="/predict",method="POST"} 1.7510127755113187e+09
# HELP request_latency_seconds Request latency in seconds
# TYPE request_latency_seconds histogram
request_latency_seconds_bucket{endpoint="/predict",le="0.005",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="0.01",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="0.025",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="0.05",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="0.075",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="0.1",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="0.25",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="0.5",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="0.75",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="1.0",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="2.5",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="5.0",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="7.5",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="10.0",method="POST"} 1436.0
request_latency_seconds_bucket{endpoint="/predict",le="+Inf",method="POST"} 1446.0
request_latency_seconds_count{endpoint="/predict",method="POST"} 1446.0
request_latency_seconds_sum{endpoint="/predict",method="POST"} 152.27128072499727
# HELP request_latency_seconds_created Request latency in seconds
# TYPE request_latency_seconds_created gauge
request_latency_seconds_created{endpoint="/predict",method="POST"} 1.7510127755113482e+09
# HELP prediction_exceptions_total Total number of prediction exceptions
# TYPE prediction_exceptions_total counter
prediction_exceptions_total 0.0
# HELP prediction_exceptions_created Total number of prediction exceptions
# TYPE prediction_exceptions_created gauge
prediction_exceptions_created 1.7510127182637656e+09"""
                }
            },
            "description": "Prometheus metrics in plain text format"
        }
    },
    summary="Prometheus metrics endpoint",
    description="Exposes internal metrics for Prometheus scraping."
)
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


# Request schema
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        example=[0.5, 1.2, 3.4, 5.6],
        description="List of numeric features for model prediction"
    )


from fastapi import Request

@app.post(
    "/predict",
    summary="Generate a prediction",
    response_description="Prediction result as a list",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": [1]
                    }
                }
            }
        },
        422: {
            "description": "Validation error"
        }
    }
)
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

@app.get("/", summary="Health check", description="Returns basic health check status.")
def read_root():
    return {"message": "API is running!"}
