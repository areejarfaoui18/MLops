import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
from fastapi.testclient import TestClient
from main import app, get_model

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running!"}

def test_predict_valid():
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}  # Iris example features
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == 1

def test_predict_invalid():
    payload = {"features": "not a list"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_get_model_caching():
    m1 = get_model()
    m2 = get_model()
    assert m1 is m2  # Same cached model instance
