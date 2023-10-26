import pytest
from fastapi.testclient import TestClient

from src.app.api import app


@pytest.fixture(scope="module", autouse=True)
def client():
    # Use the TestClient with a `with` statement to trigger the startup and shutdown events.
    with TestClient(app) as client:
        return client


@pytest.fixture
def payload():
    return {
        "sentiment": "i like dress",
        "probabilities": 0.65,
    }


def test_root(client):
    response = client.get("/")
    json = response.json()
    assert response.status_code == 200
    assert (
        json["data"]["message"]
        == "Welcome to Sentiment Analysis "
           "Clothing Reviews! Please, read the `/docs`!"
    )
    assert json["message"] == "OK"
    assert json["status-code"] == 200
    assert json["method"] == "GET"
    assert json["url"] == "http://testserver/"
    assert json["timestamp"] is not None

def test_model_prediction(client, payload):
    response = client.post("/predict", json=payload)
    json = response.json()
    assert response.status_code == 200
    assert json["data"]["prediction"] == 2
    assert json["message"] == "OK"
    assert json["status-code"] == 200
    assert json["method"] == "POST"
    assert json["url"] == "http://testserver/predict"
    assert json["timestamp"] is not None