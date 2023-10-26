import pytest
from fastapi.testclient import TestClient
from src.app.api import app


@pytest.fixture(scope="module", autouse=True)
def client():
    # Use the TestClient with a `with` statement
    # to trigger the startup and shutdown events.
    with TestClient(app) as client:
        return client


@pytest.fixture
def payload():
    return {
        "text": "i like dress"
    }


def test_root_api(client):
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


def test_predict_valid_request(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "probabilities" in data


def test_predict_exception(client):
    json_not_valid = {"text": ""}
    response = client.post("/predict", json=json_not_valid)
    assert response.status_code == 400


def test_predict_invalid_input(client):
    json_not_valid = {"text": [0, 3, 4, 1, 9]}
    response = client.post("/predict", json=json_not_valid)
    assert response.status_code == 422
