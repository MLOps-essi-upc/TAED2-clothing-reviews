"""
This module is created to test functions and
functionalities on src/app/api.py
"""

import pytest
from fastapi.testclient import TestClient
from src.app.api import app


@pytest.fixture(scope="module", autouse=True)
def client():
    """
    Fixture function for creating a TestClient instance.

    This fixture is designed to be used in Pytest for
    creating a TestClient instance that can be used to
    interact with a web application during testing. The
    TestClient is created within a `with` statement to
    trigger the startup and shutdown events
    automatically.

    Returns:
        TestClient: A TestClient instance for
        interacting with the web application.
    """

    with TestClient(app) as client:
        return client


@pytest.fixture
def payload_valid():
    """
    Fixture function for generating a sample valid payload.

    This fixture is designed to generate a sample payload,
    typically in the form of a dictionary (json), that can be
    used as input data in test cases.
    In this case, we use it on test_predict_valid_request.

    Returns:
        dict: A dictionary representing a sample payload.
    """

    return {
        "text": "i like dress"
    }


@pytest.fixture
def payload_exception():
    """
    Fixture function for generating a
    sample payload that raises an exception.

    This fixture is designed to generate a sample payload,
    typically in the form of a dictionary (json), that can be
    used as input data in test cases.
    In this case, we use it on test_predict_exception.

    Returns:
        dict: A dictionary representing a sample payload.
    """

    return {
        "text": ""
    }


@pytest.fixture
def payload_not_valid():
    """
    Fixture function for generating a sample not valid
     payload.

    This fixture is designed to generate a sample payload,
    typically in the form of a dictionary (json),
    that can be used as input data in test cases.
    In this case, we use it on test_predict_invalid_request

    Returns:
        dict: A dictionary representing a sample payload.
    """

    return {
        "text": [0, 3, 4, 1, 9]
    }


def test_root_api(client):
    """
    Test the root API endpoint for the Sentiment Analysis
    Clothing Reviews application.

    This test function sends a GET request to the root
    endpoint of the application and validates the response
    to ensure that the API is working correctly.

    Args:
        client (TestClient): A TestClient instance for
        interacting with the application.
    """

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


def test_predict_valid_request(client, payload_valid):
    """
    Test the 'predict' API endpoint with a valid request.

    This test function sends a POST request to the
    'predict' endpoint of the application
    with a valid payload and validates the response to
    ensure that the prediction API is
    functioning correctly.

    Args:
        client (TestClient): A TestClient instance for
        interacting with the application.
        payload_valid (dict): A valid payload for
        sentiment analysis prediction.
    """

    response = client.post("/predict", json=payload_valid)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "probabilities" in data


def test_predict_exception(client, payload_exception):
    """
    Test the 'predict' API endpoint with an
    exception-inducing request.

    This test function sends a POST request to the
    'predict' endpoint of the application
    with a payload designed to trigger an exception
    and validates that the response
    status code is 400, indicating a client error.

    Args:
        client (TestClient): A TestClient instance for
        interacting with the application.
        payload_exception (dict): A payload that is
        expected to cause an exception.
    """

    response = client.post("/predict", json=payload_exception)
    assert response.status_code == 400


def test_predict_invalid_request(client, payload_not_valid):
    """
    Test the 'predict' API endpoint with an invalid request.

    This test function sends a POST request to the
    'predict' endpoint of the application
    with an invalid payload and validates that the
    response status code is 422, indicating
    unprocessable entity due to the invalid request.

    Args:
        client (TestClient): A TestClient instance for
        interacting with the application.
        payload_not_valid (dict): An invalid payload
        that should result in a 422 status code.
    """

    response = client.post("/predict", json=payload_not_valid)
    assert response.status_code == 422
