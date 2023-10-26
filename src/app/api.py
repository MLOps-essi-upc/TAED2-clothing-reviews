"""
This module is created to construct an api
from where users can interact with our model.
"""

from functools import wraps
from typing import List
from http import HTTPStatus
import datetime
from fastapi import FastAPI, HTTPException, Request
from datasets import Dataset
import pandas as pd
import torch
from torch import argmax
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src import ROOT_PATH
from src.models.train_model import tokenize_dataset
from src.app.schemas import SentimentRequest, SentimentResponse


MODEL_PATH = ROOT_PATH / "model"

model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title="Sentiment Analysis - Clothing Reviews",
    description="This API lets you make predictions on "
                "clothing-reviews dataset using bert-finetuned model.",
    version="0.1",
)

sentiment_model = None


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.on_event("startup")
def _load_models():
    """Loads models"""
    global sentiment_model
    sentiment_model = torch.load(
        MODEL_PATH / "transfer-learning.pt", map_location="cpu"
    )


@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "message": "Welcome to Sentiment Analysis "
                       "Clothing Reviews! Please, read the `/docs`!"
        },
    }
    return response


def preprocess(data) -> Dataset:
    """
    Preprocess data to be used as model input.
    Args:
        data: review
    Returns: a Dataset with the review
            in torch format
    """
    words = data.split()
    data = pd.DataFrame({'Review Text': words})
    # Convert Python DataFrame to Hugging Face arrow dataset
    hg_data = Dataset.from_pandas(data)

    # Tokenize the data sets
    dataset = hg_data.map(tokenize_dataset)
    # Remove the review and index columns because it
    # will not be used in the model
    dataset = dataset.remove_columns(["Review Text"])
    # Change the format to PyTorch tensors
    dataset.set_format("torch")

    return dataset


def predict_sentiment(text: str):
    """
    Computes the sentiment prediction
    Args:
        text: sentence to be analyzed

    Returns:
        sentiment: prediction
            1 if positive sentiment
            0 otherwise
        probability_value: which
        probability of containing good
        or bad sentiment
    """
    dataset_text = preprocess(text)
    text_dataloader = DataLoader(
        dataset=dataset_text, shuffle=True,
        batch_size=4
    )

    sentiment_model.eval()

    #  A list for all logits
    logits_all = []

    # A list for all predicted probabilities
    pred_prob_all = []

    # A list for all predicted labels
    predictions_all = []

    # Loop through the batches in the evaluation dataloader
    for batch in text_dataloader:
        # Disable the gradient calculation
        with torch.no_grad():
            # Compute the model output
            outputs = sentiment_model(**batch)
        # Get the logits
        logits = outputs.logits
        # Append the logits batch to the list
        logits_all.append(logits)
        # Get the predicted probabilities for the batch
        predicted_prob = F.softmax(logits, dim=1)
        # Append the predicted probabilities for the batch to
        # all the predicted probabilities
        pred_prob_all.append(predicted_prob)
        # Get the predicted labels for the batch
        prediction = argmax(logits, dim=-1)
        # Append the predicted labels for the batch
        # to all the predictions
        predictions_all.append(prediction)

    class_names = ["No Top", "Top"]

    average_probabilities = torch.mean(
        torch.stack(
            [p[:len(pred_prob_all[-1])] for p in pred_prob_all]
        ), dim=0
    )

    probabilities_dict = dict(zip(class_names, average_probabilities[0]))

    probabilities = probabilities_dict
    sentiment = max(probabilities, key=probabilities.get)
    probability_value = probabilities[sentiment].item()

    print(f"sentiment = {sentiment}, probabilities = {probability_value}")

    return sentiment, round(probability_value,3)


@app.post("/predict", response_model=SentimentResponse)
def _predict(request: SentimentRequest):
    """
    Return a prediction to the user.
    Args:
        request: Request with some text to be analyzed
    Returns:
        Sentiment prediction on that text
    """
    try:
        sentiment, prob = predict_sentiment(request.text)
        return SentimentResponse(sentiment=sentiment, probabilities=prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
