"""Definitions for the objects used by our resource endpoints."""

from enum import Enum

from pydantic import BaseModel


class SentimentRequest(BaseModel):
    """
    This class has been created to define
    which type of variable would define the
    request made by the user
    """
    text: str


class SentimentResponse(BaseModel):
    """
    This class has been created to
    define which type of variables would
    define the sentiment and probability
    returned to the user given their text
    request
    """

    sentiment: str
    probabilities: float


class SentimentType(Enum):
    """
    This class has been created to define
    with which values will be each sentiment
    category associated with
    """

    NoTop = 0
    TopProduct = 1
