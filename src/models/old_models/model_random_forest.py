"""
This module is created to contain all functions
related to the random forest model.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from src.data.get_and_save_data import get_data_from_local
from src import PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH


def stemming(df, stem=True) -> tuple:
    """
     Preprocesses data and selects features and target
     based on the stemming flag.
     Args:
         df (DataFrame):
         - The input DataFrame.
         stem (bool):
         - A flag indicating whether stemming is applied.
     Returns:
         tuple:
         - A tuple containing the feature variable (x) and the
         target variable (y).
    """
    # Check if stemming should be applied
    if stem:
        # If stemming is enabled,
        # use the "Stemmed Review Text" as the feature
        x = df["Stemmed Review Text"]
    else:
        # If stemming is disabled,
        # use the original "Review Text" as the feature
        x = df["Review Text"]
    # Extract the target variable "Top Product"
    y = df["Top Product"]
    return x, y


def vectorization(x_train, x_test):
    """
    Converts text data into TF-IDF vector representations.
    Args:
        x_train: Training text data.
        x_test: Testing text data.
    Returns:
        tuple:
        - A tuple containing TF-IDF vectors for training
        and testing data.
    """
    # Initialize the TF-IDF vectorizer
    tf_idf_vectorizer = TfidfVectorizer()
    # Fit and transform the vectorizer on the training data
    x_train_tf_idf = tf_idf_vectorizer.fit_transform(x_train)
    # Transform the testing data using the same vectorizer
    x_test_tf_idf = tf_idf_vectorizer.transform(x_test)
    return x_train_tf_idf, x_test_tf_idf


def classification_task(
        model,
        x_train_scaled,
        y_train,
        x_test_scaled,
        y_test,
        predic,
        model_name
):
    """
    Evaluates a classification model and returns performance metrics.
    Args:
        model: The trained classification model.
        x_train_scaled: Scaled training features.
        y_train: Training labels.
        x_test_scaled: Scaled testing features.
        y_test: Testing labels.
        predic: Predicted labels.
        model_name: Name of the model for
        identification in the results.
    Returns:
        DataFrame: Performance metrics including
        accuracy, precision, recall, F1-score.
    """
    perf_df = pd.DataFrame(
        {'Train_Score': model.score(x_train_scaled, y_train),
         'Test_Score': model.score(x_test_scaled, y_test),
         'Precision_Score': precision_score(y_test, predic),
         'Recall_Score': recall_score(y_test, predic),
         'F1_Score': f1_score(y_test, predic),
         'Accuracy': accuracy_score(y_test, predic)},
        index=[model_name])
    return perf_df


def train_and_save_model(x, y, stem=True):
    """
     Trains a Random Forest model on the provided data and saves it.
     Args:
         x: Feature variable.
         y: Target variable.
         stem (bool): A flag indicating whether stemming is applied.
     Returns:
         None.
    """
    # Train the model Random Forest
    random_forest = RandomForestClassifier()
    random_forest.fit(x, y)
    # Determine the model file name based on whether stemming is applied
    filename = (
        "model/model_rf_stem.joblib" if stem else "model/model_rf.joblib"
    )
    # Save the model
    joblib.dump(random_forest, filename)


def loading(stem=True):
    """
     Loads a trained model based on the stemming flag.
     Args:
         stem (bool): A flag indicating whether stemming is applied.
     Returns:
         model: The loaded model.
    """
    # Determine the model file name based on whether stemming is applied
    filename = (
        "model/model_rf_stem.joblib" if stem else "model/model_rf.joblib"
    )
    # Load the model
    model = joblib.load(filename)
    return model


def prediction(model, x_test) -> list:
    """
     Makes predictions using a loaded model.
     Args:
         model: The loaded model.
         x_test: Testing dataset.
     Returns:
         predictions: Predicted values.
    """
    # Predict using the loaded model
    predictions = model.predict(x_test)
    return predictions


if __name__ == '__main__':
    # Load and preprocess the data
    test = get_data_from_local(PROCESSED_TEST_DATA_PATH)
    train = get_data_from_local(PROCESSED_TRAIN_DATA_PATH)

    # Set this flag based on whether stemming is applied or not
    USE_STEMMING = True
    x_train, y_train = stemming(test, USE_STEMMING)
    x_test, y_test = stemming(test, USE_STEMMING)

    # Vectorize the text data
    x_train_tf_idf, x_test_tf_idf = vectorization(x_train, x_test)

    # Train and save the machine learning model
    train_and_save_model(x_train_tf_idf, y_train, USE_STEMMING)

    # Load the model
    loaded_model = loading(USE_STEMMING)
    # Make predictions using the loaded model
    predictions = prediction(loaded_model, x_test_tf_idf)

    # Evaluate and print the model's performance
    eval_fores = classification_task(
        loaded_model, x_train_tf_idf, y_train, x_test_tf_idf, y_test,
        predictions, "Random Forest"
    )
    print(eval_fores)
