# IMPORTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import model_svc
import joblib
import mlflow


# FUNCTIONS
def tracking():
    """
        Set up MLflow tracking URI and enable automatic logging.

        Returns:
            None
    """
    mlflow.set_tracking_uri("https://github.com/MLOps-essi-upc/clothing-reviews/tree/develop/models")
    mlflow.autolog()


def stemming(df, stem=True) -> tuple:
    """
        Preprocesses data and selects features and target based on the stemming flag.
        Args:
            df (DataFrame): The input DataFrame.
            stem (bool): A flag indicating whether stemming is applied.
        Returns:
            tuple: A tuple containing the feature variable (x) and the target variable (y).
    """
    if stem:
        x = df["Stemmed Review Text"]
    else:
        x = df["Review Text"]
    y = df["Top Product"]
    return x, y


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
    filename = "models/model_rf_stem" if stem else "models/model_rf"
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
    filename = "models/model_rf_stem" if stem else "models/model_rf"
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
    tracking(

    )
    # Read the data and preprocess it as needed
    df = read_data()

    # Set this flag based on whether stemming is applied or not
    use_stemming = True
    x, y = stemming(df, use_stemming)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=101)

    # Vectorize the text data
    x_train_tf_idf, x_test_tf_idf = model_svc.vectorization(x_train, x_test)

    # Train and save the machine learning model
    train_and_save_model(x_train_tf_idf, y_train, use_stemming)

    # Load the model
    loaded_model = loading(use_stemming)
    # Make predictions using the loaded model
    predictions = prediction(loaded_model, x_test_tf_idf)

    # Evaluate and print the model's performance
    eval_fores = model_svc.classification_task(loaded_model, x_train_tf_idf, y_train, x_test_tf_idf, y_test,
                                               predictions, "Random Forest")
    print(eval_fores)
