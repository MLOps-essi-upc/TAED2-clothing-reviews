# IMPORTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import model_svc
import joblib


# FUNCTIONS

def main():
    df = read_data()

    stem = True
    if stem:
        x = df["Stemmed Review Text"]
    else:
        x = df["Review Text"]
    y = df["Top Product"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=101)

    x_train_tf_idf, x_test_tf_idf = model_svc.vectorization(x_train, x_test)

    # Train Random forest model on the scaled data
    random_forest = RandomForestClassifier()
    # Fit the model
    random_forest.fit(x_train_tf_idf, y_train)

    # Saving the model as an artifact.
    if stem:
        filename = "models/model_rf_stem"
    else:
        filename = "models/model_rf"

    # save model
    joblib.dump(random_forest, filename)

    # load model
    loaded_model = joblib.load(filename)
    # predict x_test_scaled
    pred_rand = loaded_model.predict(x_test_tf_idf)
    # calling the score function
    eval_fores = model_svc.classification_task(loaded_model, x_train_tf_idf, y_train, x_test_tf_idf, y_test, pred_rand,
                                               "Random Forest")
    print(eval_fores)


if __name__ == '__main__':
    main()
