from predict_model import *
from transformers import AutoModelForSequenceClassification
import joblib
from src import PROCESSED_DATA_DIR, MODELS_DIR

TRAIN_ALL_MODEL = False

if __name__ == '__main__':
    # Read the train and test datasets
    train_data = read_data(PROCESSED_DATA_DIR / "train/train_data.csv")
    test_data = read_data(PROCESSED_DATA_DIR / "test/test_data.csv")

    # Set this flag based on whether stemming is applied or not
    use_stemming = True
    train_data = stemming(train_data, use_stemming)
    test_data = stemming(test_data, use_stemming)

    # Preprocess and tokenize data
    dataset_train = preprocess_and_tokenize_data(train_data, use_stemming)
    dataset_test = preprocess_and_tokenize_data(test_data, use_stemming)

    # Empty cache
    torch.cuda.empty_cache()

    # DataLoader
    train_dataloader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=4)
    eval_dataloader = DataLoader(dataset=dataset_test, batch_size=4)

    if TRAIN_ALL_MODEL:
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        training(train_dataloader, model)
    else:
        model = joblib.load(MODELS_DIR / 'dvclive/transfer-learning.joblib')

    prediction(eval_dataloader, model)
