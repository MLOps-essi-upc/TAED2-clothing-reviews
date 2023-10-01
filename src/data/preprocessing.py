import pandas
import pandas as pd
import altair as alt
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import requests
import os
from zipfile import ZipFile

def download_df() -> string:
    dataset_url = "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews"
    download_dir = "https://github.com/MLOps-essi-upc/clothing-reviews/blob/97e27f428bab87e3720513dc88a98977ca19e698/data/external"
    os.makedirs(download_dir, exist_ok=True)
    # Download the dataset
    response = requests.get(dataset_url)


    if response.status_code == 200:
        # Save the downloaded dataset as a CSV file
        with open(os.path.join(download_dir, "dataset.csv"), "wb") as file:
            file.write(response.content)

        print("Dataset downloaded as dataset.csv.")
    else:
        print("Failed to download the dataset. Check the URL or your internet connection.")
def create_df() -> pd.DataFrame:
    """
        Reads data from csv and creates a DataFrame from it.

        Args:
            None.

        Returns:
            DataFrame: The DataFrame with the csv file's data.
    """
    dataframe = pd.read_csv("clothing-reviews/data/external/dataset.csv")
    return dataframe


def dropping(dataframe) -> pd.DataFrame:
    """
        Erases all non-necessary columns from the DataFrame.

        Args:
            dataframe (DataFrame): the input DataFrame to be changed.

        Returns:
            dataframe: The changed DataFrame.
    """
    dataframe.drop(
        ["Unnamed:0", "Title", "Positive Feedback Count", "Division Name", "Department Name", "Class Name", "Age",
         "Clothing ID", "Rating"], axis=1, inplace=True)
    dataframe.drop_duplicates(inplace=True)
    return dataframe


def binarization(dataframe) -> pd.DataFrame:
    """
        Takes the Ratings variable and binarizes its values to 1 or 0.

        Args:
            dataframe (DataFrame): Input DataFrame to be modified.

        Returns:
            dataframe: Modified DataFrame.
    """
    dataframe.loc[dataframe['Rating'] <= 4, 'Recommended IND'] = 0
    dataframe.rename(columns={'Recommended IND': 'Top Product'}, inplace=True)
    return dataframe


def tokenization(dataframe) -> pd.DataFrame:
    """
            Tokenizes the text and lower-cases it.

            Args:
                dataframe (DataFrame): Input DataFrame to be modified.

            Returns:
                dataframe: Modified DataFrame.
    """
    nltk.download("punkt")
    nltk.download("stopwords")
    english_sw = set(stopwords.words('english') + list(string.punctuation))
    dataframe['Review Text'] = dataframe['Review Text'].apply(
        lambda x: " ".join([w.lower() for w in word_tokenize(str(x)) if w.lower() not in english_sw]))
    dataframe['Review Text'] = dataframe['Review Text'].apply(lambda text: " ".join([w for w in text if w not in "'s"]))

    return dataframe

def stemmed_text(dataframe) -> pd.DataFrame:
    """
                Stems the Review Text from the dataframe.

                Args:
                    dataframe (DataFrame): Input DataFrame to be modified.

                Returns:
                    dataframe: Modified DataFrame.
        """
    stem = SnowballStemmer('english')
    dataframe['Stemmed Review Text'] = dataframe['Review Text'].apply(lambda text: " ".join([stem.stem(w) for w in text]))
    return dataframe

if __name__ == '__main__':
    download_df()
    df = create_df()
    df = binarization(df)
    df = dropping(df)
    df = tokenization(df)
    df = stemmed_text(df)

