"""
This module is created to unify all functions to
retrieve/save data from the directories desired.
"""

import json
import os
import kaggle
import pandas as pd
from src import RAW_DATA_PATH, ROOT_PATH


def get_data_from_source() -> None:
    """
    Downloads kaggle dataset and saves it in a given path.

    Args:
        None

    Returns:
        None
    """

    # path where to save data csv
    download_dir = RAW_DATA_PATH
    # path where to find configs to get kaggle dataset
    connection_source = (
            ROOT_PATH / "config" /
            "kaggle_connection_config.json"
    )

    # read configs file
    with open(connection_source, "r", encoding="utf-8") as file:
        source_data = json.load(file)

    # save json configs
    username = source_data["username"]
    dataset_name = source_data["dataset_name"]

    # download the dataset in CSV format
    kaggle.api.dataset_download_files(
        f"{username}/{dataset_name}", path=download_dir, unzip=True
    )

    original_file_path = os.path.join(
        download_dir, r"Womens Clothing E-Commerce Reviews.csv"
    )
    desired_file_name = os.path.join(download_dir, "raw_data.csv")

    # Rename the file
    os.rename(original_file_path, desired_file_name)


def get_data_from_local(path_to_data: str) -> pd.DataFrame:
    """
    Reads data from csv and creates a DataFrame from it.

    Args:
        path_to_data: Path where data we want can be found

    Returns:
        DataFrame: The DataFrame with the csv file's data.
    """

    dataframe = pd.read_csv(path_to_data)

    return dataframe


def save_data_to_local(path_to_save: str, dataframe: pd.DataFrame) -> None:
    """
    Saves a dataframe on a certain path

    Args:
        path_to_save: Path where we want data to be stored
        dataframe: pd.DataFrame with data

    Returns:
        None
    """

    dataframe.to_csv(path_to_save.as_posix(), index=False)


if __name__ == "__main__":
    pass
