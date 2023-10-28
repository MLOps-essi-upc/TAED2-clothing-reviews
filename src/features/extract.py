"""
This module is created to download data from Kaggle
and save it on data/raw folder.
"""

from src.data.get_and_save_data import get_data_from_source

if __name__ == '__main__':
    # perform the stage that gets data from
    # Kaggle and stores it on data/raw/raw_data.csv
    get_data_from_source()
