"""
This module is created to validate some data
properties that we must ensure in order to
expect the desired behaviour of these when
giving them as input to the model. The module
is implemented using great_expectations
"""

import great_expectations as gx
import pandas as pd
from src import PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH

if __name__ == '__main__':
    # initialize the Great Expectations context
    context = gx.get_context()

    # add or update an expectation suite
    context.add_or_update_expectation_suite("reviews_training_suite")

    # define a data source
    datasource = context.sources.add_or_update_pandas(name="reviews_dataset")

    # load processed training and testing data
    train = pd.read_csv(PROCESSED_TRAIN_DATA_PATH / "train_data.csv")
    test = pd.read_csv(PROCESSED_TEST_DATA_PATH / "test_data.csv")

    # concatenate data to test all together
    processed_data = pd.concat([train, test], axis=0)

    # add the data to the datasource
    data_asset = datasource.add_dataframe_asset(
        name="processed", dataframe=processed_data
    )

    # built a batch request
    batch_request = data_asset.build_batch_request()

    # get a validator based on the batch request
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="reviews_training_suite",
        datasource_name="reviews_dataset",
        data_asset_name="processed",
    )

    # check if main three columns exist
    validator.expect_column_to_exist("Review Text")
    validator.expect_column_to_exist("Top Product")
    validator.expect_column_to_exist("Stemmed Review Text")

    # check if they follow a determinate order
    validator.expect_table_columns_to_match_ordered_list(
        column_list=[
            "Review Text",
            "Top Product",
            "Stemmed Review Text"
        ]
    )

    # check that the target value is not null, integer and 0 or 1
    validator.expect_column_values_to_not_be_null("Top Product")
    validator.expect_column_values_to_be_of_type("Top Product", "int64")
    validator.expect_column_values_to_be_between(
        "Top Product", min_value=0, max_value=1
    )

    # check that reviews are not null strings
    validator.expect_column_values_to_not_be_null("Review Text")
    validator.expect_column_values_to_be_of_type("Review Text", "str")

    # check that stemmed reviews are not null strings
    validator.expect_column_values_to_not_be_null("Stemmed Review Text")
    validator.expect_column_values_to_be_of_type(
        "Stemmed Review Text", "str"
    )

    # save the expectation suite
    validator.save_expectation_suite(discard_failed_expectations=False)

    # define a checkpoint
    checkpoint = context.add_or_update_checkpoint(
        name="reviews_checkpoint",
        validator=validator,
    )

    # run the checkpoint and view the validation result
    checkpoint_result = checkpoint.run()
    context.view_validation_result(checkpoint_result)
