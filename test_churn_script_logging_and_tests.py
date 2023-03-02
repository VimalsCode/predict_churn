"""
Unit test to test the churn_library.
author: Vimal
Date: 02.March.2023
"""

import os
import logging
import pytest
import churn_library as clib

logging.basicConfig(
    filename='./logs/churn_library_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

test_csv = ["./data/bank_data.csv"]
test_path = ["./images/eda/"]
test_cat_columns = [[
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'], [
    'Test_Gender',
    'Test_Education_Level']]


@pytest.fixture(scope="module")
def path():
    '''
    To return the path to the input data frame
    '''
    return "./data/bank_data.csv"


@pytest.fixture(scope="module")
def prepare_train():
    '''
    To prepare the data for training and return the train and test data
    '''
    data = pytest.df
    return clib.perform_feature_engineering(data, "Churn")


@pytest.mark.parametrize("pth", test_csv)
def test_import_data(pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        bank_data_df = clib.import_data(pth)
        logging.info("Testing import_data: SUCCESS")
        pytest.df = bank_data_df
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err
    try:
        assert bank_data_df.shape[0] > 0
        assert bank_data_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.mark.parametrize("image_pth", test_path)
def test_eda(image_pth):
    '''
    test perform eda function
    '''
    bank_data_df = pytest.df
    try:
        clib.perform_eda(bank_data_df, image_pth)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: FAILED")
        raise err


@pytest.mark.parametrize("cat_cols", test_cat_columns)
def test_encoder_helper(cat_cols):
    '''
    test encoder helper
    '''
    bank_data_df = pytest.df
    try:
        encoded_df = clib.encoder_helper(bank_data_df, cat_cols, "Churn")
        assert len(encoded_df.columns.tolist()) == 28
        logging.info("Testing encoder_helper: SUCCESS")
    except KeyError:
        logging.error(
            "Testing encoder_helper: provided category column is not available in dataframe")
    except Exception as err:
        logging.error("Testing encoder_helper: FAILED")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    bank_data_df = pytest.df
    try:
        X_train, X_test, y_train, y_test = clib.perform_feature_engineering(
            bank_data_df, "Churn")
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_feature_engineering: FAILED")
        raise err


def test_train_models(prepare_train):
    '''
    test train_models
    '''
    try:
        clib.train_models(*prepare_train)
        logging.info("Testing train_models: SUCCESS")
        assert os.path.exists('./models/logistic_model.pkl')
        assert os.path.exists('./models/rfc_model.pkl')
        logging.info("Testing save best models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: FAILED")
        raise err


if __name__ == "__main__":
    pass
