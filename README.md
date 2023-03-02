# Predict Customer Churn
The project **Predict Customer Churn** is delivered to demonstrate clean code principles as part of ML DevOps Engineer Nanodegree Udacity.
The main scope of the project is to implement a machine learning algorithm to identify credit card customers that are likley to churn.

## Project Description
The main objective of the implementation is to demonstrate clean code principles with focus on testing, logging  and coding best practices. The project structure is based on the following approach,
- Load the CSV data and perform Exploratory data analysis
- Prepare the classification label based on the given input features
- perform feature engineering to prepare the data for model training
- Train classification model based on Random forest and logistic regression and save the best model
- Identify the key input features affecting the classification

## Files and data description
Overview of the files and data present in the root directory. 

### Source Files
|File name                             | Description                                                  |
|--------------------------------------|--------------------------------------------------------------|
|churn_library.py                      | Module containing the functions to train classification model|
|test_churn_script_logging_and_tests.py| Unit test file to test the churn_library module              |
|conftest.py                           | configuration to be used during the test execution           |
|pytest.ini                            | pytest configuration                                         |

### Output folder structure
|Folder name                           | Description                                                |
|--------------------------------------|------------------------------------------------------------|
|logs                                  | log information about the library module and test execution|
|models                                | To store the trained models                                |
|images                                | Location to save eda and trained model results plots       |


## Running Files
Setup the environment and and install dependencies based on the provided requirments_py3.6.txt
* create a virtual environment using the following command
```
python -m venv <env-name>
```
* Activate the virtual environment
```
source <env-name>/bin/activate
```

* Install the required dependencies
```
pip3 install -r requirments_py3.6.txt
```

* To run the churn library
```
python churn_library.py
```

* To test the churn library
```
pytest
```

* additionally to meet pep8 standard
```
autopep8 --in-place --aggressive --aggressive test_churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```

* additionally to perform code analysis
```
pylint churn_library.py
pylint test_churn_script_logging_and_tests.py
```