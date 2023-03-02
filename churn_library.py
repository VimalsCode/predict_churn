# library doc string
"""
Module to predict customer churn based on the provided customer details.
author: Vimal
Date: 27.Feb.2023
"""

# import libraries
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='./logs/churn_library_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            bank_data_df: bank data dataframe
    '''
    try:
        bank_data_df = pd.read_csv(pth)
        logging.info(
            'SUCCESS: sucessfully loaded data file %s from the specified path',
            pth)
    except FileNotFoundError as err:
        logging.error(
            'ERROR: data file %s is not found in the specified path', pth)
        raise err
    return bank_data_df


def perform_eda(eda_df, plot_pth):
    '''
    perform eda on df and save figures to images folder
    input:
            eda_df: bank data dataframe

    output:
            None
    '''

    logging.info('INFO: starting eda and plot visualization')
    # create label
    eda_df['Churn'] = eda_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    logging.debug(
        'DEBUG: available churn information in dataframe : %s',
        eda_df['Churn'].unique())
    # univariate plot - attrition flag column
    plt.figure(figsize=(25, 10))
    plt.hist(eda_df['Churn'])
    plt.ylabel('customer frequency')
    plt.xlabel('customer attrition')
    plt.title('Churn distribution')
    plt.savefig(plot_pth + 'churn_distribution.png')
    plt.close()

    # univariate plot - customer age column
    plt.figure(figsize=(20, 10))
    plt.hist(eda_df['Customer_Age'], edgecolor='black')
    plt.ylabel('customer age frequency')
    plt.xlabel('customer age')
    plt.title('Customer age distribution')
    plt.savefig(plot_pth + 'customer_age_distribution.png')
    plt.cla()
    plt.clf()
    plt.close()

    # univariate plot - marital status
    marital_status = eda_df['Marital_Status'].value_counts('normalize')
    logging.debug(
        'DEBUG: marital status in dataframe : %s',
        marital_status.to_dict)
    plt.figure(figsize=(20, 10))
    marital_status.plot(kind='bar')
    #plt.plot(list(marital_status.keys), list(marital_status.values))
    plt.ylabel('Number of occurences')
    plt.xlabel('marital status')
    plt.title('Customer Marital status distribution')
    plt.savefig(plot_pth + 'marital_status_distribution.png')
    plt.cla()
    plt.clf()
    plt.close()

    # univariate plot - total transaction
    plt.figure(figsize=(20, 10))
    sns.histplot(eda_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(plot_pth + 'total_trans.png')
    plt.cla()
    plt.clf()
    plt.close()
    # multivariate plot - heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(eda_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(plot_pth + 'heatmap.png')
    plt.cla()
    plt.clf()
    plt.close()
    logging.info(
        'SUCCESS: completed successfully the eda and plot visualization')


def encoder_helper(encoder_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
            encoder_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            encoder_df: pandas dataframe with new columns for
    '''
    # encoded column
    logging.info(
        'INFO: starting column encoding with size: %s',
        len(category_lst))
    for category in category_lst:
        category_column_lst = []
        category_column_groups = encoder_df.groupby(category).mean()['Churn']

        for val in encoder_df[category]:
            category_column_lst.append(category_column_groups.loc[val])

        if response:
            # naming variable is available
            encoder_df[category + '_' + response] = category_column_lst
        else:
            encoder_df[category] = category_column_lst
    logging.info('INFO: sucessfully completed comlun encoding')
    return encoder_df


def perform_feature_engineering(feature_engg_df, response):
    '''
    input:
              feature_engg_df: pandas dataframe
              response: string of response name [optional argument that could be used for
              naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    logging.info(
        'INFO: starting feature engineering for the dataframe: %s',
        feature_engg_df.shape)
    # available category column from the input dataframe
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    # perform column encoding - original df 23 columns
    encoded_df = encoder_helper(feature_engg_df, cat_columns, response)
    # target feature
    y = encoded_df['Churn']
    # new dataframe
    X = pd.DataFrame()
    # retain 19 imp feature columns for training
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    # retain the feature dataframe
    X[keep_cols] = encoded_df[keep_cols]
    logging.info(
        'INFO: completed feature engineering and the new feature column size is: %s',
        X.shape)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return (X_train, X_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as
    image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    logging.info(
        "INFO:generating classification report for training and testing results")
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        './images/results/classification_report_random_forest.png',
        bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        './images/results/classification_report_logistic_regression.png',
        bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()
    logging.info(
        "SUCCESS:generated classification report for training and testing results")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    logging.info("INFO:generating feature importances for training data")
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, bbox_inches='tight')

    logging.info(
        "SUCCESS:generated feature importances plot for training data")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    logging.info('INFO: starting model training')
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # perform predict
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    logging.info('SUCCESS: model training completed')
    # save the best model
    logging.info("INFO:Saving the random forest trained model")
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    logging.info("SUCESS:Saved the random forest trained model")

    logging.info("INFO:Saving the logistic regression trained model")
    joblib.dump(lrc, './models/logistic_model.pkl')
    logging.info("SUCESS:Saved the logistic regression trained model")

    # classification report image
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    # roc plot
    roc_plot(X_test, y_test)
    # feature importance plot
    feature_importance_plot(cv_rfc, pd.concat(
        [X_train, X_test]), './images/results/feature_importance.png')


def roc_plot(X_test, y_test):
    '''
    create and store roc plots
    input:
              X_test: X testing data
              y_test: y testing data
    output:
              None
    '''
    # load the model
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    # plots
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    current_ax = plt.gca()
    rfc_disp = plot_roc_curve(
        rfc_model,
        X_test,
        y_test,
        ax=current_ax,
        alpha=0.8)
    lrc_plot.plot(ax=current_ax, alpha=0.8)
    plt.savefig('./images/results/roc_results.png')
    plt.cla()
    plt.clf()
    plt.close()
    logging.info(
        "SUCESS:generated receiver operating characteristic(roc) plot")


if __name__ == "__main__":
    # location where the input source data is available
    SOURCE_DATA_PATH = './data/bank_data.csv'
    EDA_PLOT_PATH = './images/eda/'
    logging.info(
        "INFO:starting the training pipeline to predict customers that are most likely to churn")
    # import data
    source_df = import_data(SOURCE_DATA_PATH)
    # perform eda
    perform_eda(source_df, EDA_PLOT_PATH)
    # perform feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        source_df, response='Churn')
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
    logging.info(
        "SUCCESS:completed the model training to predict customers that are most likely to churn")
