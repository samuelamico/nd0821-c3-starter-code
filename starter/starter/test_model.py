import os
import pickle
from numpy import dtype
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model
import numpy as np

@pytest.fixture(scope='session')
def get_files():

    data_dir = os.getcwd()
    print(data_dir)
    gb_model = pickle.load(open(f"{data_dir}/starter/model/gbclassifier.pkl", "rb"))
    rf_model = pickle.load(open(f"{data_dir}/starter/model/randomforest.pkl", "rb"))
    encoder = pickle.load(open(f"{data_dir}/starter/model/encoder.pkl", "rb"))
    lb = pickle.load(open(f"{data_dir}/starter/model/lb.pkl", "rb"))

    census = f"{data_dir}/starter/data/census_clean.csv"
    data = pd.read_csv(census)

    train, test = train_test_split(data, test_size=0.20, random_state=42)

    return data, gb_model, rf_model, encoder, lb, train, test

@pytest.fixture(scope='session')
def get_inference(get_files):
    data, gb_model, rf_model, encoder, lb, train, test = get_files

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=encoder,
        lb=lb,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_train_pred_rf = inference(rf_model, X_train)
    #y_train_pred_gb = inference(gb_model, X_train)

    return y_train_pred_rf


def test_rf_inference_type(get_inference):
    y_train_pred_rf = get_inference
    
    assert y_train_pred_rf.dtype == np.dtype('float64')

def test_gb_inference_type(get_inference):
    y_train_pred_rf = get_inference
    
    assert y_train_pred_rf.dtype == np.dtype('float64')

def test_rf_metrics(get_inference):
    data_dir = os.getcwd()
    census = f"{data_dir}/starter/data/census_clean.csv"
    census_cleaned_df = pd.read_csv(census)
    y_train_pred_rf = get_inference
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(census_cleaned_df, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    
    #r_squared = rf_model.score(X_test, y_test)
    mae = mean_absolute_error(y_train, y_train_pred_rf)

    assert mae >= 0.15


def test_train_model():
    '''
    
    data, gb_model, rf_model, encoder, lb, train, test = get_files
    train, test = train_test_split(data, test_size=0.20)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    '''
    data_dir = os.getcwd()
    filepath = f"{data_dir}/starter/model/gbclassifier.pkl"
    #model = train_model(X_train, y_train)

    assert os.path.exists(filepath)

