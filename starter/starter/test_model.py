import os
import pickle
from numpy import dtype
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from starter.starter.ml.data import process_data
from starter.starter.ml.model import compute_model_metrics, inference, train_model


@pytest.fixture(scope='session')
def get_files():

    data_dir = os.getcwd()

    data = pd.read_csv("../data/census_cleaned.csv")

    gb_model = os.path.join(data_dir, "../model/gbclassifier.pkl")
    with open(gb_model, "rb") as f:
        gb_model = pickle.load(f)

    rf_model = os.path.join(data_dir, "../model/randomforest.pkl")
    with open(rf_model, "rb") as f:
        rf_model = pickle.load(f)

    encoder = os.path.join(data_dir, "../model/encoder.pkl")
    with open(encoder, "rb") as f:
        encoder = pickle.load(f)

    lb = os.path.join(data_dir, "../model/lb.pkl")
    with open(lb, "rb") as f:
        lb = pickle.load(f)

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
    y_train_pred_gb = inference(gb_model, X_train)

    return y_train_pred_rf, y_train_pred_gb


def test_rf_inference_type(get_inference):
    y_train_pred_rf, y_train_pred_gb = get_inference
    
    assert y_train_pred_rf.dtpye == dtype(object)

def test_gb_inference_type(get_inference):
    y_train_pred_rf, y_train_pred_gb = get_inference
    
    assert y_train_pred_gb.dtpye == dtype(object)

def test_rf_metrics(get_inference):
    y_train_pred_rf, y_train_pred_gb = get_inference
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

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    
    r_squared = rf_model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_train_pred_rf)

    assert mae >= 0.15


def test_train_model(get_files):
    data_dir = os.getcwd()
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
    filepath = os.path.join(data_dir, "../model/gbclassifier_test.pkl")
    model = train_model(X_train, y_train, filepath=filepath)

    assert os.path.exists(filepath)
    return X_train, y_train, model, encoder, lb

