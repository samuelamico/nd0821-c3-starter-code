# Script to train machine learning model.
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import logging
import pickle
from sklearn.metrics import mean_absolute_error
import os
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

'''
Author: Samuel Amico

Goal of this script is to pull the census cleaned dataset and apply a ML inference,
we have to split the data using K-Fold / Validation set and training the model
'''

# Add the necessary imports for the starter code.

# Add code to load in the data.
census_cleaned_df = pd.read_csv('../data/census_clean.csv')

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

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
pickle.dump(lb, open('../model/lb.pkl', "wb"))
pickle.dump(encoder, open('../model/encoder.pkl', 'wb'))

# Train and save a model.
logger.info("Train and Save Models for Training Data")
gb_model, randomforest = train_model(X_train, y_train)
y_train_pred_gb = inference(gb_model, X_train)
y_train_pred_rf = inference(randomforest, X_train)

logger.info("Metrcis for GB model")
_, _, _ = compute_model_metrics(y_train, y_train_pred_gb)

logger.info("Metrcis for RF model")
r_squared = randomforest.score(X_train, y_train)
mae = mean_absolute_error(y_train, y_train_pred_rf)
logger.info(f"R_squared = {r_squared} --- MAE = {mae}")

logger.info("Train and Save Models for Test Data")
y_test_pred_gb = inference(gb_model, X_test)
y_test_pred_rf = inference(randomforest, X_test)

logger.info("Metrcis for GB model")
test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, y_test_pred_gb)

logger.info("Metrcis for RF model")
r_squared = randomforest.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_test_pred_gb)
logger.info(f"R_squared = {r_squared} --- MAE = {mae}")

# performance testing using model slicing
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

dataframe = pd.DataFrame(columns=["feature", "value", "precision", "recall", "fbeta_score"])
for feature in cat_features:
    for value in data[feature].unique():
        subset_data = data[data[feature] == value]
        X_test_subset, y_test_subset, encoder, lb = process_data(
            subset_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        y_pred_subset = gb_model.predict(X_test_subset)
        precision, recall, fbeta = compute_model_metrics(y_test_subset, y_pred_subset)
        dataframe = dataframe.append({"feature": feature, "value":value, "precision": precision, "recall": recall, "fbeta_score": fbeta}, ignore_index = True)
dataframe.to_csv('slicing_columns_test.csv',index=False)