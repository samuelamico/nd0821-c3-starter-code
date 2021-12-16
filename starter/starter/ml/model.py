from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import logging
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    gradient_parameters = {"n_estimators": (5, 10),
                  "learning_rate": (0.1, 0.01),
                  "max_depth": [2, 3],
                  "max_features": ("auto", "log2")}
    random_parametes = {"n_estimators": 100,
                    "max_depth": 15,
                    "min_samples_split": 4,
                    "min_samples_leaf": 3}

    randomforest = RandomForestRegressor(max_depth=15, min_samples_split=4, min_samples_leaf=3,random_state=0)
    gbc = GradientBoostingClassifier(random_state=42)

    logger.info("Traning Random Forest model")
    randomforest.fit(X_train,y_train)
    logger.info("Traning Gradient Boosting model")
    clf = GridSearchCV(gbc, gradient_parameters)
    clf.fit(X_train, y_train)

    logger.info("Exporting model")
    grandient_filepath = "../model/gbclassifier.pkl"
    with open(grandient_filepath, 'wb') as file:
        pickle.dump(clf.best_estimator_, file)
    gb_model = clf.best_estimator_

    random_filepath = "../model/randomforest.pkl"
    with open(random_filepath, 'wb') as file:
        pickle.dump(randomforest, file)

    return gb_model, randomforest


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    logger.info(f"fbeta : {fbeta} precision : {precision} recall : {recall}")
    return precision, recall, fbeta




def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds
