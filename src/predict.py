from logging import getLogger
import warnings
import os
from mlflow.sklearn import load_model

from src.hdd_preprocessing import load_preprocess_testdata

# from src.hdd_feature_engineering import hdd_preprocessor

warnings.filterwarnings("ignore")
logger = getLogger(__name__)


def __get_data():
    """Load and preprocess sample test dataset.

    Returns:
        _type_: X_test
    """
    X_test = load_preprocess_testdata(
        days=30, filename="ST4000DM000_history_total", path=os.getcwd()
    )
    return X_test


def __get_model():
    """Load the saved model for prediction.

    Returns:
        _type_: Model
    """
    model_path = "models/deployment_xgb"
    model = load_model(model_path)
    return model


def run_predict(X_test):
    """Load the test data, preprocess it and run the prediction.

    Returns:
        _type_: Predicted targets
    """
    logger.info("Loading model")
    model = __get_model()
    logger.info("Loading and preprocessing data")
    # X_test = __get_data()
    # logger.info("Feature engineering on test")
    # preprocessor = hdd_preprocessor(days=30, trigger=0.05)
    # X_test = preprocessor.fit_transform(X_test)  # Nothing saved in the fit
    logger.info("Prediction in progress")
    y_proba = model.predict_proba(X_test)
    return y_proba > 0.501


if __name__ == "__main__":
    import logging
    import pandas as pd

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    # avoid excessive logs
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)
    logger.setLevel(logging.INFO)
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_pred = run_predict(X_test)
    print(y_pred)
    pd.Series(y_pred[:, 0]).to_csv("data/processed/y_pred.csv", index=False)
