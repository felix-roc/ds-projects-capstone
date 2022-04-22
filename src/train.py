from logging import getLogger
import warnings
import os
from mlflow.sklearn import save_model

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from src.hdd_preprocessing import load_preprocess_data, train_test_splitter
from src.hdd_feature_engineering import hdd_preprocessor, log_transformer

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

RSEED = 42

warnings.filterwarnings("ignore")
logger = getLogger(__name__)


def __create_ann_model__(input_dim=19):
    """Build function to construct the artificial neural network. This function is
    needed by the KerasWrapper.

    Args:
        input_dim (int, optional): Input dimension of the first layer (number
        of features). Defaults to 19.

    Returns:
        _type_: The model
    """
    # initiate the instance
    model = Sequential()
    # layers
    model.add(Dense(units=30,
                    kernel_initializer='uniform',
                    activation='relu',
                    input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(units=30,
                    kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))
    # compiling the ANN
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['Recall', 'Precision'])
    return model


def __get_data():
    """Load and preprocess the data for the modeling. The data is loaded and
    split into train and test datasets. Afterward, we preprocess the data and
    create the features for both datasets.

    Returns:
        _type_: Train and test datasets.
    """
    logger.info("Loading and preprocessing data")
    X, y = load_preprocess_data(
        days=30,
        filename="ST4000DM000_history_total",
        path=os.getcwd()
        )
    logger.info("Train-test splitting")
    X_train, X_test, y_train, y_test = train_test_splitter(
        X, y, test_size=0.30, random_state=RSEED
        )
    logger.info("Feature engineering on train")
    # Create instance of our preprocessor
    preprocessor = hdd_preprocessor(days=30, trigger=0.05)
    X_train = preprocessor.fit_transform(X_train)
    logger.info("Feature engineering on test")
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test


def run_training():
    """Load the data, construct the model and fit it. Save the model for
    deployment.
    """
    logger.info("Getting the data")
    X_train, X_test, y_train, y_test = __get_data()
    logger.info("Training")
    # Scaling pipeline
    scaling_pipe = Pipeline([
        ('scaler_log', log_transformer(offset=1)),
        ('scaler_minmax ', MinMaxScaler()),
        ])
    # ANN model, wrapped for use in sklearn
    ann_classifier = KerasClassifier(
        build_fn=__create_ann_model__,
        epochs=150,
        batch_size=40000,
        class_weight={0: 1.0, 1: 0.4*len(y_train)/y_train.sum()},
        verbose=0,
        )
    # specify the model type
    ann_classifier._estimator_type = "classifier"
    # XGBoost model
    estimators = [
        ('xgb', XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=0.4*len(y_train)/y_train.sum(),
            colsample_bytree=0.4,
            subsample=0.3,
            eta=0.01,
            gamma=1,
            max_depth=6,
            n_estimators=50,
            min_child_weight=2,
            reg_lambda=0.7,
            reg_alpha=1,
            use_label_encoder=False,
            )),
        ('ann', ann_classifier),
        ]
    # Stacking
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            class_weight=0.4*len(y_train)/y_train.sum()),
        n_jobs=-1)
    # Include the scaling pipeline
    model = Pipeline([
        ('scaling', scaling_pipe),
        ('stacking', clf),
        ])
    logger.info("Fitting in progress")
    model.fit(X_train, y_train)
    # logger.info("Pickle")
    # filename = 'deployment.bin'
    # with open(filename, 'wb') as file_out:
    #     pickle.dump(model, file_out)
    # saving the model
    logger.info("Saving model in the models folder")
    path = "models/deployment_stacked"
    save_model(sk_model=clf, path=path)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    # avoid excessive logs
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)
    logger.setLevel(logging.INFO)

    run_training()
