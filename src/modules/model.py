from typing import Any, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from src import __version__ as _version
from src.config.config import MODEL_REPOSITORY, config
from src.logger import get_logger

MODULE_NAME = "model"


def load_model(file_name: str) -> Any:
    """Loads a saved model.

    Args:
        file_name: file_name of the model to be un-pickled

    Returns:
        Un-pickled model object.
    """
    method_name = "load_model"
    logger_name = MODULE_NAME + "_" + method_name
    logger = get_logger(logger_name)

    try:
        file_path = MODEL_REPOSITORY / file_name
        trained_model = joblib.load(filename=file_path)
    except Exception as e:
        logger.exception(e)

    return trained_model


def save_model(model_to_save: str) -> Any:
    """Persists the model.

    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called.

    Args:
        model_to_save: The model object to be persisted.

    Returns:
        None
    """
    method_name = "save_model"
    logger_name = MODULE_NAME + "_" + method_name
    logger = get_logger(logger_name)

    # Prepare versioned save file name
    try:
        save_file_name = f"{config.app_config.model_file_name}{_version}.pkl"
        save_path = MODEL_REPOSITORY / save_file_name

        remove_old_models(files_to_keep=[save_file_name])
        joblib.dump(model_to_save, save_path)
    except Exception as e:
        logger.exception(e)


def remove_old_models(*, files_to_keep: List[str]) -> None:
    """Removes old model models.

    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.

    Args:
        files_to_keep: The persisted model files to keep in 
        the repository

    Returns:
        None
    """
    method_name = "remove_old_models"
    logger_name = MODULE_NAME + "_" + method_name
    logger = get_logger(logger_name)

    try:
        do_not_delete = files_to_keep + ["__init__.py"]
        for model_file in MODEL_REPOSITORY.iterdir():
            if model_file.name not in do_not_delete:
                model_file.unlink()
    except Exception as e:
        logger.exception(e)


class CourseRecommender:
    """Course Recommendation model class.

    This creates the Course Recommendation model pipeline.

    Attributes:
        config: the configuration dictionary

        numeric_pipeline: the pipeline for numeric feature
        imputation and scaling

        categorical_pipeline: the pipeline for categorical
        feature imputation and one-hot encoding

        predictor: column transformer object combining the
        numeric and categorical pipelines

        model: The sklearn pipeline for preprocessing the
        predictors and LogisticRegression

        target_encoder: 
    """

    def __init__(self, config) -> None:
        """Initializes DataLoader with member attributes.

        Args:
            config: the configuration dictionary

        Returns: 
            None
        """
        self.config = config

        self.numeric_pipeline = Pipeline(
            steps=[
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', MinMaxScaler())
            ]
        )

        self.categorical_pipeline = Pipeline(
            steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]
        )

        self.predictor = ColumnTransformer(
            transformers=[
                ('number', self.numeric_pipeline,
                 self.config.model_config.numerical_vars),
                ('category', self.categorical_pipeline,
                 self.config.model_config.categorical_vars)
            ]
        )

        self.model = Pipeline(
            steps=[
                ('preprocess', self.predictor),
                ('model', LogisticRegression())
            ]
        )

    def fit(self, X, y) -> None:
        """Fits the model.

        Fits a Label encoder on the target variable and assigns
        to target_encoder attribute. Fits the model pipeline on
        the input data (X , y).

        Args:
            X: the predictor variables
            y: the target variable

        Returns: 
            None
        """
        method_name = "fit"
        logger_name = MODULE_NAME + "_" + method_name
        logger = get_logger(logger_name)

        try:
            # Label encode target feature
            le = LabelEncoder()
            y = le.fit_transform(y)

            # Assign target encoing to member variable
            self.target_encoder = le
        except Exception as e:
            logger.exception(e)

        try:
            # fit the model
            self.model.fit(X, y)
        except Exception as e:
            logger.exception(e)

    def predict(self, X) -> pd.DataFrame:
        """Predicts using the trained model.

        Args:
            X: the predictor variables

        Returns: 
            y_pred: array of predicted values
        """
        return self.model.predict(X)
