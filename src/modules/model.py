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


def load_model(file_name: str) -> Any:
    """
    Loads a saved model.
    """
    file_path = MODEL_REPOSITORY / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def save_model(model_to_save: str) -> Any:
    """
    Persists the model.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called.
    """
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.model_file_name}{_version}.pkl"
    save_path = MODEL_REPOSITORY / save_file_name

    remove_old_models(files_to_keep=[save_file_name])
    joblib.dump(model_to_save, save_path)


def remove_old_models(*, files_to_keep: List[str]) -> None:
    """
    Removes old model models.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in MODEL_REPOSITORY.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


class CourseRecommender:
    """
    Course Recommendation model class.

    """

    def __init__(self, config) -> None:

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

        # Label encode target feature
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Assign target encoing to member variable
        self.target_encoder = le

        # fit the model
        self.model.fit(X, y)

    def predict(self, X) -> pd.DataFrame:

        return self.model.predict(X)
