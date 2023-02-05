from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

from src.config.config import config
from src.modules.dataloader import DataLoader
from src.modules.model import CourseRecommender, save_model

from app import app


def get_model_metrics(model: Any, X: Any, y: Any) -> None:
    """
    Display the performance metrics of the trained model
    """
    y_true = model.target_encoder.transform(y)
    y_pred = model.predict(X)

    return {
        "Accuracy Score": accuracy_score(y_true, y_pred),
        "Weighted F1 score": f1_score(y_true, y_pred, average="weighted"),
    }


@app.route('/train', methods=['POST'])
def train_model() -> None:
    """
    Train model on the training data.

    """

    # Load the training data
    dataLoader = DataLoader(config)
    df = dataLoader.get_data()

    # Train test split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df[config.model_config.features],  # predictors
        df[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # Create the model
    model = CourseRecommender(config)

    # Fit the model
    model.fit(X_train, y_train)

    # Persist the trained model
    save_model(model)

    # Get performance metrics of the trained model
    return jsonify(metrics=get_model_metrics(model, X_test, y_test))


# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0')
