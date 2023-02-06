from typing import Any

from flask import Flask, jsonify
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.config.config import config
from src.logger import get_logger
from src.modules.dataloader import DataLoader
from src.modules.model import CourseRecommender, save_model

MODULE_NAME = "train"

app = Flask(config.app_config.package_name)


def get_model_metrics(model: Any, X: Any, y: Any) -> None:
    """Calculates performance metrics of the trained model.

    Args:
        model: the trained model
        X: predictors
        y: target

    Returns:
        metric dictionary with accuracy score and weighted f1-score.
    """
    method_name = "get_model_metrics"
    logger_name = MODULE_NAME + "_" + method_name
    logger = get_logger(logger_name)

    try:
        y_true = model.target_encoder.transform(y)
    except Exception as e:
        logger.exception(e)

    try:
        y_pred = model.predict(X)
    except Exception as e:
        logger.exception(e)

    return {
        "Accuracy Score": accuracy_score(y_true, y_pred),
        "Weighted F1 score": f1_score(y_true, y_pred, average="weighted"),
    }


@app.route('/train', methods=['POST'])
def train_model() -> None:
    """Model training end-point.

    Args:
        None

    Returns:
        JSON payload with the calculated metrics after training.
    """
    method_name = "train_model"
    logger_name = MODULE_NAME + "_" + method_name
    logger = get_logger(logger_name)

    try:
        dataLoader = DataLoader(config)
    except Exception as e:
        logger.exception(e)

    try:
        df = dataLoader.get_training_data()
    except Exception as e:
        logger.exception(e)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df[config.model_config.features],
            df[config.model_config.target],
            test_size=config.model_config.test_size,
            random_state=config.model_config.random_state,
        )
    except Exception as e:
        logger.exception(e)

    # Create the model
    try:
        model = CourseRecommender(config)
    except Exception as e:
        logger.exception(e)

    # Fit the model
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logger.exception(e)

    # Persist the trained model
    try:
        save_model(model)
    except Exception as e:
        logger.exception(e)

    return jsonify(metrics=get_model_metrics(model, X_test, y_test))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
