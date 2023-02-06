from typing import List, Union

from flask import Flask, jsonify, request

from src import __version__ as _version
from src.config.config import config
from src.logger import get_logger
from src.modules.dataloader import DataLoader
from src.modules.model import load_model
from src.modules.validate import validate_inputs

app = Flask(config.app_config.package_name)

model_file_name = f"{config.app_config.model_file_name}{_version}.pkl"
model = load_model(model_file_name)

MODULE_NAME = "infer"


@app.route('/predict', methods=['POST'])
def get_predictions() -> dict:
    """Make a prediction using a saved model.

    Args:
        None

    Returns:
        JSON response from the prediction
    """
    method_name = "get_predictions"
    logger_name = MODULE_NAME + "_" + method_name
    logger = get_logger(logger_name)

    try:
        dataloader = DataLoader(config)
        data = dataloader.get_prediction_data(request)

        validated_data, errors = validate_inputs(input_data=data)
        results = {"predictions": None, "version": _version, "errors": errors}

        if not errors:
            predictions = model.predict(
                X=validated_data[config.model_config.features]
            )
            predictions = model.target_encoder.inverse_transform(predictions)
            results = {
                # type: ignore
                "predictions": [pred for pred in predictions],
                "version": _version,
                "errors": errors,
            }
    except Exception as e:
        logger.exception(e)

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
