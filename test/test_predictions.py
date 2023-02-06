from infer import get_predictions
import requests


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = "btech"
    expected_no_predictions = 49

    # Get the json payload
    payload = {}
    for col in sample_input_data.columns:
        payload[col] = sample_input_data[col].to_list()
    response = requests.api.post(
        "http://192.168.29.5:5000/predict",  # Please check the url accordingly
        json=payload
    )

    # When
    assert response.status_code == 200

    result = response.json()

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], object)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert predictions[0] == expected_first_prediction_value
