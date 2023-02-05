from infer import get_predictions


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = "btech"
    expected_no_predictions = 49

    # When
    result = get_predictions(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], object)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert predictions[0] == expected_first_prediction_value
