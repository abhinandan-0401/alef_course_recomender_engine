from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from src.config.config import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""

    validated_data = input_data.copy()
    vars_with_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() != 0
    ]
    validated_data.dropna(subset=vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # drop nan inputs
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans
        MultipleInputs(
            inputs=validated_data.replace(
                {np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class InputSchema(BaseModel):
    gender: Optional[str]
    stream: Optional[str]
    subject: Optional[str]
    marks: Optional[int]
    course: Optional[str]


class MultipleInputs(BaseModel):
    inputs: List[InputSchema]
