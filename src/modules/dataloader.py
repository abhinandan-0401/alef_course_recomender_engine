from pathlib import Path
from typing import Any

import pandas as pd
from flask import request

from src.config.config import DATASET_REPOSITORY
from src.logger import get_logger

MODULE_NAME = "dataloader"

class DataLoader:
    """Data loader class.

    This class reads incoming data and returns in desired
    data format.

    Attributes:
        config: the configuration dictionary
    """

    def __init__(self, config) -> None:
        """Initializes DataLoader with config."""
        self.config = config

    def get_training_data(self,) -> pd.DataFrame:
        """Fetch training data.
        
        Reads input data in csv from file path in config
        and returns a dataframe.

        Args:
            None
        
        Returns:
            Pandas Dataframe
        """

        method_name = "get_training_data"
        logger_name = MODULE_NAME + "_" + method_name
        logger = get_logger(logger_name)

        try:
            df = pd.read_csv(
                Path(f"{DATASET_REPOSITORY}/{self.config.app_config.train_data}"))
        except Exception as e:
            logger.exception(e)
        
        return df

    def get_prediction_data(self, request: Any) -> pd.DataFrame:
        """Extract and formats the prediction data.
        
        Reads input data from request and returns a dataframe.

        Args:
            None
        
        Returns:
            Pandas Dataframe
        """

        method_name = "get_prediction_data"
        logger_name = MODULE_NAME + "_" + method_name
        logger = get_logger(logger_name)

        try:
            jsonfile = request.get_json()
            data = pd.DataFrame(jsonfile)
        except Exception as e:
            logger.exception(e)
        
        return data