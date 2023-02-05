from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.config.config import DATASET_REPOSITORY


class DataLoader:
    """
    Data loader class

    """

    def __init__(self, config) -> None:
        self.config = config

    # read training data
    def get_data(self,) -> pd.DataFrame:
        df = pd.read_csv(
            Path(f"{DATASET_REPOSITORY}/{self.config.app_config.train_data}"))
        return df
