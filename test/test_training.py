import pandas as pd
from src.config.config import config
from train import get_model_metrics, train_model

from src.modules.model import load_model
from src.config.config import config
from src import __version__ as _version

model_file_name = f"{config.app_config.model_file_name}{_version}.pkl"
model = load_model(model_file_name)


def test_get_model_metrics():

    X = pd.DataFrame(
        {
            "gender": ["male"],
            "stream": ["science"],
            "subject": ["physics"],
            "marks": [99],
        }
    )
    y = pd.Series(["btech"])

    result = get_model_metrics(model, X, y)

    # Test the full data load
    assert result['Accuracy Score'] == 1.0
    assert result['Weighted F1 score'] == 1.0
