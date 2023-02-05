import pytest

from src.config.config import config
from src.modules.dataloader import DataLoader


@pytest.fixture()
def sample_input_data():
    dataloader = DataLoader(config)
    return dataloader.get_data()