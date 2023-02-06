import pytest
import requests

from src.config.config import config
from src.modules.dataloader import DataLoader


@pytest.fixture()
def sample_input_data():
    dataloader = DataLoader(config)
    return dataloader.get_training_data()


@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    def stunted_get():
        raise RuntimeError("Network access not allowed during testing!")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: stunted_get())
