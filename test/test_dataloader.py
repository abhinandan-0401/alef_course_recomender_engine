from src.config.config import config
from src.modules.dataloader import DataLoader


def test_dataloader(sample_input_data):
    dataloader = DataLoader(config)
    df = dataloader.get_training_data()

    # Test the full data load
    assert df.shape[0] == 49
    assert df.shape[1] == 5
