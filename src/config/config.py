from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import src

# Project Directories
PACKAGE_ROOT = Path(src.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_REPOSITORY = PACKAGE_ROOT / "config.yaml"
DATASET_REPOSITORY = PACKAGE_ROOT / "datasets"
MODEL_REPOSITORY = PACKAGE_ROOT / "models"


class AppConfig(BaseModel):
    """Application-level config."""

    package_name: str
    train_data: str
    test_data: str
    model_file_name: str


class ModelConfig(BaseModel):
    """All configuration relevant to model training."""

    target: str
    features: List[str]
    test_size: float
    random_state: int
    numerical_vars: Sequence[str]
    categorical_vars: Sequence[str]


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_REPOSITORY.is_file():
        return CONFIG_REPOSITORY
    raise Exception(f"Config not found at {CONFIG_REPOSITORY!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
