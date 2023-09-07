import json
import pathlib
import warnings
from typing import Any, Literal, TypedDict

import tomli

from ..typing import FilePath
from .dataset import DatasetConfig


class ModelConfig(TypedDict, total=True):
    family: Literal["Llama2", "Platypus2"]
    size: Literal["7B", "13B"]


class Config(TypedDict, total=False):
    project_name: str
    exp_name: str
    model: ModelConfig
    dataset: DatasetConfig
    trainer_state: dict[str, Any]


def get_config(config_path: FilePath) -> Config:
    with open(config_path, "rb") as f:
        if pathlib.Path(config_path).suffix == ".json":
            config = json.load(f)
        else:
            config = tomli.load(f)

    if "prompt_id" in config.keys():
        warnings.warn("prompt_id should be in dataset key", DeprecationWarning)
        config["dataset"]["prompt_id"] = config.pop("prompt_id")

    return config


def get_checkpoint_path(config: Config):
    return (
        pathlib.Path("models")
        / config["model"]["family"].lower()
        / config["project_name"]
        / config["model"]["size"].lower()
        / config["exp_name"]
    )


def get_config_from_checkpoint(ckpt_path: FilePath, drop_log_history: bool = True) -> Config:
    ckpt_path = pathlib.Path(ckpt_path)
    config = get_config(ckpt_path / "train_config.json")

    if (ckpt_path / "trainer_state.json").exists():
        with open(ckpt_path / "trainer_state.json") as f:
            config["trainer_state"] = json.load(f)
            if drop_log_history:
                config["trainer_state"].pop("log_history")

    return config
