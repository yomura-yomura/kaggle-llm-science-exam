import json
import pathlib
from typing import Any, TypedDict

from ..typing import FilePath
from .dataset import DatasetConfig


class Config(TypedDict, total=False):
    dataset: DatasetConfig
    prompt_id: int
    trainer_state: dict[str, Any]


def get_config(ckpt_path: FilePath, drop_log_history: bool = True) -> Config:
    ckpt_path = pathlib.Path(ckpt_path)
    with open(ckpt_path / "train_config.json") as f:
        config = json.load(f)
    if (ckpt_path / "trainer_state.json").exists():
        with open(ckpt_path / "trainer_state.json") as f:
            config["trainer_state"] = json.load(f)
            if drop_log_history:
                config["trainer_state"].pop("log_history")

    return config
