import json
import pathlib
import warnings
from typing import Any, Literal, TypedDict

import tomli

from .. import deberta, llama2
from ..typing import FilePath
from .dataset import DatasetConfig


class Config(TypedDict, total=False):
    project_name: str
    exp_name: str
    model: llama2.model.ModelConfig | deberta.model.ModelConfig
    dataset: DatasetConfig
    train: "TrainConfig"
    trainer_state: dict[str, Any]


class TrainConfig(TypedDict):
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int

    learning_rate: float
    weight_decay: float

    logging_steps: int
    save_steps: int
    save_total_limit: int

    lr_scheduler_type: Literal["constant", "cosine"]
    warmup_steps: int

    num_train_epochs: int

    early_stopping_patience: int

    lora: "TrainLoraConfig"


class TrainLoraConfig(TypedDict):
    r: int
    alpha: int
    dropout: float


def get_config(config_path: FilePath) -> Config:
    with open(config_path, "rb") as f:
        if pathlib.Path(config_path).suffix == ".json":
            config = json.load(f)
        else:
            config = tomli.load(f)

    if "prompt_id" in config.keys():
        warnings.warn("prompt_id should be in dataset key", DeprecationWarning)
        config["dataset"]["prompt_id"] = config.pop("prompt_id")

    if "context" not in config["dataset"].keys():
        config["dataset"]["context"] = {}

    if "upper_limit_of_n_words" not in config["dataset"]["context"].keys():
        config["dataset"]["context"]["upper_limit_of_n_words"] = -1

    if "quant_n_bits" not in config["model"]:
        config["model"]["quant_n_bits"] = 4

    return config


def get_checkpoint_path(config: Config):
    if "family" in config["model"]:
        return (
            pathlib.Path("models")
            / config["model"]["family"].lower()
            / config["project_name"]
            / config["model"]["size"].lower()
            / config["exp_name"]
        )
    elif "version" in config["model"]:
        return (
            pathlib.Path("models")
            / f'deberta-{config["model"]["version"]}-{config["model"]["size"]}'.lower()
            / config["project_name"]
            / config["exp_name"]
        )
    else:
        raise NotImplementedError(f"unexpected model config: {config['model']}")


def get_config_from_checkpoint(ckpt_path: FilePath, drop_log_history: bool = True) -> Config:
    ckpt_path = pathlib.Path(ckpt_path)
    if (ckpt_path / "train_config.json").exists():
        config = get_config(ckpt_path / "train_config.json")
    elif (ckpt_path / "train_config.toml").exists():
        config = get_config(ckpt_path / "train_config.toml")
    else:
        raise FileNotFoundError(f"{ckpt_path}/train_config.[json,toml]")

    if (ckpt_path / "trainer_state.json").exists():
        with open(ckpt_path / "trainer_state.json") as f:
            config["trainer_state"] = json.load(f)
            if drop_log_history:
                config["trainer_state"].pop("log_history")

    return config
