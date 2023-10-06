import pathlib
from typing import Literal, TypedDict

from transformers import AutoModelForMultipleChoice, AutoTokenizer, DebertaV2ForMultipleChoice, DebertaV2TokenizerFast

from ...typing import FilePath
from ...utils import timer
from ..model import get_model_kwargs as _get_model_kwargs

ModelVersion = Literal["v3"]
ModelSize = Literal["large"]


class ModelConfig(TypedDict, total=True):
    version: ModelVersion
    size: ModelSize
    quant_n_bits: int


def get_model(
    model_config: ModelConfig, model_name: str | None = None
) -> tuple[DebertaV2ForMultipleChoice, DebertaV2TokenizerFast]:
    if model_name is None:
        model_name = get_model_name(model_config)

    print(f"-- Loading {model_name}")

    # model = AutoModelForMultipleChoice.from_pretrained(model_name, **get_model_kwargs(model_config["quant_n_bits"]))
    model = AutoModelForMultipleChoice.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def get_model_kwargs(quant_n_bits: int | None) -> dict:
    model_kwargs = _get_model_kwargs(quant_n_bits)
    model_kwargs.pop("device_map")
    return model_kwargs


def get_model_name(model_config: ModelConfig) -> str:
    match model_config:
        case {"version": "v3", "size": "large"}:
            return "microsoft/deberta-v3-large"
        case _:
            raise NotImplementedError(f"unexpected model_config: {model_config}")


def get_model_from_checkpoint(
    model_config: ModelConfig, ckpt_path: FilePath
) -> tuple[DebertaV2ForMultipleChoice, DebertaV2TokenizerFast]:
    ckpt_path = pathlib.Path(ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.pad_token = tokenizer.eos_token

    with timer("loading model from merged"):
        model = AutoModelForMultipleChoice.from_pretrained(
            ckpt_path, **get_model_kwargs(model_config.get("quant_n_bits", None))
        )

    return model, tokenizer
