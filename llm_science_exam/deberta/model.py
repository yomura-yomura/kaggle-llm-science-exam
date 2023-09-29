from typing import Literal, TypedDict

import torch
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    BitsAndBytesConfig,
    DebertaV2Model,
    DebertaV2TokenizerFast,
)

ModelVersion = Literal["v3"]
ModelSize = Literal["large"]


class ModelConfig(TypedDict, total=True):
    version: ModelVersion
    size: ModelSize


def get_model(model_config: ModelConfig) -> tuple[DebertaV2Model, DebertaV2TokenizerFast]:
    model_name = get_model_name(model_config)
    print(f"-- Loading {model_name}")

    model = AutoModelForMultipleChoice.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def get_model_name(model_config: ModelConfig) -> str:
    match model_config:
        case {"version": "v3", "size": "large"}:
            return "microsoft/deberta-v3-large"
        case _:
            raise NotImplementedError(f"unexpected model_config: {model_config}")
