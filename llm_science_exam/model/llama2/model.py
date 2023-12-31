import pathlib
import shutil
import warnings
from typing import Literal, TypedDict

import bitsandbytes as bnb
import torch
import torch.nn
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizerFast

from ... import pj_struct_paths
from ...typing import FilePath
from ...utils import timer
from ..model import get_model_kwargs

ModelFamily = Literal["Llama2", "Platypus2", "OpenOrca-Platypus2"]
ModelSize = Literal["7B", "13B"]


class ModelConfig(TypedDict, total=True):
    family: ModelFamily
    size: ModelSize
    quant_n_bits: int


def get_model(model_config: ModelConfig) -> tuple[LlamaForCausalLM, LlamaTokenizerFast]:
    model_name = get_model_dir_path(model_config)
    print(f"-- Loading {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **get_model_kwargs(model_config["quant_n_bits"]))
    # this should be set as False for fine-tuning
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_model_dir_path(model_config: ModelConfig) -> pathlib.Path:
    if model_config["family"] == "Llama2":
        if model_config["size"] == "7B":
            model_name = pj_struct_paths.get_data_dir_path() / "Llama2-7b-hf"
        elif model_config["size"] == "13B":
            model_name = pj_struct_paths.get_data_dir_path() / "weyaxi" / "llama2-13b"
        else:
            raise ValueError(f"unexpected model size for family '{model_config['family']}': {model_config['size']}")
    elif model_config["family"] == "Platypus2":
        if model_config["size"] == "7B":
            model_name = pj_struct_paths.get_data_dir_path() / "Platypus2-7B"
        else:
            raise ValueError(f"unexpected model size for family '{model_config['family']}': {model_config['size']}")
    elif model_config["family"] == "OpenOrca-Platypus2":
        if model_config["size"] == "13B":
            model_name = pj_struct_paths.get_data_dir_path() / "OpenOrca-Platypus2-13B"
        else:
            raise ValueError(f"unexpected model size for family '{model_config['family']}': {model_config['size']}")
    else:
        raise ValueError(f"unexpected model family: {model_config['family']}")
    return model_name


def find_linear_layers(model: LlamaForCausalLM, *, quant_n_bits: int) -> list[str]:
    """find linear layers in given transformer model"""
    lora_module_names = set()
    for name, module in model.named_modules():
        if quant_n_bits == 4:
            # 4 bits for qlora
            if isinstance(module, bnb.nn.Linear4bit):
                lora_module_names.add(name.rsplit(".", 1)[-1])
        elif quant_n_bits in [16, 32]:
            if isinstance(module, torch.nn.Linear):
                lora_module_names.add(name.rsplit(".", 1)[-1])
        else:
            raise ValueError(f"Unexpected quant_n_bits: {quant_n_bits}")

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


def get_model_from_checkpoint(
    model_config: ModelConfig, ckpt_path: FilePath
) -> tuple[LlamaForCausalLM, LlamaTokenizerFast]:
    ckpt_path = pathlib.Path(ckpt_path)

    if (ckpt_path / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        tokenizer.pad_token = tokenizer.eos_token

        with timer("loading model from merged"):
            model = LlamaForCausalLM.from_pretrained(ckpt_path, **get_model_kwargs(model_config["quant_n_bits"]))
    else:
        if not (ckpt_path / "merged" / "train_config.json").exists():
            merge_model(ckpt_path)
        return get_model_from_checkpoint(model_config, ckpt_path / "merged")

    return model, tokenizer


def merge_model(ckpt_path: FilePath):
    ckpt_path = pathlib.Path(ckpt_path)

    merged_ckpt_path = ckpt_path / "merged"

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    lora_config = LoraConfig.from_pretrained(str(ckpt_path))

    with timer("loading model"):
        model = AutoPeftModelForCausalLM.from_pretrained(str(ckpt_path), config=lora_config)

    with timer("merge and unload"):
        model = model.merge_and_unload()
    model.half()

    if merged_ckpt_path.exists():
        warnings.warn(f"Merged Model cannot be saved, because directory already exists: {merged_ckpt_path}")
    else:
        model.save_pretrained(ckpt_path / "merged")
        tokenizer.save_pretrained(ckpt_path / "merged")
        shutil.copy(ckpt_path / "train_config.json", ckpt_path / "merged" / "train_config.json")

    return model, tokenizer
