import pathlib
import shutil
import warnings

import bitsandbytes as bnb
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizerFast

from .. import pj_struct_paths
from ..typing import FilePath
from ..utils import timer

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def get_model(model_config: dict) -> tuple[LlamaForCausalLM, LlamaTokenizerFast]:
    if model_config["family"] == "Llama2":
        if model_config["size"] == "7B":
            model_name = pj_struct_paths.get_data_dir_path() / "Llama2-7b-hf"
        elif model_config["size"] == "13B":
            model_name = pj_struct_paths.get_data_dir_path() / "weyaxi" / "llama2-13b"
        else:
            raise ValueError(f"unexpected model size: {model_config['size']}")
    else:
        raise NotImplementedError(f"unexpected model family: {model_config['family']}")

    print(f"-- Loading {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
    # this should be set as False for fine-tuning
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def find_linear_layers(model: LlamaForCausalLM) -> list[str]:
    """find linear layers in given transformer model"""
    lora_module_names = set()
    for name, module in model.named_modules():
        # 4 bits for qlora
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


def get_model_from_checkpoint(ckpt_path: FilePath) -> tuple[LlamaForCausalLM, LlamaTokenizerFast]:
    ckpt_path = pathlib.Path(ckpt_path)

    if (ckpt_path / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

        with timer("loading model from merged"):
            model = LlamaForCausalLM.from_pretrained(
                ckpt_path,
                quantization_config=bnb_config,
                # torch_dtype=torch.float16,
                device_map="auto",
            )
    else:
        model, tokenizer = merge_model(ckpt_path)

    return model, tokenizer


def merge_model(ckpt_path: FilePath):
    ckpt_path = pathlib.Path(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    lora_config = LoraConfig.from_pretrained(str(ckpt_path))

    with timer("loading model"):
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(ckpt_path),
            config=lora_config,
            # quantization_config=bnb_config
        )

    with timer("merge and unload"):
        model = model.merge_and_unload()
    model.half()

    merged_ckpt_path = ckpt_path / "merged"
    if merged_ckpt_path.exists():
        warnings.warn(f"Merged Model cannot be saved, because directory already exists: {merged_ckpt_path}")
    else:
        model.save_pretrained(ckpt_path / "merged")
        tokenizer.save_pretrained(ckpt_path / "merged")
        shutil.copy(ckpt_path / "train_config.json", ckpt_path / "merged" / "train_config.json")

    model.to("cuda")

    return model, tokenizer