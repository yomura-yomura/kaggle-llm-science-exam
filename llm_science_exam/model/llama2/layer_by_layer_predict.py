import functools
import pathlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import tqdm
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ...typing import FilePath
from ...utils import clean_memory


def run_model(
    checkpoint_path: FilePath | Sequence[FilePath],
    df: pd.DataFrame,
    devices: list[int],
    max_length: int,
    n_batches: int,
    output_token_ids: Sequence[int] = (4874, 694),
):
    def run_on_device(split_df: pd.DataFrame, device: int):
        clean_memory()

        model = ShardedLlama(checkpoint_path, device=device, dtype=torch.float16, max_length=max_length)
        f = functools.partial(get_tokens, tokenizer=model.tokenizer, max_length=max_length)
        inputs = split_df.apply(f, axis=1).values

        del model.tokenizer
        clean_memory()

        outputs = []
        for i, batch in enumerate(np.array_split(inputs, n_batches)):
            print(f"* batch #{i + 1} of {n_batches} on device {device}")
            #         outputs += model(batch, output_token=4874)
            outputs += model(batch, output_token=output_token_ids)
        #         outputs += model(batch, output_token=[3582, 1217])
        return outputs

    with ThreadPoolExecutor() as executor:
        logits = list(executor.map(run_on_device, np.array_split(df, len(devices)), devices))
        return sum(logits, [])


system_prefix = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Your task is to analyze the question and answer below. If the answer is correct, respond yes, if it is not correct respond no. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be pertinent.

### Input:
Context:
{context}

Question:
{prompt}

Proposed answer:
"""


def get_prompts(row):
    prompt_prefix = system_prefix.format(context=row["context"], prompt=row["prompt"])
    prompt_suffix = [f"{row[letter]}\n\n### Response:\n" for letter in "ABCDE"]

    return prompt_prefix, prompt_suffix


def get_tokens(row, tokenizer, max_length):
    prompt_prefix, prompt_suffix = get_prompts(row)

    prefix = tokenizer(
        prompt_prefix,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    suffix = tokenizer(
        prompt_suffix,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=max_length,
        #         padding="max_length" if max_length is not None else True,
        padding=True,
    )["input_ids"][:, 1:]

    return prefix, suffix


class ShardedLlama:
    def __init__(
        self,
        checkpoint_path: FilePath | Sequence[FilePath],
        device: int,
        max_length: int,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Sharded version of LlamaForCausalLM : the model is split into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM, but
        as Kaggle accelerators have more GPU memory than CPU, we simply batch the inputs and keep them on the GPU.

        Parameters
        ----------
        checkpoint_path : path to the checkpoint
        device : device
        dtype : dtype, by default torch.float16
        """

        # Save parameters
        self.checkpoint_paths = sorted(
            [pathlib.Path(checkpoint_path)]
            if not isinstance(checkpoint_path, Sequence)
            else list(map(pathlib.Path, checkpoint_path))
        )
        self.device = f"cuda:{device}"
        self.device_id = device
        self.dtype = dtype
        self.max_length = max_length

        # Create model
        for ckpt_path in self.checkpoint_paths:
            if (ckpt_path / "config.json").exists():
                self.config = AutoConfig.from_pretrained(ckpt_path)
                break
        else:
            raise FileNotFoundError(f"config.json not found in {self.checkpoint_paths}")

        # For flash attention when Turing architecture will be supported :
        # https://github.com/Dao-AILab/flash-attention/issues/542
        # self.config.auto_map = {
        #   "AutoModelForCausalLM" : togethercomputer/LLaMA-2-7B-32K--modeling_flash_llama.LlamaForCausalLM"
        # }

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.tokenizer_pad_token_id = self.tokenizer.pad_token_id

        self.init_model()
        self.layer_names = (
            ["model.embed_tokens"]
            + [f"model.layers.{i}" for i in range(len(self.model.model.layers))]
            + ["model.norm", "lm_head"]
        )

        self.model = None
        self.layers = None

    def init_model(self):
        # Load Meta model (no memory used)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
            self.model.tie_weights()

        self.layers = (
            [self.model.model.embed_tokens]
            + list(self.model.model.layers)
            + [self.model.model.norm, self.model.lm_head]
        )

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.device, value=buffer, dtype=self.dtype)

    def load_layer(self, layer_name: str):
        safetensors_filename = f"{layer_name}.safetensors"
        for ckpt_path in self.checkpoint_paths:
            if (ckpt_path / safetensors_filename).exists():
                state_dict = load_file(ckpt_path / safetensors_filename, device=self.device)
                break
        else:
            raise FileNotFoundError(f"{safetensors_filename} not found in {self.checkpoint_paths}")

        for param_name, param in state_dict.items():
            assert param.dtype != torch.int8, "int8 not supported (need to add fp16_statistics)"
            set_module_tensor_to_device(self.model, param_name, self.device, value=param, dtype=self.dtype)

    def __call__(self, inputs, output_token: int | Sequence[int]):
        # inputs = [(prefix, suffix), ...] with prefix.shape[0] = 1 and suffix.shape[0] = 5

        # Reboot the model to make sure buffers are loaded and memory is clean
        del self.model
        clean_memory()
        self.init_model()

        # Send batch to device
        batch = [(prefix.to(self.device), suffix.to(self.device)) for prefix, suffix in inputs]
        n_suffixes = len(batch[0][1])
        suffix_eos = [(suffix != self.tokenizer_pad_token_id).sum(1) - 1 for _, suffix in inputs]

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.finfo(self.dtype).min * torch.ones(self.max_length, self.max_length)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...]
        attention_mask = attention_mask.to(self.device)
        position_ids = torch.arange(self.max_length, dtype=torch.long, device=self.device)[None, :]

        with ThreadPoolExecutor() as executor, torch.inference_mode():
            # Load first layer
            future = executor.submit(self.load_layer, "model.embed_tokens")

            for i, (layer_name, layer) in enumerate(
                zip(
                    tqdm.tqdm(
                        self.layer_names,
                        desc=f"inference layer by layer on device {self.device}",
                        position=self.device_id + 1,
                    ),
                    self.layers,
                )
            ):
                # Wait for previous layer to be loaded and load next layer
                # start = time.time()
                future.result()
                if (i + 1) < len(self.layer_names):
                    future = executor.submit(self.load_layer, self.layer_names[i + 1])
                # load_time = time.time() - start

                # Run layer
                for j, (prefix, suffix) in enumerate(batch):
                    if layer_name == "model.embed_tokens":
                        batch[j] = (layer(prefix), layer(suffix))
                    elif layer_name == "model.norm":
                        # Only keep the last hidden state at this point
                        batch[j] = (None, layer(suffix[torch.arange(n_suffixes), suffix_eos[j]][:, None]))
                    elif layer_name == "lm_head":
                        batch[j] = (None, layer(suffix))
                    else:
                        # Run prefix
                        len_p, len_s = prefix.shape[1], suffix.shape[1]
                        new_prefix, (k_cache, v_cache) = layer(
                            prefix, use_cache=True, attention_mask=attention_mask[:, :, -len_p:, -len_p:]
                        )

                        # Run suffix
                        # pos = position_ids[:, len_p : len_p + len_s].repeat(n_suffixes, 1)
                        # attn = attention_mask[:, :, -len_s:, -len_p - len_s :].repeat(n_suffixes, 1, 1, 1)
                        # kv_cache = (k_cache.repeat(n_suffixes, 1, 1, 1), v_cache.repeat(n_suffixes, 1, 1, 1))
                        pos = position_ids[:, len_p : len_p + len_s].expand(n_suffixes, -1)
                        attn = attention_mask[:, :, -len_s:, -len_p - len_s :].expand(n_suffixes, -1, -1, -1)
                        kv_cache = (k_cache.repeat(n_suffixes, 1, 1, 1), v_cache.expand(n_suffixes, -1, -1, -1))

                        new_suffix = layer(suffix, past_key_value=kv_cache, position_ids=pos, attention_mask=attn)[0]
                        batch[j] = (new_prefix, new_suffix)

                # Remove previous layer from memory (including buffers)
                layer.to("meta")

        # Get scores
        if isinstance(output_token, Sequence):
            batch = [
                torch.softmax(suffix[:, -1, list(output_token)], dim=1)[..., 0].detach().cpu().numpy()
                for _, suffix in batch
            ]
        else:
            batch = [suffix[:, -1, output_token].detach().cpu().numpy() for _, suffix in batch]

        return batch
