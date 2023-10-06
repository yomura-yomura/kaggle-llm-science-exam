import gc
import json
import pathlib
import shutil

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from llm_science_exam.typing import FilePath


def save_layers(checkpoint_path: FilePath, output_checkpoint_path: FilePath | None = None):
    """
    Save the all layers of a model sharded checkpoint using safetensors.
    """
    checkpoint_path = pathlib.Path(checkpoint_path)

    if output_checkpoint_path is None:
        output_checkpoint_path = checkpoint_path.with_name("layers")

    output_checkpoint_path = pathlib.Path(output_checkpoint_path)

    output_checkpoint_path.mkdir(exist_ok=True, parents=True)

    for k in ["config.json", "train_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(checkpoint_path / k, output_checkpoint_path / k)

    with open(checkpoint_path / "pytorch_model.bin.index.json", "rb") as f:
        index = json.load(f)["weight_map"]

    n_layers = len(set([int(k.split(".")[2]) for k in index.keys() if "model.layers" in k]))
    layers = ["model.embed_tokens."] + [f"model.layers.{i}." for i in range(n_layers)] + ["model.norm.", "lm_head."]
    shard = 0
    n_shards = len(set(index.values()))
    state_dict = {}

    for layer in tqdm(layers):
        # Optionnally load next shard
        shards = [int(v.split("-")[1]) for k, v in index.items() if k.startswith(layer)]
        if max(shards) > shard:
            shard += 1
            print(f"Loading shard {shard}/{n_shards}")
            state_dict.update(
                torch.load(
                    checkpoint_path / f"pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin", map_location="cpu"
                )
            )

        # Get layer state dict
        layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(layer)])

        # Save layer state dict as using safetensors
        save_file(layer_state_dict, output_checkpoint_path / (layer + "safetensors"))

        # Free memory
        for k in layer_state_dict.keys():
            del state_dict[k]
        del layer_state_dict
        gc.collect()


if __name__ == "__main__":
    # src = pathlib.Path("../llama2/models-on-a100/06/yes_or_no_as_answer_with_context/checkpoint-9200/merged")
    # src = pathlib.Path("../llama2/models-on-a100/06/yes_or_no_as_answer_with_context/checkpoint-9200/merged")
    src = pathlib.Path("../deberta/models/deberta-v3-large/01-base/test-max-length-1024/checkpoint-950/")

    dest = src.with_name("layers")
    save_layers(src, dest)
