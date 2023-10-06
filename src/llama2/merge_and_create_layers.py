import argparse
import json
import pathlib

from create_layer_by_layer_safetensors import save_layers

import llm_science_exam.model.llama2

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path", type=pathlib.Path)
args = parser.parse_args()
# ckpt_path = pathlib.Path("models-on-a100/07-16bits-after-270k/r-512-270k/checkpoint-5600/")
# ckpt_path = pathlib.Path("models-on-a100/07-16bits-after-270k/base/checkpoint-4800")
ckpt_path = args.checkpoint_path

if not (ckpt_path / "merged").exists():
    llm_science_exam.model.llama2.model.merge_model(ckpt_path)

save_layers(ckpt_path / "merged")

dataset_metadata_json = {
    "title": "LLaMa2 13B layers",
    "id": "ranchantan/llmse-llama2-13b-layers-es-with-full-td",
    "licenses": [{"name": "CC0-1.0"}],
}
with open(ckpt_path / "layers" / "dataset-metadata.json", "w") as f:
    json.dump(dataset_metadata_json, f)
