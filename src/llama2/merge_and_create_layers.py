import pathlib

from create_layer_by_layer_safetensors import save_layers

import llm_science_exam.llama2

ckpt_path = pathlib.Path("models/llama2/06-16bits/13b/yes_or_no_as_answer_with_context_v2/best/")
llm_science_exam.llama2.model.merge_model(ckpt_path)

save_layers(ckpt_path / "merged")
