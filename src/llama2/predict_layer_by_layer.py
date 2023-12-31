import argparse
import pathlib
from typing import Literal

import numpy as np
import pandas as pd

import llm_science_exam.data.config
import llm_science_exam.model.checkpoint
import llm_science_exam.model.llama2
import llm_science_exam.score

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path", type=str)
parser.add_argument("--context-version", "-c", type=int, choices=[3, 4], default=None)
parser.add_argument("--valid-type", "-v", choices=["200", "200+300", "200+c300", "200+800", "800"], default="200")
parser.add_argument("--batch-size", "-b", type=int, default=4)
args = parser.parse_args()

context_version = args.context_version

ckpt_path = pathlib.Path(args.checkpoint_path)

if (ckpt_path / "layers").exists():
    pass
else:
    ckpt_path = llm_science_exam.model.checkpoint.get_best_checkpoint_path(ckpt_path)
config = llm_science_exam.data.config.get_config_from_checkpoint(ckpt_path, drop_log_history=True)

config["dataset"]["context"]["version"] = 3
# config["dataset"]["valid_additional_datasets"] = ["takeshisuzuki/additional-dataset-800articles-4000rows/only-q1"]
# config["dataset"]["test_size"] = 0

print(ckpt_path)

dataset_type: Literal["train", "valid", "test"] = "valid"

match args.valid_type:
    case "200":
        pass
    case "200+300":
        config["dataset"]["valid_additional_datasets"] = ["yalickj/dataset-wiki-new-1"]
    case "200+c300":
        config["dataset"]["valid_additional_datasets"] = ["wuwenmin/llm-sci-eval300-gpt4-corrected"]
    case "200+800":
        config["dataset"]["valid_additional_datasets"] = [
            "takeshisuzuki/additional-dataset-800articles-4000rows/only-q1"
        ]
    case "800":
        config["dataset"]["valid_additional_datasets"] = [
            "takeshisuzuki/additional-dataset-800articles-4000rows/only-q1"
        ]
        config["dataset"]["test_size"] = 0
    case _:
        assert False

config["dataset"]["context"]["top_n_sentences"] = 3
if context_version is not None:
    config["dataset"]["context"]["version"] = context_version
else:
    context_version = config["dataset"]["context"]["version"]

dataset = llm_science_exam.data.dataset.get_dataset(dataset_type, config["dataset"])[dataset_type]
df = dataset.to_pandas()


prob_path = ckpt_path / f"prob-{args.valid_type}-v{context_version}.csv"

if prob_path.exists():
    preds = pd.read_csv(prob_path).to_numpy("f8")
else:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path / "layers")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    inputs = df.apply(
        lambda row: llm_science_exam.model.llama2.layer_by_layer_predict.get_tokens(row, tokenizer, max_length=None),
        axis=1,
    ).values
    MAX_LENGTH = max(p.shape[1] + s.shape[1] for p, s in inputs)
    print(f"{MAX_LENGTH = }")
    del inputs

    logits = llm_science_exam.model.llama2.layer_by_layer_predict.run_model(
        ckpt_path / "layers", df, devices=[0], max_length=MAX_LENGTH, n_batches=args.batch_size
    )

    import torch

    preds = torch.softmax(torch.Tensor(np.array(logits)), dim=1).numpy()
    pd.DataFrame(preds).to_csv(prob_path, index=False)


llm_science_exam.score.print_map_at_3(df["answer"], np.take(list("ABCDE"), np.argsort(preds, axis=1)[:, ::-1]))
