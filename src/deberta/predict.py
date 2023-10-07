import argparse
from typing import Literal

import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader

import llm_science_exam.data.config
import llm_science_exam.model.checkpoint
import llm_science_exam.model.deberta
import llm_science_exam.model.deberta.predict
import llm_science_exam.score

# config_path = "config/a100/deberta-v3-large.toml"
# config = llm_science_exam.data.config.get_config(config_path)
# ckpt_path = llm_science_exam.data.config.get_checkpoint_path(config)

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path", type=str)
parser.add_argument("--context-version", "-c", type=int, choices=[3, 4], default=None)
parser.add_argument("--valid-type", "-v", choices=["200", "200+300", "200+c300", "200+800", "800"], default="200")
args = parser.parse_args()

context_version = args.context_version

ckpt_path = args.checkpoint_path
# ckpt_path = "models/deberta-v3-large/01-base/test-shuffled"

ckpt_path = llm_science_exam.model.checkpoint.get_best_checkpoint_path(ckpt_path)
config = llm_science_exam.data.config.get_config_from_checkpoint(ckpt_path, drop_log_history=True)

batch_size = 8

print(ckpt_path)

dataset_type: Literal["train", "valid", "test"]
dataset_type = "valid"


llm_science_exam.model.deberta.custom_forward_method.enable_memory_efficient_forward_method()

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

print(f"{context_version = }")

dataset = llm_science_exam.data.dataset.get_dataset(dataset_type, config["dataset"])[dataset_type]
df = dataset.to_pandas()
df = df.dropna()

print(f"{df['context'].iloc[-1]}")

dataset = Dataset.from_pandas(df, preserve_index=False)

print(f"{len(df) = }")


prob_path = ckpt_path / f"prob-{args.valid_type}-v{context_version}.csv"

if prob_path.exists():
    probs = pd.read_csv(prob_path).to_numpy("f8")
else:
    model, tokenizer = llm_science_exam.model.deberta.model.get_model_from_checkpoint(config["model"], ckpt_path)
    model.cuda()
    model.half()
    model.eval()

    tokenized_dataset = llm_science_exam.model.deberta.dataset.map_preprocess(
        dataset,
        tokenizer,
        with_answer=True,
        max_length=2 * 1024,
        # max_length=1024,s
        num_proc=None,
    )

    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=llm_science_exam.model.deberta.dataset.DataCollatorForMultipleChoice(tokenizer=tokenizer),
    )

    probs = []
    for batch in tqdm.tqdm(data_loader, desc="inference"):
        for k in batch.keys():
            batch[k] = batch[k].to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            probs.append(torch.softmax(outputs.logits, dim=-1).cpu().detach().numpy())
    probs = np.concatenate(probs)
    pd.DataFrame(probs, columns=[f"top{i + 1}" for i in range(5)]).to_csv(prob_path, index=False)


preds = np.take(list("ABCDE"), np.argsort(probs, axis=1)[:, ::-1])


if "answer" in df.columns:
    if len(df) > 200:
        print("\nOld CV:")
        llm_science_exam.score.print_map_at_3(df["answer"][:200], preds[:200])

        print("\nNew CV:")
    else:
        print("\nCV:")
    llm_science_exam.score.print_map_at_3(df["answer"], preds)
