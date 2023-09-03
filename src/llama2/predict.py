import collections
import pathlib
import shutil
from typing import Literal

import numpy as np
import pandas as pd
import tqdm

import llm_science_exam.data.config
import llm_science_exam.data.dataset
import llm_science_exam.llama2
import llm_science_exam.score

# ckpt_path = pathlib.Path("SFT-llama2-7b") / "1000steps"
# ckpt_path = pathlib.Path("SFT-llama2-7b") / "with_early_stopping_with_more_datasets"
# ckpt_path = pathlib.Path("SFT-llama2-7b") / "02-prompt-fixed" / "1000steps"
# ckpt_path = pathlib.Path("SFT-llama2-7b") / "1000steps_with_more_datasets"
# ckpt_path = pathlib.Path("SFT-llama2-7b") / "with_early_stopping_with_2_more_datasets"
# ckpt_path = pathlib.Path("SFT-llama2-13b") / "with_early_stopping_with_2_more_datasets"

# ckpt_path = pathlib.Path("SFT-llama2-13b") / "with_early_stopping_with_2_more_datasets"
# ckpt_path = pathlib.Path("SFT-llama2-13b") / "with_early_stopping_with_2_more_datasets_tol50" / "checkpoint-5400"

# ckpt_path = pathlib.Path("models") / "SFT-llama2-7b" / "with_extra_+6.5k_+13.5k" / "best"
# ckpt_path = pathlib.Path("models") / "SFT-llama2-7b" / "with_extra_+6.5k_+13.5k_+1k" / "best"

# base_dir = pathlib.Path("models") / "02-trained-with-only-extra-datasets"
# ckpt_path = base_dir / "SFT-llama2-13b" / "with_extra_+6.5k_+13.5k_+1k"

# ckpt_path = pathlib.Path("models/03-short-prompt/SFT-llama2-7b/with_extra_+6.5k_+13.5k_+1k")
# ckpt_path = pathlib.Path("models") / "03-short-prompt" / "SFT-llama2-13b" / "with_extra_+6.5k_+13.5k_+1k"
ckpt_path = pathlib.Path("models") / "03-short-prompt2" / "SFT-llama2-7b" / "with_extra_+6.5k_+13.5k_+1k"

ckpt_path = llm_science_exam.llama2.model.get_best_checkpoint_path(ckpt_path)

print(ckpt_path)

# ckpt_path /= "merged"

dataset_type: Literal["train", "test"] = "train"


config = llm_science_exam.data.config.get_config(ckpt_path)

dataset = llm_science_exam.data.dataset.get_dataset(dataset_type, config["dataset"])
dataset = llm_science_exam.llama2.dataset.add_prompt_field(
    dataset, prompt_id=config["prompt_id"], new_field_name="text", with_answer=False
)


model, tokenizer = llm_science_exam.llama2.model.get_model_from_checkpoint(ckpt_path)


preds = collections.defaultdict(list)

for dataset_type, example in tqdm.tqdm(
    ((dataset_type_, example_) for dataset_type_, examples in dataset.items() for example_ in examples),
    total=sum(len(v) for v in dataset.values()),
):
    if dataset_type == "train":
        continue

    pred = llm_science_exam.llama2.predict.get_predicted_labels(model, tokenizer, example, config["prompt_id"])
    if "answer" in example:
        print(
            f"""
    True: {example["answer"]}
    Pred: {pred}
        """
        )
    preds[dataset_type].append(pred)


if "train" in dataset:
    for dataset_type in preds:
        map_at_3_score = llm_science_exam.score.map_at_3(dataset[dataset_type]["answer"], preds[dataset_type])
        print(f"MAP@3 ({dataset_type}) = {map_at_3_score:.3f}")


if "test" in dataset:

    def format_prediction(row, k=3):
        best_k_preds = row[:k]
        return " ".join(best_k_preds)

    test_df = pd.DataFrame(preds)
    test_df["prediction"] = test_df.apply(lambda x: format_prediction(x), axis=1)
    test_df["id"] = test_df.index

    submission_df = test_df[["id", "prediction"]]
    submission_df.to_csv("submission.csv", index=False)


# del model, tokenizer
# import torch
# torch.cuda.empty_cache()
