import argparse
from typing import Literal

import numpy as np
import pandas as pd

import llm_science_exam.data.config
import llm_science_exam.model.checkpoint
import llm_science_exam.model.llama2
import llm_science_exam.score

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path", type=str)
parser.add_argument("--valid-type", "-v", choices=["200", "200+300", "800"], default="200")
args = parser.parse_args()

ckpt_path = args.checkpoint_path

# config = llm_science_exam.data.config.get_config(config_path)
# ckpt_path = llm_science_exam.data.config.get_checkpoint_path(config)

ckpt_path = llm_science_exam.model.checkpoint.get_best_checkpoint_path(ckpt_path)
config = llm_science_exam.data.config.get_config_from_checkpoint(ckpt_path, drop_log_history=True)

# batch_size = 16
# batch_size = 8
batch_size = 8

print(ckpt_path)

dataset_type: Literal["train", "valid", "test"] = "valid"

match args.valid_type:
    case "200":
        pass
    case "200+300":
        config["dataset"]["valid_additional_datasets"] = ["yalickj/dataset-wiki-new-1"]
    case "800":
        config["dataset"]["valid_additional_datasets"] = [
            "takeshisuzuki/additional-dataset-800articles-4000rows/only-q1"
        ]
    case _:
        assert False


dataset = llm_science_exam.data.dataset.get_dataset(dataset_type, config["dataset"])[dataset_type]
df = dataset.to_pandas()

dataset = llm_science_exam.model.llama2.dataset.add_prompt_field(
    dataset,
    model_family_name=config["model"]["family"],
    prompt_id=config["dataset"]["prompt_id"],
    new_field_name="text",
    with_answer=False,
    context_upper_limit_of_n_words=-1,
    n_cpus=None,
)
print(dataset[0]["text"])

prob_path = ckpt_path / f"prob-{args.valid_type}.csv"

if prob_path.exists():
    probs = pd.read_csv(prob_path).to_numpy(str)
else:
    model, tokenizer = llm_science_exam.model.llama2.model.get_model_from_checkpoint(config["model"], ckpt_path)

    probs = llm_science_exam.model.llama2.predict.get_predicted_probs(
        model,
        tokenizer,
        dataset,
        model_family=config["model"]["family"],
        prompt_id=config["dataset"]["prompt_id"],
        batch_size=batch_size,
    )
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
