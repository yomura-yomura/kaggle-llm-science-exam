import pathlib
from typing import Literal

import numpy as np
import pandas as pd
import tqdm

import llm_science_exam.data.config
import llm_science_exam.llama2
import llm_science_exam.pj_struct_paths
import llm_science_exam.score

llm_science_exam.pj_struct_paths.set_pj_struct_paths(
    kaggle_dataset_dir_path=llm_science_exam.pj_struct_paths.get_data_dir_path()
    / "kaggle-llm-science-exam-with-context"
)


# config_path = "config/llama2.toml"
# config_path = "config/platypus2.toml"
# config_path = "config/openorca-platypus2.toml"
# config_path = "config/llama2-with-context-500w.toml"
# config_path = "config/llama2-with-context-300w.toml"
# config_path = "config/a100/llama2-with-context-300w.toml"
config_path = "config/a100/llama2-16bits.toml"


config = llm_science_exam.data.config.get_config(config_path)

ckpt_path = llm_science_exam.data.config.get_checkpoint_path(config)
ckpt_path = llm_science_exam.llama2.model.get_best_checkpoint_path(ckpt_path)

# ckpt_path /= "best"

# ckpt_path = pathlib.Path("models-on-a100/06/yes_or_no_as_answer_with_context_v2/checkpoint-7200/")
# ckpt_path = pathlib.Path("models-on-a100/06/checkpoint-5200/")
# ckpt_path = pathlib.Path("models/llama2/06-16bits/13b/base/best/")
# ckpt_path = pathlib.Path("models/platypus2/02-prompt-change/7b/with_extra_+60k/_checkpoint-4100")
# ckpt_path = pathlib.Path("../../data/Llama2-7b-hf/")
# ckpt_path = pathlib.Path("../../data/weyaxi/llama2-13b/")


# batch_size = 16
# batch_size = 8
batch_size = 8

print(ckpt_path)

dataset_type: Literal["train", "valid", "test"] = "valid"


# config["dataset"]["prompt_id"] = 9
# config["dataset"]["context"]["upper_limit_of_n_words"] = 300

# config = llm_science_exam.data.config.get_config_from_checkpoint(ckpt_path)

dataset = llm_science_exam.data.dataset.get_dataset(dataset_type, config["dataset"])[dataset_type]
df = dataset.to_pandas()

dataset = llm_science_exam.llama2.dataset.add_prompt_field(
    dataset,
    model_family_name=config["model"]["family"],
    prompt_id=config["dataset"]["prompt_id"],
    new_field_name="text",
    with_answer=False,
    # context_upper_limit_of_n_words=config["dataset"]["context"]["upper_limit_of_n_words"],
    context_upper_limit_of_n_words=-1,
    n_cpus=None,
)
print(dataset[0]["text"])

pred_path = ckpt_path / "pred.csv"

if pred_path.exists():
    preds = pd.read_csv(pred_path).to_numpy(str)
else:
    model, tokenizer = llm_science_exam.llama2.model.get_model_from_checkpoint(config["model"], ckpt_path)

    preds = llm_science_exam.llama2.predict.get_predicted_labels(
        model,
        tokenizer,
        dataset,
        model_family=config["model"]["family"],
        prompt_id=config["dataset"]["prompt_id"],
        upper_limit_to_split_samples=1800,  # 13B
        # upper_limit_to_split_samples=2500,  # 7B
        batch_size=batch_size,
    )
    pd.DataFrame(preds, columns=[f"top{i + 1}" for i in range(5)]).to_csv(pred_path, index=False)

if "answer" in df.columns:
    map_at_3_scores = llm_science_exam.score.map_at_3(df["answer"], preds, reduction=None)
    print(f"MAP@3 ({dataset_type}) = {np.mean(map_at_3_scores):.3f}")

    # indices = df.loc[map_at_3_scores == 0, "id"].tolist()
    #
    # dataset_df = dataset.to_pandas()
    #
    # pd.set_option("display.max_colwidth", None)
    # for _, row in dataset_df[dataset_df["id"] == indices[-1]].iterrows():
    #     print(f"{row['text']}{row['yes_or_no']}\n\n")
