import argparse
import json
import pathlib

import numpy as np
import torch.cuda
from peft import LoraConfig
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import EvalPrediction
from trl import SFTTrainer

import llm_science_exam.data.dataset
import llm_science_exam.model.llama2
import llm_science_exam.pj_struct_paths
import llm_science_exam.score

parser = argparse.ArgumentParser()
parser.add_argument("config_path")
parser.add_argument("--resume-from-checkpoint", type=str, default=None)
parser.add_argument("--use-map-at-3-as-metric", action="store_true", default=True)
args = parser.parse_args()

use_map_at_3_as_metric = args.use_map_at_3_as_metric
resume_from_checkpoint = args.resume_from_checkpoint

# config_path = "config/llama2.toml"
# config_path = "config/platypus2.toml"
config_path = args.config_path


config = llm_science_exam.data.config.get_config(config_path)
save_dir = llm_science_exam.data.config.get_checkpoint_path(config)

if save_dir.exists() and args.resume_from_checkpoint is None:
    raise FileExistsError(save_dir)

print(f"saves at {save_dir}")

if resume_from_checkpoint is not None:
    print(f"resume from checkpoint: {resume_from_checkpoint}")
    if not pathlib.Path(resume_from_checkpoint).exists():
        raise FileNotFoundError(resume_from_checkpoint)


dataset = llm_science_exam.data.dataset.get_dataset("train", config["dataset"])


from datasets import Dataset

# dataset["train"] = Dataset.from_pandas(dataset["train"].to_pandas().sample(n=20_000, random_state=42))


# valid_df_no_answer = dataset["valid"].to_pandas()
# valid_df_no_answer = valid_df_no_answer.sample(80, random_state=42).sort_values("id")
# dataset["valid"] = Dataset.from_pandas(valid_df_no_answer)


valid_answers = dataset["valid"]["answer"]

# valid_dataset_no_answer = llm_science_exam.model.llama2.dataset.add_prompt_field(
#     dataset["valid"],
#     model_family_name=config["model"]["family"],
#     prompt_id=config["dataset"]["prompt_id"],
#     new_field_name="text",
#     with_answer=False,
#     context_upper_limit_of_n_words=config["dataset"]["context"]["upper_limit_of_n_words"],
# )


dataset["train"] = llm_science_exam.model.llama2.dataset.add_prompt_field(
    dataset["train"],
    model_family_name=config["model"]["family"],
    prompt_id=config["dataset"]["prompt_id"],
    new_field_name="text",
    with_answer=True,
    context_upper_limit_of_n_words=config["dataset"]["context"]["upper_limit_of_n_words"],
)
print(f"prompt example:\n{dataset['train']['text'][0]}")

dataset["valid"] = llm_science_exam.model.llama2.dataset.add_prompt_field(
    dataset["valid"],
    model_family_name=config["model"]["family"],
    prompt_id=config["dataset"]["prompt_id"],
    new_field_name="text",
    with_answer=False,
    # context_upper_limit_of_n_words=-1,
    context_upper_limit_of_n_words=config["dataset"]["context"]["upper_limit_of_n_words"],
)


save_dir.mkdir(exist_ok=True, parents=True)
with open(save_dir / "train_config.json", "w") as f:
    json.dump(config, f, indent=2)


model, tokenizer = llm_science_exam.model.llama2.model.get_model(config["model"])

# max_seq_length = 650
# max_seq_length = 600
# max_seq_length = 500
max_seq_length = -1

if max_seq_length > 0:
    print(f"Train dataset length changed due to max_seq_length: {len(dataset['train']):,} -> ", end="", flush=True)
    lens = dataset["train"].map(
        lambda e: {"n": len(tokenizer(e["text"])["input_ids"])}, desc="counting n_tokens", num_proc=4
    )["n"]
    print(f"{np.quantile(lens, [0, 0.5, 0.75, 0.9, 0.95, 1]) = }")
    reduced_examples = [e for e, len_ in zip(dataset["train"], lens) if len_ <= max_seq_length]
    dataset["train"] = Dataset.from_list(reduced_examples)
    print(f"{len(dataset['train']):,}")


training_args = llm_science_exam.model.llama2.train.get_training_args(
    config, output_dir=save_dir, use_early_stopping="valid" in dataset, use_map_at_3_as_metric=use_map_at_3_as_metric
)

batch_size = 32


def compute_metrics(_eval_prediction: EvalPrediction):
    probs = llm_science_exam.model.llama2.predict.get_predicted_probs(
        model,
        tokenizer,
        dataset["valid"],
        model_family=config["model"]["family"],
        prompt_id=config["dataset"]["prompt_id"],
        # upper_limit_to_split_samples=1800,
    )
    preds = np.take(list("ABCDE"), np.argsort(probs, axis=1)[:, ::-1])
    map_at_3_score = llm_science_exam.score.print_map_at_3(valid_answers, preds)
    if len(valid_answers) > 200:
        old_map_at_3_score = llm_science_exam.score.map_at_3(valid_answers[:200], preds[:200])
        return {"MAP@3": map_at_3_score, "old_MAP@3": old_map_at_3_score}
    else:
        return {"MAP@3": map_at_3_score}


def preprocess_logits_for_metrics(_logits: torch.Tensor, _labels: torch.Tensor):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    # pred_ids = torch.argmax(logits[0], dim=-1)
    # return pred_ids, labels
    return torch.Tensor(), torch.Tensor()


callbacks = [
    llm_science_exam.model.llama2.train.SavePeftModelCallback,
]
if len(dataset.get("valid", [])) > 0:
    callbacks.append(
        EarlyStoppingCallback(early_stopping_patience=config["train"]["early_stopping_patience"]),
    )


# for llama 2 (they need different target module)
lora_config = LoraConfig(
    r=config["train"]["lora"]["r"],  # dimension of the updated matrices
    lora_alpha=config["train"]["lora"]["alpha"],  # parameter for scaling
    target_modules=llm_science_exam.model.llama2.model.find_linear_layers(
        model, quant_n_bits=config["model"]["quant_n_bits"]
    ),  # this chooses on which layers QLoRA is applied
    lora_dropout=config["train"]["lora"]["dropout"],  # dropout probability for layers
    bias="none",
    task_type="CAUSAL_LM",
)


supervised_finetuning_trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    eval_dataset=dataset.get("valid", None),
    args=training_args,
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
    # max_seq_length=3000,
    # max_seq_length=700,
    callbacks=callbacks,
    dataset_num_proc=12,
    compute_metrics=compute_metrics if use_map_at_3_as_metric else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics if use_map_at_3_as_metric else None,
)
supervised_finetuning_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# supervised_finetuning_trainer.save_state()
# supervised_finetuning_trainer.save_model()
#
# del model, supervised_finetuning_trainer
# gc.collect()
# torch.cuda.empty_cache()
#
# best_ckpt_path = llm_science_exam.model.llama2.model.get_best_checkpoint_path(ckpt_path=save_dir)
# llm_science_exam.model.llama2.model.merge_model(best_ckpt_path)
