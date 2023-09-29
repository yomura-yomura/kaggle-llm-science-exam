import argparse
import gc
import json
import pathlib
import shutil

import numpy as np
import torch.cuda
import tqdm
from peft import LoraConfig
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import llm_science_exam.data.dataset
import llm_science_exam.llama2
import llm_science_exam.pj_struct_paths
import llm_science_exam.score

llm_science_exam.pj_struct_paths.set_pj_struct_paths(
    kaggle_dataset_dir_path=llm_science_exam.pj_struct_paths.get_data_dir_path()
    / "kaggle-llm-science-exam-with-context"
)


parser = argparse.ArgumentParser()
parser.add_argument("config_path")
parser.add_argument("--resume-from-checkpoint", type=str, default=None)
# parser.add_argument("--use-map-at-3-as-metric", action="store_true", default=False)
# args = parser.parse_args(["config/a100/llama2-16bits.toml"])
args = parser.parse_args()

# use_map_at_3_as_metric = args.use_map_at_3_as_metric
use_map_at_3_as_metric = True


# config_path = "config/llama2.toml"
# config_path = "config/platypus2.toml"
config_path = args.config_path


config = llm_science_exam.data.config.get_config(config_path)
save_dir = llm_science_exam.data.config.get_checkpoint_path(config)

if save_dir.exists() and args.resume_from_checkpoint is None:
    raise FileExistsError(save_dir)

print(f"saves at {save_dir}")


dataset = llm_science_exam.data.dataset.get_dataset("train", config["dataset"])


from datasets import Dataset

# dataset["train"] = Dataset.from_pandas(dataset["train"].to_pandas().sample(n=20_000, random_state=42))


# valid_df_no_answer = dataset["valid"].to_pandas()
# valid_df_no_answer = valid_df_no_answer.sample(80, random_state=42).sort_values("id")
# dataset["valid"] = Dataset.from_pandas(valid_df_no_answer)


valid_answers = dataset["valid"]["answer"]

# valid_dataset_no_answer = llm_science_exam.llama2.dataset.add_prompt_field(
#     dataset["valid"],
#     model_family_name=config["model"]["family"],
#     prompt_id=config["dataset"]["prompt_id"],
#     new_field_name="text",
#     with_answer=False,
#     context_upper_limit_of_n_words=config["dataset"]["context"]["upper_limit_of_n_words"],
# )


dataset["train"] = llm_science_exam.llama2.dataset.add_prompt_field(
    dataset["train"],
    model_family_name=config["model"]["family"],
    prompt_id=config["dataset"]["prompt_id"],
    new_field_name="text",
    with_answer=True,
    context_upper_limit_of_n_words=config["dataset"]["context"]["upper_limit_of_n_words"],
)
print(f"prompt example:\n{dataset['train']['text'][0]}")

dataset["valid"] = llm_science_exam.llama2.dataset.add_prompt_field(
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


model, tokenizer = llm_science_exam.llama2.model.get_model(config["model"])

# max_seq_length = 650
# max_seq_length = 600
max_seq_length = 500

print(f"Train dataset length changed due to max_seq_length: {len(dataset['train']):,} -> ", end="", flush=True)
lens = dataset["train"].map(
    lambda e: {"n": len(tokenizer(e["text"])["input_ids"])}, desc="counting n_tokens", num_proc=4
)["n"]
print(f"{np.quantile(lens, [0, 0.5, 0.75, 0.9, 0.95, 1]) = }")
reduced_examples = [e for e, len_ in zip(dataset["train"], lens) if len_ <= max_seq_length]
dataset["train"] = Dataset.from_list(reduced_examples)
print(f"{len(dataset['train']):,}")


training_args = TrainingArguments(
    output_dir=str(save_dir),
    per_device_train_batch_size=config["train"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["train"]["per_device_eval_batch_size"],
    # per_device_train_batch_size=8,
    # per_device_eval_batch_size=16,
    gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
    # learning_rate=2e-4,
    # learning_rate=1e-4,
    learning_rate=config["train"]["learning_rate"],
    logging_steps=config["train"]["logging_steps"],
    logging_strategy="steps",
    #
    save_steps=config["train"]["save_steps"],
    save_strategy="steps",
    save_total_limit=config["train"]["save_total_limit"],
    #
    # Learning-Rate Scheduler
    lr_scheduler_type=config["train"]["lr_scheduler_type"],
    warmup_steps=config["train"]["warmup_steps"],
    # warmup_ratio=config["train"]["warmup_ratio"],
    #
    num_train_epochs=config["train"]["num_train_epochs"],
    # max_steps=1000,
    optim="paged_adamw_8bit",
    fp16=True,
    bf16=False,
    # fp16=False,
    # bf16=True,
    weight_decay=config["train"]["weight_decay"],
    run_name="baseline-llama2-sft",
    report_to=["none"],
    #
    # for Early Stopping
    load_best_model_at_end="valid" in dataset,
    evaluation_strategy="steps" if "valid" in dataset else "no",
    eval_steps=None,  # same as logging_steps
    #
    # Metric
    metric_for_best_model="MAP@3" if use_map_at_3_as_metric else None,
    greater_is_better=True if use_map_at_3_as_metric else None,
)

import numpy as np
from transformers.trainer_utils import EvalPrediction

batch_size = 32


def compute_metrics(_eval_prediction: EvalPrediction):
    preds = llm_science_exam.llama2.predict.get_predicted_labels(
        model,
        tokenizer,
        dataset["valid"],
        model_family=config["model"]["family"],
        prompt_id=config["dataset"]["prompt_id"],
        upper_limit_to_split_samples=1800,
    )
    map_at_3_score = llm_science_exam.score.map_at_3(valid_answers, preds)

    return {"MAP@3": map_at_3_score}


def preprocess_logits_for_metrics(_logits: torch.Tensor, _labels: torch.Tensor):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    # pred_ids = torch.argmax(logits[0], dim=-1)
    # return pred_ids, labels
    return torch.Tensor(), torch.Tensor()


import os

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = pathlib.Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = checkpoint_folder / "pytorch_model.bin"
        if pytorch_model_path.exists():
            os.remove(pytorch_model_path)

        train_config_json = checkpoint_folder.parent / "train_config.json"
        if train_config_json.exists():
            shutil.copy(train_config_json, checkpoint_folder / train_config_json.name)

        return control


callbacks = [
    SavePeftModelCallback,
]
if len(dataset.get("valid", [])) > 0:
    callbacks.append(
        EarlyStoppingCallback(early_stopping_patience=config["train"]["early_stopping_patience"]),
    )


if False:
    data_collator = DataCollatorForCompletionOnlyLM(response_template="\n\n### Answer:", tokenizer=tokenizer)
    # data_collator = DataCollatorForCompletionOnlyLM(response_template="\n\nAnswer: \n", tokenizer=tokenizer)
    if data_collator.response_token_ids[0] == 29871:
        data_collator.response_token_ids = data_collator.response_token_ids[1:]  # Remove 29871 if exists
else:
    data_collator = None

# for llama 2 (they need different target module)
lora_config = LoraConfig(
    r=config["train"]["lora"]["r"],  # dimension of the updated matrices
    lora_alpha=config["train"]["lora"]["alpha"],  # parameter for scaling
    target_modules=llm_science_exam.llama2.model.find_linear_layers(
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
    data_collator=data_collator,
    callbacks=callbacks,
    compute_metrics=compute_metrics if use_map_at_3_as_metric else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics if use_map_at_3_as_metric else None,
)
supervised_finetuning_trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

supervised_finetuning_trainer.save_state()
supervised_finetuning_trainer.save_model()
#
# del model, supervised_finetuning_trainer
# gc.collect()
# torch.cuda.empty_cache()
#
# best_ckpt_path = llm_science_exam.llama2.model.get_best_checkpoint_path(ckpt_path=save_dir)
# llm_science_exam.llama2.model.merge_model(best_ckpt_path)
