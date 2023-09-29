import json

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

import llm_science_exam.data
import llm_science_exam.deberta
import llm_science_exam.pj_struct_paths

llm_science_exam.pj_struct_paths.set_pj_struct_paths(
    kaggle_dataset_dir_path=llm_science_exam.pj_struct_paths.get_data_dir_path()
    / "kaggle-llm-science-exam-with-context"
)

config_path = "config/deberta-v3-large.toml"


config = llm_science_exam.data.config.get_config(config_path)
save_dir = llm_science_exam.data.config.get_checkpoint_path(config)

if save_dir.exists():
    raise FileExistsError(save_dir)

print(f"saves at {save_dir}")


model, tokenizer = llm_science_exam.deberta.model.get_model(config["model"])

print("We are using PEFT.")
from peft import LoraConfig, TaskType, get_peft_model

# import llm_science_exam.llama2

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    task_type=TaskType.SEQ_CLS,
    lora_dropout=0.1,
    bias="none",
    inference_mode=False,
    target_modules=["query_proj", "value_proj"],
    # target_modules=llm_science_exam.llama2.model.find_linear_layers(model),
    modules_to_save=["classifier", "pooler"],
)
model = get_peft_model(model, peft_config)


dataset = llm_science_exam.data.dataset.get_dataset("train", config["dataset"])
tokenized_dataset = llm_science_exam.deberta.dataset.map_preprocess(dataset, tokenizer, max_length=1024)

print(f"prompt example:\n{tokenized_dataset['train']['input_ids'][0]}")

save_dir.mkdir(exist_ok=True, parents=True)
with open(save_dir / "train_config.json", "w") as f:
    json.dump(config, f, indent=2)


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
    metric_for_best_model="MAP@3",
)


import llm_science_exam.score


def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"MAP@3": llm_science_exam.score.map_at_3(predictions, labels)}


import dataclasses

import torch
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy


@dataclasses.dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)
trainer.train()

trainer.save_state()
trainer.save_model()
