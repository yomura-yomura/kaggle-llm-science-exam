import gc
import json
import pathlib

import torch.cuda
from peft import LoraConfig
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import llm_science_exam.data.dataset
import llm_science_exam.llama2

# save_dir = pathlib.Path("SFT-llama2-7b") / "1000steps"
# save_dir = pathlib.Path("SFT-llama2-7b") / "with_early_stopping_with_more_datasets"
# save_dir = pathlib.Path("SFT-llama2-7b") / "1000steps_with_more_datasets"
# save_dir = pathlib.Path("SFT-llama2-7b") / "with_early_stopping_with_2_more_datasets"

# save_dir = pathlib.Path("SFT-llama2-13b") / "with_early_stopping_with_2_more_datasets"
# save_dir = pathlib.Path("models") / "SFT-llama2-13b" / "with_extra_+6.5k_+13.5k_+1k"

# save_dir = pathlib.Path("models") / "03-short-prompt" / "SFT-llama2-7b" / "with_extra_+6.5k_+13.5k_+1k"
# save_dir = pathlib.Path("models") / "03-short-prompt" / "SFT-llama2-13b" / "with_extra_+6.5k_+13.5k_+1k"
save_dir = pathlib.Path("models") / "03-short-prompt2" / "SFT-llama2-7b" / "with_extra_+6.5k_+13.5k_+1k"

if save_dir.exists():
    raise FileExistsError(save_dir)

config = {
    "model": {
        "family": "Llama2",
        # "size": "7B",
        "size": "13B",
    },
    "dataset": llm_science_exam.data.dataset.DatasetConfig(
        additional_datasets=[
            "radek1/additional-train-data-for-llm-science-exam",
            "radek1/15k-high-quality-examples",
            "leonidkulyk/wikipedia-stem-1k",
        ],
        # train_test_split=True,
        train_test_split=False,
        test_size=1,
    ),
    "prompt_id": 4,
}

dataset = llm_science_exam.data.dataset.get_dataset("train", config["dataset"])
dataset = llm_science_exam.llama2.dataset.add_prompt_field(
    dataset, prompt_id=config["prompt_id"], new_field_name="text", with_answer=True
)

save_dir.mkdir(exist_ok=True, parents=True)
with open(save_dir / "train_config.json", "w") as f:
    json.dump(config, f, indent=2)


model, tokenizer = llm_science_exam.llama2.model.get_model(config["model"])


# for llama 2 (they need different target module)
lora_config = LoraConfig(
    r=16,  # dimension of the updated matrices
    lora_alpha=64,  # parameter for scaling
    target_modules=llm_science_exam.llama2.model.find_linear_layers(
        model
    ),  # this chooses on which layers QLoRA is applied
    lora_dropout=0.1,  # dropout probability for layers
    bias="none",
    task_type="CAUSAL_LM",
)


training_args = TrainingArguments(
    output_dir=str(save_dir),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    # per_device_train_batch_size=8,
    # per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    # learning_rate=2e-4,
    # learning_rate=1e-4,
    learning_rate=5e-5,
    logging_steps=20,
    logging_strategy="steps",
    save_steps=100,
    save_strategy="steps",
    #
    # Learning-Rate Scheduler
    lr_scheduler_type="constant",
    # warmup_steps=2,
    warmup_ratio=0.03,
    #
    num_train_epochs=100,
    # max_steps=1000,
    optim="paged_adamw_8bit",
    fp16=True,
    # bf16=True,
    weight_decay=0.001,
    run_name="baseline-llama2-sft",
    save_total_limit=3,  # can be increased, but beware of kaggle notebook output size limit
    report_to=["none"],
    #
    # for Early Stopping
    load_best_model_at_end="valid" in dataset,
    evaluation_strategy="steps" if "valid" in dataset else "no",
    eval_steps=None,  # same as logging_steps
)

supervised_finetuning_trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    eval_dataset=dataset.get("valid", None),
    args=training_args,
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=3000,
    # data_collator=DataCollatorForCompletionOnlyLM(
    #     response_template=tokenizer.encode("[/INST] ", add_special_tokens=False), tokenizer=tokenizer
    # ),
    callbacks=[
        EarlyStoppingCallback(
            # early_stopping_patience=3
            # early_stopping_patience=10
            early_stopping_patience=50
        )
    ]
    if config["dataset"].get("train_test_split", False)
    else [],
)
supervised_finetuning_trainer.train()

# model_to_save = (
#     supervised_finetuning_trainer.model.module
#     if hasattr(supervised_finetuning_trainer.model, "module")
#     else supervised_finetuning_trainer.model
# )
# model_to_save.save_pretrained("outputs")
supervised_finetuning_trainer.save_state()
supervised_finetuning_trainer.save_model()
# TrainOutput(global_step=1, training_loss=2.932173013687134, metrics={'train_runtime': 7.1062, 'train_samples_per_second': 1.126, 'train_steps_per_second': 0.141, 'train_loss': 2.932173013687134, 'epoch': 0.04})

del model, supervised_finetuning_trainer
gc.collect()
torch.cuda.empty_cache()

llm_science_exam.llama2.model.merge_model(save_dir)
