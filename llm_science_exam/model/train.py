from transformers import TrainingArguments

from ..typing import FilePath


def get_training_args(
    config, *, output_dir: FilePath, use_early_stopping: bool, use_map_at_3_as_metric: bool
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
        #
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
        warmup_steps=config["train"].get("warmup_steps", 0),
        warmup_ratio=config["train"].get("warmup_ratio", 0),
        # warmup_ratio=config["train"]["warmup_ratio"],
        #
        num_train_epochs=config["train"]["num_train_epochs"],
        # max_steps=1000,
        optim=config["train"]["optim"],
        fp16=config["train"]["fp16"],
        bf16=False,
        # fp16=False,
        # bf16=True,
        weight_decay=config["train"]["weight_decay"],
        # run_name="baseline-llama2-sft",
        report_to=["none"],
        #
        # for Early Stopping
        load_best_model_at_end=use_early_stopping,
        evaluation_strategy="steps" if use_early_stopping else "no",
        eval_steps=None,  # same as logging_steps
        #
        # Metric
        metric_for_best_model="MAP@3" if use_map_at_3_as_metric else None,
        greater_is_better=True if use_map_at_3_as_metric else None,
    )
