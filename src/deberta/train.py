import argparse
import json
import pathlib

import numpy as np
from datasets import Dataset
from transformers import EarlyStoppingCallback, Trainer

import llm_science_exam.data
import llm_science_exam.model.deberta
import llm_science_exam.score

llm_science_exam.model.deberta.custom_forward_method.enable_memory_efficient_forward_method()

# MAX_LENGTH = 256
# MAX_LENGTH = 1024 * 2

parser = argparse.ArgumentParser()
parser.add_argument("config_path")
parser.add_argument("--resume-from-checkpoint", type=str, default=None)
parser.add_argument("--use-map-at-3-as-metric", action="store_true", default=True)
args = parser.parse_args()

use_map_at_3_as_metric = args.use_map_at_3_as_metric
resume_from_checkpoint = args.resume_from_checkpoint

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

model, tokenizer = llm_science_exam.model.deberta.model.get_model(
    config["model"], model_name=config["train"].get("pretrained_model", None)
)

# print("We are using PEFT.")
# from peft import LoraConfig, TaskType, get_peft_model
#
# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=64,
#     task_type=TaskType.SEQ_CLS,
#     lora_dropout=0.1,
#     bias="none",
#     inference_mode=False,
#     target_modules=["query_proj", "value_proj"],
#     # target_modules=llm_science_exam.model.llama2.model.find_linear_layers(model),
#     modules_to_save=["classifier", "pooler"],
# )
# model = get_peft_model(model, peft_config)

# BOOLEAN TO FREEZE EMBEDDINGS
FREEZE_EMBEDDINGS = config["model"]["freeze_embeddings"]
if FREEZE_EMBEDDINGS:
    print("Freezing embeddings.")
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False

# DEBERTA LARGE HAS TOTAL OF 24 LAYERS
FREEZE_LAYERS = config["model"]["freeze_layers"]
if FREEZE_LAYERS > 0:
    print(f"Freezing {FREEZE_LAYERS} layers.")
    for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False


dataset = llm_science_exam.data.dataset.get_dataset("train", config["dataset"])
for dataset_name in dataset.keys():
    df = dataset[dataset_name].to_pandas()
    print(f"{dataset_name} dropna: {len(df)} -> ", end="")
    # df = df.dropna()

    df.fillna(value="", inplace=True)
    # df = df.iloc[:150]
    print(f"len({dataset_name}) = {len(df)}")
    dataset[dataset_name] = Dataset.from_pandas(df)


tokenized_dataset = llm_science_exam.model.deberta.dataset.map_preprocess(
    dataset,
    tokenizer,
    with_answer=True,
    max_length=config["dataset"]["max_length"],
    num_proc=12,
)

print(f"prompt example:\n{tokenizer.decode(tokenized_dataset['train']['input_ids'][0][0])}")

save_dir.mkdir(exist_ok=True, parents=True)
with open(save_dir / "train_config.json", "w") as f:
    json.dump(config, f, indent=2)


label_array = np.array(list("ABCDE"))


def compute_metrics(p):
    predictions = label_array[np.argsort(p.predictions.tolist())[:, ::-1]]
    labels = label_array[np.expand_dims(p.label_ids.tolist(), axis=1)]
    if len(predictions) > 200:
        return {
            "MAP@3": llm_science_exam.score.map_at_3(labels, predictions),
            "old_MAP@3": llm_science_exam.score.map_at_3(labels[:200], predictions[:200]),
        }
    else:
        return {"MAP@3": llm_science_exam.score.map_at_3(labels, predictions)}


# def map_at_3(predictions, labels):
#     map_sum = 0
#     pred = np.argsort(-1 * np.array(predictions), axis=1)[:, :3]
#     for x, y in zip(pred, labels):
#         z = [1 / i if y == j else 0 for i, j in zip([1, 2, 3], x)]
#         map_sum += np.sum(z)
#     return map_sum / len(predictions)
#
#
# def compute_metrics(p):
#     predictions = p.predictions.tolist()
#     labels = p.label_ids.tolist()
#     return {"MAP@3": map_at_3(predictions, labels)}


trainer = Trainer(
    model=model,
    args=llm_science_exam.model.deberta.train.get_training_args(
        config,
        output_dir=save_dir,
        use_early_stopping="valid" in dataset,
        use_map_at_3_as_metric=use_map_at_3_as_metric,
    ),
    tokenizer=tokenizer,
    data_collator=llm_science_exam.model.deberta.dataset.DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config["train"]["early_stopping_patience"])],
)
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# trainer.save_state()
# trainer.save_model()
