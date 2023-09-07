import llm_science_exam.data
import llm_science_exam.llama2.model

config_path = "config/llama2.toml"

config = llm_science_exam.data.config.get_config(config_path)
config["dataset"] = llm_science_exam.data.config.DatasetConfig(prompt_id=5, train_test_split=False, test_size=0)

model, tokenizer = llm_science_exam.llama2.model.get_model(model_config=config["model"])

dataset = llm_science_exam.data.dataset.get_dataset("train", config=config["dataset"])
dataset = llm_science_exam.llama2.dataset.add_prompt_field(
    dataset,
    model_family_name=config["model"]["family"],
    prompt_id=config["dataset"]["prompt_id"],
    new_field_name="text",
    with_answer=False,
)

idx = 10
example = dataset["train"][idx]
text = example["text"]

inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to(
    model.device
)
from transformers import GenerationConfig

outputs = model.generate(
    **inputs,
    return_dict_in_generate=True,
    max_length=1024,
    output_scores=True,
    generation_config=GenerationConfig(do_sample=False, temperature=1, top_p=1),
)
print("-" * 10)
print(tokenizer.decode(outputs["sequences"][0]))
print("-" * 10)
print("Answer: " + dataset["train"]["answer"][idx])

predicted = llm_science_exam.llama2.predict.get_predicted_labels(
    model, tokenizer, example, model_family=config["model"]["family"], prompt_id=config["dataset"]["prompt_id"]
)
print(f"Predicted: {predicted}")
