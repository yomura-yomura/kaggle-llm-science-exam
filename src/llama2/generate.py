from transformers import GenerationConfig

import llm_science_exam.data
import llm_science_exam.model.llama2.model
import llm_science_exam.pj_struct_paths

llm_science_exam.pj_struct_paths.set_pj_struct_paths(
    kaggle_dataset_dir_path=llm_science_exam.pj_struct_paths.get_data_dir_path()
    / "kaggle-llm-science-exam-with-context"
)


# config_path = "config/llama2.toml"
# config_path = "config/openorca-platypus2.toml"
config_path = "config/llama2-with-context-300w.toml"


config = llm_science_exam.data.config.get_config(config_path)
config["dataset"] = llm_science_exam.data.config.DatasetConfig(
    # prompt_id=6,
    prompt_id=7,
    train_test_split=False,
    test_size=0,
    # with_context=True
    with_context=True,
)

model, tokenizer = llm_science_exam.model.llama2.model.get_model(model_config=config["model"])


dataset = llm_science_exam.data.dataset.get_dataset("train", dataset_config=config["dataset"])
dataset = llm_science_exam.model.llama2.dataset.add_prompt_field(
    dataset,
    model_family_name=config["model"]["family"],
    prompt_id=config["dataset"]["prompt_id"],
    new_field_name="text",
    with_answer=False,
)

idx = 2
example = dataset["train"][idx]
text = example["text"]

inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to(
    model.device
)

outputs = model.generate(
    **inputs,
    return_dict_in_generate=True,
    max_length=1024 * 3,
    output_scores=True,
    generation_config=GenerationConfig(do_sample=False, temperature=1, top_p=1),
)
print("-" * 10)
print(tokenizer.decode(outputs["sequences"][0]))
print("-" * 10)
print("Answer: " + dataset["train"]["answer"][idx])

predicted = llm_science_exam.model.llama2.predict.get_predicted_probs(
    model,
    tokenizer,
    example,
    model_family=config["model"]["family"],
    prompt_id=config["dataset"]["prompt_id"],
    print_perplexities=True,
)
print(f"Predicted: {predicted}")
