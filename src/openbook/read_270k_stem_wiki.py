import pathlib

import numpy as np
import pandas as pd

import llm_science_exam.data
import llm_science_exam.llama2
import llm_science_exam.open_book_v2
import llm_science_exam.pj_struct_paths

df = pd.read_csv(llm_science_exam.pj_struct_paths.get_kaggle_dataset_dir_path() / "train.csv")


# path = pathlib.Path("tf-idf-context.npz")
#
# if path.exists():
#     tf_idf_contexts = np.load(path)["arr_0"]
# else:
#     tf_idf_contexts = llm_science_exam.open_book_v2.tf_idf.get_context(
#         df,
#         wiki_dataset_paths=[
#             llm_science_exam.pj_struct_paths.get_data_dir_path() / "all-paraphs-parsed-expanded",
#             llm_science_exam.pj_struct_paths.get_data_dir_path() / "llm-se-additional-wiki-stem-articles",
#         ],
#         join=False,
#         num_titles=10,
#     )
#     tf_idf_contexts = np.array(tf_idf_contexts)
#     np.savez(path, tf_idf_contexts)


# sf_faiss_contexts = llm_science_exam.open_book_v2.sf_faiss.get_context(
#     df,
#     wiki_dataset_paths=[
#         llm_science_exam.pj_struct_paths.get_data_dir_path() / "all-paraphs-parsed-expanded",
#         llm_science_exam.pj_struct_paths.get_data_dir_path() / "llm-se-additional-wiki-stem-articles",
#     ],
#     join=False,
#     num_titles=10,
# )
# sf_faiss_contexts = np.array(sf_faiss_contexts)

df["context"] = llm_science_exam.open_book_v2.sf_faiss.get_context(
    # df["context"] = llm_science_exam.open_book_v2.tf_idf.get_context(
    df,
    wiki_dataset_paths=[
        llm_science_exam.pj_struct_paths.get_data_dir_path() / "all-paraphs-parsed-expanded",
        llm_science_exam.pj_struct_paths.get_data_dir_path() / "llm-se-additional-wiki-stem-articles",
    ],
    # num_titles=3,
    num_titles=4,
    # max_context_per_title=1000,
)

# article_path = pathlib.Path("matched_articles.csv")

config = llm_science_exam.data.config.get_config("../llama2/config/a100/llama2-16bits.toml")
ckpt_path = llm_science_exam.data.config.get_checkpoint_path(config)

model, tokenizer = llm_science_exam.llama2.model.get_model_from_checkpoint(
    config["model"], pathlib.Path("../llama2") / ckpt_path / "checkpoint-7200"
)
# model, tokenizer = llm_science_exam.llama2.model.get_model(config["model"])

target_token_ids = np.ravel(tokenizer(["yes", "no"], add_special_tokens=False)["input_ids"])
import torch
import tqdm

yes_probs = []

prompt_template = llm_science_exam.llama2.prompts.get_prompt_template(config["model"]["family"], 9)
for idx in tqdm.trange(len(df)):
    row = df.iloc[idx]
    samples = [
        prompt_template.format(prompt=row["prompt"], context=row["context"], answer_text=row[alphabet], yes_or_no="")
        for alphabet in "ABCDE"
    ]

    for sample in samples:
        inputs = tokenizer(sample, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs.to(model.device))
            yes_probs.append(
                torch.softmax(outputs["logits"][:, -1, target_token_ids], dim=1)[:, 0].detach().cpu().numpy()
            )
            del inputs, outputs
            torch.cuda.empty_cache()


yes_probs = np.concatenate(yes_probs, axis=0).reshape(len(df), 5)
df["prediction"] = [" ".join(row) for row in np.take(list("ABCDE"), np.argsort(yes_probs, axis=1)[:, ::-1][:, :3])]

n = len(df)
for i in range(n):
    df.loc[i, "top_1"] = df.loc[i, "prediction"][0]
    df.loc[i, "top_2"] = df.loc[i, "prediction"][2]
    df.loc[i, "top_3"] = df.loc[i, "prediction"][4]

top_i = [(df[f"top_{i}"] == df["answer"]).sum() for i in [1, 2, 3]]
print(f"top1 : {top_i[0]}/{n}, top2 : {top_i[1]}/{n}, top3 : {top_i[2]}/{n} (total={sum(top_i)} / {n})")
print(f"Accuracy: {100*top_i[0]/n:.1f}%, map3: {100*(top_i[0] + top_i[1]*1/2 + top_i[2]*1/3).sum()/n:.1f}%")


# answer_probs = torch.softmax(torch.Tensor(yes_probs), dim=1).numpy()
# import plotly.express as px
# px.histogram(x=np.max(answer_probs, axis=1), color=df["top_1"] == df["answer"]).show()
# px.histogram(x=df["context"].map(lambda c: len(c.split())), color=df["top_1"] == df["answer"]).show()
