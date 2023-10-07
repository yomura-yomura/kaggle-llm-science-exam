import pandas as pd

import llm_science_exam.data
import llm_science_exam.open_book_v2
import llm_science_exam.pj_struct_paths

# dataset = llm_science_exam.data.dataset.get_dataset(
#     "train",
#     dataset_config=llm_science_exam.data.dataset.DatasetConfig(
#         prompt_id=-1,
#         additional_datasets=[
#             # "cdeotte/60k-data-with-context-v2",
#             # "cdeotte/60k-data-with-context-v2/only-stem-generated-by-chatgpt",
#             # "radek1/sci-or-not-sci-hypthesis-testing-pack/sci"
#             # "yalickj/dataset-wiki-new-1",
#             "takeshisuzuki/additional-dataset-800articles-4000rows/only-q1"
#         ],
#         train_test_split=False,
#         test_size=1,
#         with_context=False,
#         context=dict(version=1, top_n_sentences=3),
#     ),
# )
#
# dataset_type = "train"
# # dataset_type = "valid"
#
#
# ds = dataset[dataset_type]
#
# df = ds.to_pandas()
# df = df.iloc[:10]


csv_path = (
    llm_science_exam.pj_struct_paths.get_data_dir_path()
    # / "llm-se-extra-train-datasets"
    # / "takeshisuzuki"
    # / "additional-dataset-800articles-4000rows"
    # / "additional_dataset_800articles_4000rows.csv"
    #
    # / "llm-se-extra-train-datasets"
    # / "cdeotte"
    # / "60k-data-with-context-v2"
    # / "all_12_with_context2.csv"
    #
    # / "llm-se-extra-train-datasets/wuwenmin/llm-sci-eval300-gpt4-corrected/eval300_gpt4.csv"
    #
    / "llm-se-extra-train-datasets/yalickj/dataset-wiki-new-1/dataset_wiki_new_1_balanced.csv"
    #
    # / "kaggle-llm-science-exam"
    # / "train.csv"
)
print(csv_path)

df = pd.read_csv(csv_path)

contexts = llm_science_exam.open_book_v2.tf_idf.get_context(
    df,
    wiki_dataset_paths=[
        # llm_science_exam.pj_struct_paths.get_data_dir_path() / "all-paraphs-parsed-expanded",
        # llm_science_exam.pj_struct_paths.get_data_dir_path() / "llm-se-additional-wiki-stem-articles",
        llm_science_exam.pj_struct_paths.get_data_dir_path()
        / "stem-wiki-cohere-no-emb",
    ],
    join=False,
    num_titles=10,
    num_proc=6,
)

for idx, context_tuple in zip(df.index, contexts, strict=True):
    for top, c in enumerate(context_tuple):
        df.loc[idx, f"context_top{top + 1}"] = c
# df = df.drop(columns="context")

# for idx in df.index:
#     for top, (cat, score) in enumerate(df.loc[idx, "categories"].items()):
#         df.loc[idx, f"category_top{top + 1}"] = cat
#         df.loc[idx, f"category_score_top{top + 1}"] = score
# df = df.drop(columns="categories")

df.to_csv(csv_path.name, index=False)

# df.to_parquet("with_context.parquet")
# fsad
#
# contexts = llm_science_exam.open_book.get_context(ds, model_device="cuda", faiss_device="cuda")
#
# df = ds.to_pandas()
# df["context"] = contexts
# df = df[["prompt", "context", "A", "B", "C", "D", "E"]]
