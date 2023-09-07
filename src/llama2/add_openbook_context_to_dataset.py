import llm_science_exam.data
import llm_science_exam.open_book

dataset = llm_science_exam.data.dataset.get_dataset(
    "train",
    config=llm_science_exam.data.dataset.DatasetConfig(additional_datasets=[], train_test_split=False, test_size=1),
)


model = llm_science_exam.open_book.model.get_model("cpu")
faiss_index = llm_science_exam.open_book.search.get_faiss_index()

# prompt_with_all_answers = [" ".join([x["prompt"], x["A"], x["B"], x["C"], x["D"], x["E"]]) for x in dataset["valid"]]

ds = dataset["valid"]


possible_wiki_index_df = llm_science_exam.open_book.search.get_possible_wiki_index_df_matched_with_prompts(
    model, ds["prompt"], ds["id"]
)

import pandas as pd
import tqdm

wiki_fulltext_df = (
    pd.concat(
        [
            pd.read_parquet(
                llm_science_exam.open_book.search.get_wiki_fulltext_mapping_path()[filename], columns=["id", "text"]
            ).query("id.isin(@id_list)")
            for filename, id_list in tqdm.tqdm(possible_wiki_index_df.groupby("file")["id"])
        ]
    )
    .sort_values("id")
    .reset_index(drop=True)
)
