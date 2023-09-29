import pathlib

import pandas as pd

import llm_science_exam.open_book

train_df = llm_science_exam.open_book.get_df_with_reduced_context(
    pd.read_csv("../../data/kaggle-llm-science-exam-with-context/train.csv"), upper_limit_of_n_words=500
)
test_df = llm_science_exam.open_book.get_df_with_reduced_context(
    pd.read_csv("../../data/kaggle-llm-science-exam-with-context/test.csv"), upper_limit_of_n_words=500
)

# new_dir = pathlib.Path("../../data/kaggle-llm-science-exam-with-context-500w")
# new_dir.mkdir()
# train_df.to_csv(new_dir / "train.csv", index=False)
# test_df.to_csv(new_dir / "test.csv", index=False)
