from typing import Callable, Literal, TypedDict

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from .. import pj_struct_paths
from ..typing import FilePath

__all__ = ["DatasetConfig", "get_dataset"]


class DatasetConfig(TypedDict, total=False):
    prompt_id: int
    additional_datasets: list[str] | None
    valid_additional_datasets: list[str] | None

    train_test_split: bool
    test_size: float
    with_context: bool

    context: "DatasetContextConfig"


class DatasetContextConfig(TypedDict, total=False):
    version: int
    upper_limit_of_n_words: int
    top_n_sentences: int


def get_dataset(
    dataset_type: Literal["train", "valid", "test"],
    dataset_config: DatasetConfig,
    *,
    seed: int = 42,
) -> DatasetDict:
    if dataset_config["with_context"]:
        columns = ["id", "prompt", "context", "A", "B", "C", "D", "E"]
    else:
        columns = ["id", "prompt", "A", "B", "C", "D", "E"]

    if dataset_type in ["train", "valid"]:
        columns.append("answer")

    data_dir_path = pj_struct_paths.get_data_dir_path()
    kaggle_dataset_dir_path = data_dir_path / "kaggle-llm-science-exam"
    additional_dataset_dir_path = data_dir_path / "llm-se-extra-train-datasets"

    if dataset_config["with_context"]:
        match dataset_config["context"].get("version", 1):
            case 1:
                pass
            case i if isinstance(i, int) and 1 <= i <= 4:
                kaggle_dataset_dir_path = data_dir_path / f"llm-se-datasets-with-context-v{i}"
                additional_dataset_dir_path = kaggle_dataset_dir_path / "extra"
            case _:
                raise ValueError(f"Unknown context version: {dataset_config['context']['version']}")

    def add_extra_df(
        base_df: pd.DataFrame,
        extra_dataset_path: FilePath,
        dataset_length: int,
        df_collator: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        extra_dataset_path = additional_dataset_dir_path / extra_dataset_path
        extra_df = pd.read_csv(extra_dataset_path).drop(columns="id", errors="ignore").reset_index(names="id")
        if df_collator is not None:
            extra_df = df_collator(extra_df)

        assert len(extra_df) == dataset_length, len(extra_df)
        # extra_df = extra_df[extra_df["source"].isin([5, 6, 7, 8])]
        # assert len(extra_df) == 16_139, len(extra_df)

        if dataset_config["with_context"] and "context" not in extra_df.columns:
            extra_df["context"] = get_context_from_top_contexts(extra_df, dataset_config["context"]["top_n_sentences"])

        if len(base_df) > 0:
            extra_df["id"] += base_df["id"].max() + 1

        extra_df = extra_df[columns]

        return pd.concat([base_df, extra_df]).reset_index(drop=True)

    def add_all_extra_df(train_df, additional_datasets):
        if "cdeotte/60k-data-with-context-v2" in additional_datasets:
            train_df = add_extra_df(train_df, "cdeotte/60k-data-with-context-v2/all_12_with_context2.csv", 60_347)

        if "cdeotte/60k-data-with-context-v2/only-stem-generated-by-chatgpt" in additional_datasets:
            train_df = add_extra_df(
                train_df, "cdeotte/60k-data-with-context-v2/only_stem_generated_by_chatgpt.csv", 16_139
            )

        if "radek1/sci-or-not-sci-hypthesis-testing-pack/sci" in additional_datasets:
            train_df = add_extra_df(
                train_df, "radek1/sci-or-not-sci-hypthesis-testing-pack/6000_wiki_en_sci_questions.csv", 6_000
            )

        if "yalickj/dataset-wiki-new-1" in additional_datasets:
            train_df = add_extra_df(train_df, "yalickj/dataset-wiki-new-1/dataset_wiki_new_1_balanced.csv", 300)

        if "takeshisuzuki/additional-dataset-800articles-4000rows" in additional_datasets:
            train_df = add_extra_df(
                train_df,
                "takeshisuzuki/additional-dataset-800articles-4000rows/additional_dataset_800articles_4000rows.csv",
                4_005,
            )
            train_df = train_df.dropna()

        if "takeshisuzuki/additional-dataset-800articles-4000rows/only-q1" in additional_datasets:
            train_df = add_extra_df(
                train_df,
                "takeshisuzuki/additional-dataset-800articles-4000rows/additional_dataset_800articles_4000rows.csv",
                801,
                df_collator=lambda df: df.groupby("page_title").head(1),
            )
        return train_df

    df = pd.read_csv(kaggle_dataset_dir_path / f"{'train' if dataset_type in ['train', 'valid'] else 'test'}.csv")
    if dataset_config["with_context"] and "context" not in df.columns:
        df["context"] = get_context_from_top_contexts(df, dataset_config["context"]["top_n_sentences"])
    df = df[columns]

    if dataset_type in ("train", "valid"):
        valid_df = pd.DataFrame([])
        if dataset_config.get("train_test_split", True):
            train_df, valid_df = train_test_split(
                df, test_size=dataset_config.get("test_size", 0.2), random_state=seed, shuffle=True
            )
        else:
            assert dataset_config["test_size"] in [0, 1]
            if dataset_config["test_size"] == 0:
                train_df = df
            else:
                train_df = pd.DataFrame(columns=columns)
                valid_df = df

        if dataset_type == "train":
            additional_datasets = dataset_config.get("additional_datasets", [])

            train_df = add_all_extra_df(train_df, additional_datasets)
            # if "radek1/additional-train-data-for-llm-science-exam" in additional_datasets:
            #     extra_data_dir_path = (
            #         pj_struct_paths.get_data_dir_path()
            #         / "llm-se-extra-train-datasets"
            #         / "radek1"
            #         / "additional-train-data-for-llm-science-exam"
            #     )
            #     extra1_df = pd.read_csv(extra_data_dir_path / "extra_train_set.csv").reset_index(names="id")
            #     assert len(extra1_df) == 500
            #     extra1_df["id"] += 1_000
            #     extra2_df = pd.read_csv(extra_data_dir_path / "6000_train_examples.csv").reset_index(names="id")
            #     assert len(extra2_df) == 6_000
            #     extra2_df["id"] += 10_000
            #
            #     train_df = pd.concat([train_df, extra1_df, extra2_df]).reset_index(drop=True)
            #
            # if "radek1/15k-high-quality-examples" in additional_datasets:
            #     extra_data_dir_path = (
            #         pj_struct_paths.get_data_dir_path()
            #         / "llm-se-extra-train-datasets"
            #         / "radek1"
            #         / "15k-high-quality-examples"
            #     )
            #     extra_df = pd.read_csv(extra_data_dir_path / "15k_gpt3.5-turbo.csv").reset_index(names="id")
            #     assert len(extra_df) == 15_000
            #     extra_df["id"] += 100_000
            #
            #     train_df = pd.concat([train_df, extra_df]).reset_index(drop=True)
            #
            # if "leonidkulyk/wikipedia-stem-1k" in additional_datasets:
            #     extra_data_dir_path = (
            #         pj_struct_paths.get_data_dir_path()
            #         / "llm-se-extra-train-datasets"
            #         / "leonidkulyk"
            #         / "wikipedia-stem-1k"
            #     )
            #     extra_df = pd.read_csv(extra_data_dir_path / "stem_1k_full_v1.csv").reset_index(names="id")
            #     assert len(extra_df) == 1_000
            #     extra_df["id"] += 150_000
            #
            #     extra_df = extra_df.iloc[:, : len(train_df.columns)]
            #     assert (
            #         extra_df.columns
            #         == [
            #             "id",
            #             "question",
            #             "option_1",
            #             "option_2",
            #             "option_3",
            #             "option_4",
            #             "option_5",
            #             "answer",
            #         ]
            #     ).all()
            #     extra_df.columns = ["id", "prompt", "A", "B", "C", "D", "E", "answer"]
            #     extra_df["answer"] = extra_df["answer"].map(
            #         {"option_1": "A", "option_2": "B", "option_3": "C", "option_4": "D", "option_5": "E"}
            #     )
            #     assert len(extra_df.dropna()) == len(extra_df)
            #
            #     train_df = pd.concat([train_df, extra_df]).reset_index(drop=True)
            # if "cdeotte/40k-data-with-context-v2/ScienceQA" in additional_datasets:
            #     extra_data_dir_path = (
            #         pj_struct_paths.get_data_dir_path()
            #         / "llm-se-extra-train-datasets"
            #         / "cdeotte"
            #         / "40k-data-with-context-v2"
            #     )
            #     extra_df = pd.read_csv(extra_data_dir_path / "ScienceQA_with_context2.csv")
            #     extra_df = (
            #         extra_df[extra_df["subject"] == "natural science"]
            #         .query("image.notnull()")[columns[1:]]
            #         .reset_index(drop=True)
            #         .reset_index(names="id")
            #     )
            #     assert len(extra_df) == 6332, len(extra_df)
            #     extra_df["id"] += 300_000
            #
            #     train_df = pd.concat([train_df, extra_df]).reset_index(drop=True)

            # shuffle
            if len(additional_datasets) > 0:
                train_df = train_df.sample(frac=1, random_state=seed)

            assert len(train_df) > 0

            assert train_df["id"].nunique() == len(
                train_df
            ), f"duplicate id detected: {train_df['id'].value_counts()[:10]}"

            # print(f"drop nan: {len(train_df):,} -> {len(train_df.dropna()):,}")
            # train_df = train_df.dropna()

        valid_additional_datasets = dataset_config.get("valid_additional_datasets", [])
        valid_df = add_all_extra_df(valid_df, valid_additional_datasets)

        dataset = DatasetDict()
        if train_df is not None:
            dataset["train"] = Dataset.from_pandas(train_df, preserve_index=False)
        if len(valid_df) > 0:
            dataset["valid"] = Dataset.from_pandas(valid_df, preserve_index=False)

    elif dataset_type == "test":
        dataset = DatasetDict({"test": Dataset.from_dict(df)})
    else:
        raise ValueError(f"unexpected dataset_type '{dataset_type}'")

    return dataset


def get_context_from_top_contexts(df: pd.DataFrame, top_n_sentences: int) -> pd.Series:
    return df[[f"context_top{top + 1}" for top in range(top_n_sentences)]].agg(
        lambda contexts: "\n".join(f"- {c}" for c in contexts if isinstance(c, str)), axis=1
    )
