from typing import Literal, TypedDict

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from .. import pj_struct_paths

__all__ = ["DatasetConfig", "get_dataset"]


class DatasetConfig(TypedDict, total=False):
    prompt_id: int
    additional_datasets: list[
        Literal[
            "radek1/additional-train-data-for-llm-science-exam",
            "radek1/15k-high-quality-examples",
            "leonidkulyk/wikipedia-stem-1k",
            "cdeotte/60k-data-with-context-v2",
        ]
    ] | None
    train_test_split: bool
    test_size: float
    with_context: bool

    context: "DatasetContextConfig"


class DatasetContextConfig(TypedDict, total=False):
    upper_limit_of_n_words: int
    top_n_sentences: int


def get_dataset(
    dataset_type: Literal["train", "valid", "test"],
    config: DatasetConfig,
    *,
    seed: int = 42,
) -> DatasetDict:
    if config.get("with_context", False):
        columns = ["id", "prompt", "context", "A", "B", "C", "D", "E"]
    else:
        columns = ["id", "prompt", "A", "B", "C", "D", "E"]

    if dataset_type in ["train", "valid"]:
        columns.append("answer")

    df = pd.read_csv(
        # pj_struct_paths.get_kaggle_dataset_dir_path()
        # / f"{'train' if dataset_type in ['train', 'valid'] else 'test'}.csv"
        pj_struct_paths.get_data_dir_path()
        / "datasets_with_context"
        / f"{'train' if dataset_type in ['train', 'valid'] else 'test'}.csv"
    )
    df["context"] = get_context_from_top_contexts(df, config["context"]["top_n_sentences"])
    df = df[columns]

    if dataset_type in ("train", "valid"):
        valid_df = None
        if config.get("train_test_split", True):
            train_df, valid_df = train_test_split(
                df, test_size=config.get("test_size", 0.2), random_state=seed, shuffle=True
            )
        else:
            assert config["test_size"] in [0, 1]
            if config["test_size"] == 0:
                train_df = df
            else:
                train_df = pd.DataFrame(columns=columns)
                valid_df = df

        if dataset_type == "train":
            # Additional Datasets

            additional_datasets = config.get("additional_datasets", [])

            if "radek1/additional-train-data-for-llm-science-exam" in additional_datasets:
                extra_data_dir_path = (
                    pj_struct_paths.get_data_dir_path()
                    / "llm-se-extra-train-datasets"
                    / "radek1"
                    / "additional-train-data-for-llm-science-exam"
                )
                extra1_df = pd.read_csv(extra_data_dir_path / "extra_train_set.csv").reset_index(names="id")
                assert len(extra1_df) == 500
                extra1_df["id"] += 1_000
                extra2_df = pd.read_csv(extra_data_dir_path / "6000_train_examples.csv").reset_index(names="id")
                assert len(extra2_df) == 6_000
                extra2_df["id"] += 10_000

                train_df = pd.concat([train_df, extra1_df, extra2_df]).reset_index(drop=True)

            if "radek1/15k-high-quality-examples" in additional_datasets:
                extra_data_dir_path = (
                    pj_struct_paths.get_data_dir_path()
                    / "llm-se-extra-train-datasets"
                    / "radek1"
                    / "15k-high-quality-examples"
                )
                extra_df = pd.read_csv(extra_data_dir_path / "15k_gpt3.5-turbo.csv").reset_index(names="id")
                assert len(extra_df) == 15_000
                extra_df["id"] += 100_000

                train_df = pd.concat([train_df, extra_df]).reset_index(drop=True)

            if "leonidkulyk/wikipedia-stem-1k" in additional_datasets:
                extra_data_dir_path = (
                    pj_struct_paths.get_data_dir_path()
                    / "llm-se-extra-train-datasets"
                    / "leonidkulyk"
                    / "wikipedia-stem-1k"
                )
                extra_df = pd.read_csv(extra_data_dir_path / "stem_1k_full_v1.csv").reset_index(names="id")
                assert len(extra_df) == 1_000
                extra_df["id"] += 150_000

                extra_df = extra_df.iloc[:, : len(train_df.columns)]
                assert (
                    extra_df.columns
                    == [
                        "id",
                        "question",
                        "option_1",
                        "option_2",
                        "option_3",
                        "option_4",
                        "option_5",
                        "answer",
                    ]
                ).all()
                extra_df.columns = ["id", "prompt", "A", "B", "C", "D", "E", "answer"]
                extra_df["answer"] = extra_df["answer"].map(
                    {"option_1": "A", "option_2": "B", "option_3": "C", "option_4": "D", "option_5": "E"}
                )
                assert len(extra_df.dropna()) == len(extra_df)

                train_df = pd.concat([train_df, extra_df]).reset_index(drop=True)

            if "cdeotte/60k-data-with-context-v2" in additional_datasets:
                extra_data_dir_path = (
                    pj_struct_paths.get_data_dir_path()
                    # / "llm-se-extra-train-datasets"
                    / "datasets_with_context"
                    / "extra"
                    / "cdeotte"
                    / "60k-data-with-context-v2"
                )
                # extra_df = pd.read_csv(extra_data_dir_path / "all_12_with_context2.csv").reset_index(names="id")
                extra_df = (
                    pd.read_csv(extra_data_dir_path / "all_12_with_context2.csv")
                    .drop(columns="id")
                    .reset_index(names="id")
                )
                extra_df["context"] = get_context_from_top_contexts(extra_df, config["context"]["top_n_sentences"])
                # assert len(extra_df) == 60_347, len(extra_df)
                # extra_df = extra_df[extra_df["source"].isin([5, 6, 7, 8])]
                assert len(extra_df) == 16_139, len(extra_df)

                extra_df["id"] += 200_000

                extra_df = extra_df[columns]

                train_df = pd.concat([train_df, extra_df]).reset_index(drop=True)

            if "cdeotte/40k-data-with-context-v2/ScienceQA" in additional_datasets:
                extra_data_dir_path = (
                    pj_struct_paths.get_data_dir_path()
                    / "llm-se-extra-train-datasets"
                    / "cdeotte"
                    / "40k-data-with-context-v2"
                )
                extra_df = pd.read_csv(extra_data_dir_path / "ScienceQA_with_context2.csv")
                extra_df = (
                    extra_df[extra_df["subject"] == "natural science"]
                    .query("image.notnull()")[columns[1:]]
                    .reset_index(drop=True)
                    .reset_index(names="id")
                )
                assert len(extra_df) == 6332, len(extra_df)
                extra_df["id"] += 300_000

                train_df = pd.concat([train_df, extra_df]).reset_index(drop=True)

            # shuffle
            if len(additional_datasets) > 0:
                train_df = train_df.sample(frac=1, random_state=seed)

            assert train_df["id"].nunique() == len(train_df), "duplicate id detected."

            # print(f"drop nan: {len(train_df):,} -> {len(train_df.dropna()):,}")
            # train_df = train_df.dropna()

        dataset = DatasetDict()
        if train_df is not None:
            dataset["train"] = Dataset.from_pandas(train_df, preserve_index=False)
        if valid_df is not None:
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
