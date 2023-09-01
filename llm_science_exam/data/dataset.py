from typing import Literal, TypedDict

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from .. import pj_struct_paths

__all__ = ["DatasetConfig", "get_dataset"]


class DatasetConfig(TypedDict):
    additional_datasets: list[
        Literal[
            "radek1/additional-train-data-for-llm-science-exam",
            "radek1/15k-high-quality-examples",
            "leonidkulyk/wikipedia-stem-1k",
        ]
    ] | None
    train_test_split: bool
    test_size: float


def get_dataset(dataset_type: Literal["train", "test"], config: DatasetConfig, seed: int = 42) -> DatasetDict:
    df = pd.read_csv(pj_struct_paths.get_kaggle_dataset_dir_path() / f"{dataset_type}.csv")

    if dataset_type == "train":
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
                train_df = pd.DataFrame(columns=["id", "prompt", "A", "B", "C", "D", "E", "answer"])
                valid_df = df

        # Additional Datasets

        additional_datasets = config.get("additional_datasets", [])

        if "radek1/additional-train-data-for-llm-science-exam" in additional_datasets:
            extra_data_dir_path = (
                pj_struct_paths.get_data_dir_path() / "radek1" / "additional-train-data-for-llm-science-exam"
            )
            extra1_df = pd.read_csv(extra_data_dir_path / "extra_train_set.csv").reset_index(names="id")
            assert len(extra1_df) == 500
            extra1_df["id"] += 1000
            extra2_df = pd.read_csv(extra_data_dir_path / "6000_train_examples.csv").reset_index(names="id")
            assert len(extra2_df) == 6_000
            extra2_df["id"] += 10000

            train_df = pd.concat([train_df, extra1_df, extra2_df]).reset_index(drop=True)
            assert train_df["id"].nunique() == len(train_df), "duplicate id detected."

        if "radek1/15k-high-quality-examples" in additional_datasets:
            extra_data_dir_path = pj_struct_paths.get_data_dir_path() / "radek1" / "15k-high-quality-examples"
            extra_df = pd.read_csv(extra_data_dir_path / "15k_gpt3.5-turbo.csv").reset_index(names="id")
            assert len(extra_df) == 15_000
            extra_df["id"] += 100000

            train_df = pd.concat([train_df, extra_df]).reset_index(drop=True)
            assert train_df["id"].nunique() == len(train_df), "duplicate id detected."

        if "leonidkulyk/wikipedia-stem-1k" in additional_datasets:
            extra_data_dir_path = pj_struct_paths.get_data_dir_path() / "leonidkulyk" / "wikipedia-stem-1k"
            extra_df = pd.read_csv(extra_data_dir_path / "stem_1k_full_v1.csv").reset_index(names="id")
            assert len(extra_df) == 1_000
            extra_df["id"] += 150000

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
            assert train_df["id"].nunique() == len(train_df), "duplicate id detected."

        # shuffle
        if len(additional_datasets) > 0:
            train_df = train_df.sample(frac=1, random_state=seed)

        dataset = DatasetDict()
        if train_df is not None:
            dataset["train"] = Dataset.from_dict(train_df)
        if valid_df is not None:
            dataset["valid"] = Dataset.from_dict(valid_df)

    elif dataset_type == "test":
        dataset = DatasetDict({"test": Dataset.from_dict(df)})
    else:
        raise ValueError(f"unexpected dataset_type '{dataset_type}'")

    return dataset
