from collections.abc import Sequence

import datasets
import numpy as np
import pandas as pd

from ..typing import FilePath
from ..utils import clean_memory
from .utils import retrieval

__all__ = ["get_context"]


def get_context(
    df: pd.DataFrame, wiki_dataset_paths: Sequence[FilePath], num_titles: int = 3, join: bool = True
) -> list[str | tuple[str]]:
    paraphs_parsed_dataset = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in wiki_dataset_paths])

    modified_texts = paraphs_parsed_dataset.map(
        lambda example: {
            "temp_text": f"{example['title']} {example['section']} {example['text']}".replace("\n", " ").replace(
                "'", ""
            )
        },
        num_proc=None,
    )["temp_text"]

    articles_indices, merged_top_scores = retrieval(df, modified_texts)
    del modified_texts
    clean_memory()

    top_per_query = articles_indices.shape[1]
    matched_article_df = pd.DataFrame(
        [
            (
                merged_top_scores.reshape(-1)[index],
                paraphs_parsed_dataset[idx.item()]["title"],
                paraphs_parsed_dataset[idx.item()]["text"],
            )
            for index, idx in enumerate(articles_indices.reshape(-1))
        ],
        columns=["score", "title", "text"],
    )
    matched_article_df["id"] = np.repeat(np.arange(len(df)), top_per_query)
    matched_article_df = matched_article_df.sort_values(["id", "score"], ascending=[True, False])

    contexts = [tuple(article_df["text"].iloc[:num_titles]) for _, article_df in matched_article_df.groupby("id")]
    if join:
        return ["\n".join(f"- {context}" for context in context_tuple) for context_tuple in contexts]
    else:
        return contexts
