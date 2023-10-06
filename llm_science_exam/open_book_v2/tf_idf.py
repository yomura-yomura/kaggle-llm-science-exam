import pathlib
import unicodedata
from collections.abc import Sequence

import datasets
import pandas as pd

from ..typing import FilePath
from ..utils import clean_memory
from .utils import retrieval

__all__ = ["get_context"]


def get_context(
    df: pd.DataFrame,
    wiki_dataset_paths: Sequence[FilePath],
    num_titles: int = 3,
    join: bool = True,
    num_proc: int | None = None,
) -> list[str | tuple[str]]:
    if "stem-wiki-cohere-no-emb" in [pathlib.Path(p).name for p in wiki_dataset_paths]:
        assert len(wiki_dataset_paths) == 1
        wiki_dataset_version = 0
    else:
        wiki_dataset_version = 1

    print(f"{wiki_dataset_version = }")

    paraphs_parsed_dataset = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in wiki_dataset_paths])

    if wiki_dataset_version == 0:
        modified_texts = paraphs_parsed_dataset.map(
            lambda example: {
                "temp_text": unicodedata.normalize("NFKD", f"{example['title']} {example['text']}").replace('"', "")
            },
            num_proc=num_proc,
        )["temp_text"]
    elif wiki_dataset_version == 1:
        modified_texts = paraphs_parsed_dataset.map(
            lambda example: {
                "temp_text": f"{example['title']} {example['section']} {example['text']}".replace("\n", " ").replace(
                    "'", ""
                )
            },
            num_proc=num_proc,
        )["temp_text"]
    else:
        raise ValueError(f"wiki_dataset_version {wiki_dataset_version}")

    # articles_indices, merged_top_scores = retrieval(df, modified_texts)
    articles_indices, _ = retrieval(df, modified_texts)

    del modified_texts
    clean_memory()

    if wiki_dataset_version == 0:
        contexts_generator = (
            tuple(
                unicodedata.normalize("NFKD", paraphs_parsed_dataset[int(article_id)]["text"])
                for article_id in indices[:num_titles]
            )
            for i, indices in enumerate(articles_indices)
        )
    elif wiki_dataset_version == 1:
        contexts_generator = (
            tuple(paraphs_parsed_dataset[int(article_id)]["text"] for article_id in indices[:num_titles])
            for i, indices in enumerate(articles_indices)
        )
    else:
        raise ValueError(f"wiki_dataset_version {wiki_dataset_version}")

    if join:
        return ["\n".join(f"- {context}" for context in context_tuple) for context_tuple in contexts_generator]
    else:
        return list(contexts_generator)
