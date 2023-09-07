import pathlib
import string

import faiss
import numpy as np
import pandas as pd

from .. import pj_struct_paths
from ..utils import timer
from . import predict


def get_faiss_index() -> faiss.IndexFlat:
    faiss_index_path = pj_struct_paths.get_data_dir_path() / "wikipedia-2023-07-faiss-index" / "wikipedia_202307.index"
    with timer(f"Loading faiss index from {faiss_index_path}"):
        index = faiss.read_index(str(faiss_index_path))
    return index


def get_wiki_index_df() -> pd.DataFrame:
    wiki_index_path = pj_struct_paths.get_data_dir_path() / "wikipedia-20230701" / "wiki_2023_index.parquet"
    return pd.read_parquet(wiki_index_path, columns=["id", "file"])


def get_wiki_fulltext_mapping_path() -> dict[str, pathlib.Path]:
    return {
        p.name: p
        for p in filter(
            pathlib.Path.exists,
            (
                pj_struct_paths.get_data_dir_path() / "wikipedia-20230701" / f"{stem}.parquet"
                for stem in [*string.ascii_lowercase, "number", "other"]
            ),
        )
    }


def get_possible_wiki_index_df_matched_with_prompts(model, prompts: list[str], id_list: list[int], n_top: int = 5):
    faiss_index = get_faiss_index()

    prompt_embeddings = predict.get_embeddings(model, prompts)

    search_score, search_index = faiss_index.search(prompt_embeddings, n_top)

    available_wiki_index_df = get_wiki_index_df().loc[search_index.flatten()]
    available_wiki_index_df["prompt_id"] = np.repeat(id_list, n_top)
    available_wiki_index_df = (
        available_wiki_index_df[["id", "prompt_id", "file"]]
        .drop_duplicates()
        .sort_values(["file", "id"])
        .reset_index(drop=True)
    )
    return available_wiki_index_df
