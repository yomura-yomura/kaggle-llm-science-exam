import pathlib
import string

import faiss
import numpy as np
import pandas as pd
import polars as pl
import torch

from llm_science_exam.utils import timer

from .. import pj_struct_paths
from ..typing import NDArray


def get_faiss_index(device: torch.device | str | int) -> faiss.IndexFlat:
    faiss_index_path = pj_struct_paths.get_data_dir_path() / "wikipedia-2023-07-faiss-index" / "wikipedia_202307.index"
    with timer(f"Loading faiss index from {faiss_index_path}"):
        index = faiss.read_index(str(faiss_index_path))

    index = change_faiss_index_on_device(index, device)

    return index


def change_faiss_index_on_device(index: faiss.IndexFlat, device: torch.device | str | int):
    device = torch.device(device)
    if device.type == "cpu":
        pass
    elif device.type == "cuda":
        device_index = device.index or 0
        res = faiss.StandardGpuResources()  # use a single GPU
        index = faiss.index_cpu_to_gpu(res, device_index, index)
    return index


def faiss_search(
    faiss_index: faiss.IndexFlat,
    prompt_embeddings: NDArray[np.float_],
    *,
    n_top: int,
    log_steps: int = 50,
    show_progress: bool = True,
) -> tuple[NDArray[np.float_], NDArray[np.int_]]:
    if show_progress:
        search_score = np.zeros((len(prompt_embeddings), n_top), dtype=prompt_embeddings.dtype)
        search_index = np.zeros((len(prompt_embeddings), n_top), dtype=int)

        index_edges = np.arange(0, np.ceil(len(prompt_embeddings) / log_steps).astype(int) + 1) * log_steps
        assert len(prompt_embeddings) <= index_edges.max()
        for left_idx, right_idx in tqdm.tqdm(
            zip(index_edges[:-1], index_edges[1:]), total=len(index_edges) - 1, desc="faiss_index.search"
        ):
            search_score[left_idx:right_idx], search_index[left_idx:right_idx] = faiss_index.search(
                prompt_embeddings[left_idx:right_idx], n_top
            )
    else:
        search_score, search_index = faiss_index.search(prompt_embeddings, n_top)
    return search_score, search_index


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


def get_wiki_fulltext_df(possible_wiki_index_df: pd.DataFrame):
    with timer(f"reading all matched context"):
        wiki_fulltext_df = (
            pl.concat(
                [
                    pl.scan_parquet(get_wiki_fulltext_mapping_path()[filename])
                    .select(["id", "text"])
                    .filter(pl.col("id").is_in(id_list))
                    .lazy()
                    for filename, id_list in possible_wiki_index_df.groupby("file")["id"]
                ]
            )
            .collect()
            .to_pandas()
            .sort_values("id")
            .reset_index(drop=True)
        )
    return wiki_fulltext_df


def get_possible_wiki_index_df_matched_with_prompts(
    prompt_embeddings: NDArray[np.float_], id_list: list[int], *, faiss_device: str, n_top: int = 5
):
    faiss_index = get_faiss_index(device=faiss_device)

    search_score, search_index = faiss_search(faiss_index, prompt_embeddings, n_top=n_top)

    available_wiki_index_df = get_wiki_index_df().loc[search_index.flatten()]
    available_wiki_index_df["prompt_id"] = np.repeat(id_list, n_top)
    available_wiki_index_df = (
        available_wiki_index_df[["id", "prompt_id", "file"]]
        .drop_duplicates()
        .sort_values(["file", "id"])
        .reset_index(drop=True)
    )
    return available_wiki_index_df


from collections.abc import Iterable, Sequence

import blingfire as bf
import tqdm


def process_documents(
    documents: Sequence[str],
    document_ids: Iterable,
    split_sentences: bool = True,
    filter_len: int = 3,
    disable_progress_bar: bool = False,
) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param split_sentences: Flag to determine whether to further split sections into sentences
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values, df.document_id.values, df.offset.values, filter_len, disable_progress_bar)
    return df


def sectionize_documents(
    documents: Sequence[str], document_ids: Iterable, disable_progress_bar: bool = False
) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in tqdm.tqdm(
        zip(document_ids, documents), total=len(documents), disable=disable_progress_bar
    ):
        row = {}
        text, start, end = (document, 0, len(document))
        row["document_id"] = document_id
        row["text"] = text
        row["offset"] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(["document_id", "offset"]).reset_index(drop=True)
    else:
        return _df


def sentencize(
    documents: Sequence[str],
    document_ids: Iterable,
    offsets: Iterable[tuple[int, int]],
    filter_len: int = 3,
    disable_progress_bar: bool = False,
) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in tqdm.tqdm(
        zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar
    ):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1] - o[0] > filter_len:
                    sentence = document[o[0] : o[1]]
                    abs_offsets = (o[0] + offset[0], o[1] + offset[0])
                    row = {"document_id": document_id, "text": sentence, "offset": abs_offsets}
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)
