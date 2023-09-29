import numpy as np
import pandas as pd
import tqdm
from datasets import Dataset

from .. import pj_struct_paths
from . import predict, search
from .model import get_model


def get_context(
    ds: Dataset,
    *,
    model_device="cuda",
    faiss_device="cpu",
    n_top: int = 5,
    num_sentences_to_include: int = 20
    # ) -> list[tuple[str]]:
) -> list[str]:
    """
    num_sentences_to_include: Parameter to determine how many relevant sentences to include
    """
    model = get_model(device=model_device)

    print(f"* get simple prompt embeddings")
    prompt_embeddings = predict.get_embeddings(model, ds["prompt"], device=model_device, batch_size=16)

    possible_wiki_index_df = search.get_possible_wiki_index_df_matched_with_prompts(
        prompt_embeddings, ds["id"], faiss_device=faiss_device, n_top=n_top
    )
    wiki_fulltext_df = search.get_wiki_fulltext_df(possible_wiki_index_df)

    processed_wiki_fulltext_df = search.process_documents(wiki_fulltext_df["text"], wiki_fulltext_df["id"])

    print(f"* get wiki fulltext embeddings")
    wiki_data_embeddings = predict.get_embeddings(
        model, processed_wiki_fulltext_df["text"], device=model_device, batch_size=16
    )

    print(f"* get prompt+answers embeddings")
    prompt_with_all_answers = [" ".join([x["prompt"], x["A"], x["B"], x["C"], x["D"], x["E"]]) for x in ds]
    question_embeddings = predict.get_embeddings(model, prompt_with_all_answers, device=model_device, batch_size=16)

    contexts = []
    for i_record, prompt_id in enumerate(tqdm.tqdm(ds["id"], desc="get context")):
        prompt_indices = processed_wiki_fulltext_df[
            processed_wiki_fulltext_df["document_id"].isin(
                possible_wiki_index_df[possible_wiki_index_df["prompt_id"] == prompt_id]["id"]
            )
        ].index.to_numpy()

        assert prompt_indices.shape[0] > 0
        prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
        prompt_index.add(wiki_data_embeddings[prompt_indices])
        prompt_index = search.change_faiss_index_on_device(prompt_index, faiss_device)

        _context = []

        # Get the top matches
        ss, ii = search.faiss_search(
            prompt_index, question_embeddings, n_top=num_sentences_to_include, show_progress=False
        )
        for _s, _i in zip(ss[i_record], ii[i_record]):
            _context.append(str(processed_wiki_fulltext_df.loc[prompt_indices]["text"].iloc[_i]))

        # contexts.append(tuple(_context))
        contexts.append(" ".join(_context))

    return contexts


def get_df_with_reduced_context(df: pd.DataFrame, *, use_all_answers_in_prompt: bool, upper_limit_of_n_words: int):
    n_tokens_base = df["prompt"].map(lambda p: len(p.split()))

    if use_all_answers_in_prompt:
        n_tokens_base += pd.concat([df[col].map(lambda p: len(p.split())) for col in "ABCDE"], axis=1).sum(axis=1)
    else:
        n_tokens_base += [len(row[row["answer"]].split()) for _, row in df.iterrows()]

    sentences_list = [
        tuple(
            f"{s}" if s.endswith(".") else f"{s}." for ss in c.split(". ") for s in ss.split("\n-") if s.strip() != ""
        )
        for c in df["context"]
    ]
    # sentences_list = [tuple(f"{s}." for s in c.split(". ") if s.strip() != "") for c in df["context"]]
    n_tokens_list = [
        np.cumsum([len(s.split()) for s in sentences]) + n_tokens
        for sentences, n_tokens in zip(sentences_list, n_tokens_base)
    ]

    idx_over_th = [
        np.argmax(n_tokens >= upper_limit_of_n_words) if any(n_tokens >= upper_limit_of_n_words) else len(n_tokens)
        for n_tokens in n_tokens_list
    ]

    df["context"] = [" ".join(sentences[:idx]) for idx, sentences in zip(idx_over_th, sentences_list)]
    return df


import ctypes
import gc
import re
import time

import blingfire as bf
import faiss
import pandas as pd

# For RAG
import torch
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

# NUM_TITLES = 5
NUM_TITLES = 10
NUM_SENTENCES_INCLUDE = NUM_TITLES
FILTER_LEN = 100
MAX_SEQ_LEN = 512


def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


def get_context2(df: pd.DataFrame) -> pd.DataFrame:
    return df
