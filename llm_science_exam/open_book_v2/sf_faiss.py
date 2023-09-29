import time
from collections.abc import Sequence

import datasets
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .. import pj_struct_paths
from ..typing import FilePath


def get_context(
    df: pd.DataFrame,
    wiki_dataset_paths: Sequence[FilePath],
    num_titles: int = 3,
    join: bool = True,
    max_context_per_title: int | None = None,
) -> list[str | tuple[str]]:
    model_path = pj_struct_paths.get_data_dir_path() / "bge-small-faiss"

    # Load embedding model
    start = time.time()
    print(f"Starting prompt embedding, t={time.time() - start :.1f}s")
    model = SentenceTransformer(model_path, device="cuda:0")

    # Get embeddings of prompts
    inputs = df.apply(
        lambda row: " ".join([row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]]), axis=1
    ).values  # better results than prompt only
    prompt_embeddings = model.encode(inputs, show_progress_bar=False)

    # Search closest sentences in the wikipedia index
    print(f"Loading faiss index, t={time.time() - start :.1f}s")
    faiss_index = faiss.read_index(str(model_path / "faiss.index"))
    # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index) # causes OOM, and not that long on CPU

    print(f"Starting text search, t={time.time() - start :.1f}s")
    search_index = faiss_index.search(np.float32(prompt_embeddings), num_titles)[1]

    print(f"Starting context extraction, t={time.time() - start :.1f}s")
    dataset = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in wiki_dataset_paths])
    contexts = [tuple(dataset[int(j)]["text"] for j in search_index[i]) for i in range(len(df))]
    if join:
        if max_context_per_title is None:
            return ["\n".join(f"- {context}" for context in context_tuple) for context_tuple in contexts]
        else:
            return [
                "\n".join(f"- {context[:max_context_per_title]}" for context in context_tuple)
                for context_tuple in contexts
            ]
    else:
        return contexts


class SentenceTransformer:
    """
    New SentenceTransformer class similar to the one used in @Mgöksu notebook
    but relying on the transformers library only
    """

    def __init__(self, checkpoint, max_length: int = 512, device="cuda:0"):
        self.device = device
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.max_length = max_length

    def transform(self, batch):
        tokens = self.tokenizer(
            batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=self.max_length
        )
        return tokens.to(self.device)

    def get_dataloader(self, sentences, batch_size=32):
        sentences = ["Represent this sentence for searching relevant passages: " + x for x in sentences]
        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        dataloader = self.get_dataloader(sentences, batch_size=batch_size)
        pbar = tqdm.tqdm(dataloader) if show_progress_bar else dataloader

        embeddings = []
        for batch in pbar:
            with torch.no_grad():
                e = self.model(**batch).pooler_output
                e = torch.nn.functional.normalize(e, p=2, dim=1)
                embeddings.append(e.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings
