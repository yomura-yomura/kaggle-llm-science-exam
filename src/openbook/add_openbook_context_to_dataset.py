import pathlib
import re

import numpy as np

import llm_science_exam.data
import llm_science_exam.open_book

dataset = llm_science_exam.data.dataset.get_dataset(
    "train",
    config=llm_science_exam.data.dataset.DatasetConfig(
        prompt_id=-1, additional_datasets=["cdeotte/60k-data-with-context-v2"], train_test_split=False, test_size=1
    ),
)

dataset_type = "train"
# dataset_type = "valid"


ds = dataset[dataset_type]

df = ds.to_pandas()
# df = df.iloc[:10]

# contexts = llm_science_exam.open_book._open_book.get_context2(df)

save_cache = True

from llm_science_exam.open_book._open_book import *

# Load embedding model
start = time.time()
print(f"Starting prompt embedding, t={time.time() - start :.1f}s")
model = SentenceTransformer(
    str(
        pj_struct_paths.get_data_dir_path()
        / "sentencetransformers-allminilml6v2"
        / "sentence-transformers_all-MiniLM-L6-v2"
    ),
    device="cuda:0",
).half()
model.max_seq_length = MAX_SEQ_LEN


if save_cache:
    cache_dir_path = pathlib.Path("cache") / f"{dataset_type}"
    cache_dir_path.mkdir(exist_ok=True, parents=True)

    search_index_cache_path = cache_dir_path / "search_index.npz"
    wiki_data_cache_path = cache_dir_path / "wiki_data.parquet"
else:
    search_index_cache_path = None
    wiki_data_cache_path = None


# Get embeddings of prompts
inputs = df.apply(
    lambda row: " ".join(list(filter(None, [row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]]))),
    axis=1,
).values  # better results than prompt only
prompt_embeddings = model.encode(inputs, normalize_embeddings=True, show_progress_bar=True)


if save_cache and search_index_cache_path.exists():
    search_index = np.load(search_index_cache_path)["arr_0"]
else:
    # Search closest sentences in the wikipedia index
    print(f"Starting text search, t={time.time() - start :.1f}s")
    faiss_index = faiss.read_index(
        str(pj_struct_paths.get_data_dir_path() / "wikipedia-2023-07-faiss-index" / "wikipedia_202307.index")
    )
    # faiss_index = search.change_faiss_index_on_device(faiss_index, "cuda:0")
    search_index = search.faiss_search(faiss_index, prompt_embeddings, n_top=NUM_TITLES)[1]
    # search_index = faiss_index.search(prompt_embeddings, NUM_TITLES)[1]

    # Free memory
    faiss_index.reset()
    del faiss_index
    clean_memory()

    if save_cache:
        np.savez(search_index_cache_path, search_index)

# Load wikipedia index and filter it using the results of the search
print(f"Starting loading texts, t={time.time() - start :.1f}s")


if save_cache and wiki_data_cache_path.exists():
    wiki_data = pd.read_parquet(wiki_data_cache_path)
else:
    wiki_data = pd.read_parquet(
        pj_struct_paths.get_data_dir_path() / "wikipedia-20230701" / "wiki_2023_index.parquet", columns=["id", "file"]
    )
    wiki_data = wiki_data.loc[np.unique(search_index)]

    # Retrieve text from wikipedia files and add it to wiki_data
    for filename in tqdm.tqdm(wiki_data.file.unique()):
        # Load wikipedia file
        wiki_file = pd.read_parquet(
            pj_struct_paths.get_data_dir_path() / "wikipedia-20230701" / filename, columns=["id", "text", "categories"]
        )
        wiki_file.set_index("id", inplace=True)

        # Add text column
        mask = wiki_data.file == filename
        indexes = wiki_data.loc[mask, "id"].values
        wiki_data.loc[mask, "text"] = wiki_file.loc[indexes, "text"].values
        wiki_data.loc[mask, "categories"] = wiki_file.loc[indexes, "categories"].values

        # Free memory
        del wiki_file
        clean_memory()

    if save_cache:
        wiki_data.to_parquet(wiki_data_cache_path)


import collections

# reference_regex = re.compile(r" ==References== .+? described in \d{4}")

# Split the texts into sentences and only keep NUM_SENTENCES_INCLUDE sentences
print(f"Starting sentence retrieval t={time.time() - start :.1f}s")
df["categories"] = None
df["context"] = None
for idx in tqdm.tqdm(range(len(df))):
    # Get question embedding
    q_embed = prompt_embeddings[idx]

    # Get sentence embeddings
    sentences = []
    category_counter = collections.defaultdict(lambda: 0)
    for i, (text, categories) in enumerate(
        zip(wiki_data.loc[search_index[idx], "text"], wiki_data.loc[search_index[idx], "categories"])
    ):
        if len(text) > 0:
            sentences += [
                sentence if "==References==" not in (sentence := text[start:end]) else
                # reference_regex.sub("", sentence)
                sentence.split(" ==References==", maxsplit=1)[0]
                for start, end in bf.text_to_sentences_and_offsets(text)[1]
                # if (end - start) > FILTER_LEN
            ]

        for cat in categories:
            category_counter[cat] += 1 / (i + 1)

    df.at[idx, "categories"] = dict(sorted(category_counter.items(), key=lambda p: p[1], reverse=True)[:10])

    k_embed = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    # Get closest sentences
    distances = cdist(q_embed[None, :], k_embed)[0]
    knn = np.argsort(distances)[:NUM_SENTENCES_INCLUDE]
    sorted_sentences = np.array(sentences)[knn]
    # context = "\n-".join(sorted_sentences)
    #         context = ". ".join(np.array(sentences)[knn])
    # df.loc[idx, "context"] = context

    df.at[idx, "context"] = tuple(sorted_sentences)


# Free memory
model.to("meta")
del model, wiki_data, prompt_embeddings
clean_memory()
print(f"Context added, t={time.time() - start :.1f}s")


for idx in df.index:
    for top, c in enumerate(df.loc[idx, "context"]):
        df.loc[idx, f"context_top{top + 1}"] = c
df = df.drop(columns="context")

for idx in df.index:
    for top, (cat, score) in enumerate(df.loc[idx, "categories"].items()):
        df.loc[idx, f"category_top{top + 1}"] = cat
        df.loc[idx, f"category_score_top{top + 1}"] = score
df = df.drop(columns="categories")

df.to_csv(f"{dataset_type}_with_context.csv", index=False)

# df.to_parquet("with_context.parquet")
# fsad
#
# contexts = llm_science_exam.open_book.get_context(ds, model_device="cuda", faiss_device="cuda")
#
# df = ds.to_pandas()
# df["context"] = contexts
# df = df[["prompt", "context", "A", "B", "C", "D", "E"]]
