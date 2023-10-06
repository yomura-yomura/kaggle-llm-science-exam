import numpy as np
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from ..typing import NDArray

# fmt: off
stop_words = [
    'each', 'you', 'the', 'use', 'used',
    'where', 'themselves', 'nor', "it's", 'how', "don't", 'just', 'your',
    'about', 'himself', 'with', "weren't", 'hers', "wouldn't", 'more', 'its', 'were',
    'his', 'their', 'then', 'been', 'myself', 're', 'not',
    'ours', 'will', 'needn', 'which', 'here', 'hadn', 'it', 'our', 'there', 'than',
    'most', "couldn't", 'both', 'some', 'for', 'up', 'couldn', "that'll",
    "she's", 'over', 'this', 'now', 'until', 'these', 'few', 'haven',
    'of', 'wouldn', 'into', 'too', 'to', 'very', 'shan', 'before', 'the', 'they',
    'between', "doesn't", 'are', 'was', 'out', 'we', 'me',
    'after', 'has', "isn't", 'have', 'such', 'should', 'yourselves', 'or', 'during', 'herself',
    'doing', 'in', "shouldn't", "won't", 'when', 'do', 'through', 'she',
    'having', 'him', "haven't", 'against', 'itself', 'that',
    'did', 'theirs', 'can', 'those',
    'own', 'so', 'and', 'who', "you've", 'yourself', 'her', 'he', 'only',
    'what', 'ourselves', 'again', 'had', "you'd", 'is', 'other',
    'why', 'while', 'from', 'them', 'if', 'above', 'does', 'whom',
    'yours', 'but', 'being', "wasn't", 'be'
]
# fmt: on


def retrieval(
    df,
    modified_texts,
    *,
    #     chunk_size = 100_000,
    chunk_size=10_000,
    top_per_chunk=10,
    top_per_query=10,
    n_articles_to_fit=500_000,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    corpus_df_valid = df.apply(
        lambda row: "\n".join(
            [row["prompt"], row["prompt"], row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]]
        ),
        axis=1,
    ).values
    vectorizer1 = TfidfVectorizer(
        ngram_range=(1, 2), token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'", stop_words=stop_words
    )
    vectorizer1.fit(corpus_df_valid)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
        stop_words=stop_words,
        vocabulary=vectorizer1.get_feature_names_out(),
    )
    del vectorizer1

    vectorizer.fit(modified_texts[:n_articles_to_fit])
    corpus_vector = vectorizer.transform(corpus_df_valid)

    print(f"length of vectorizer vocab is {len(vectorizer.get_feature_names_out())}")

    articles_indices = np.zeros((len(df), top_per_query), dtype="i8")
    top_scores = np.zeros((len(df), top_per_query), dtype="f4")

    for idx in tqdm.tqdm(range(0, len(modified_texts), chunk_size), desc="retrieve wiki articles chunk by chunk"):
        wiki_vectors = vectorizer.transform(modified_texts[idx : idx + chunk_size])
        cosine_scores = (corpus_vector * wiki_vectors.T).toarray()

        chunk_top_indices = cosine_scores.argpartition(-top_per_chunk, axis=1)[:, -top_per_chunk:]
        chunk_top_scores = np.take_along_axis(cosine_scores, chunk_top_indices, axis=1)

        chunk_top_indices += idx

        order = np.argsort(chunk_top_scores, axis=1)[:, ::-1]
        chunk_top_indices = np.take_along_axis(chunk_top_indices, order, axis=1)
        chunk_top_scores = np.take_along_axis(chunk_top_scores, order, axis=1)

        # update top_scores and article_indices
        merged_indices = np.concatenate([articles_indices, chunk_top_indices], axis=1)
        merged_scores = np.concatenate([top_scores, chunk_top_scores], axis=1)

        top_indices = merged_scores.argpartition(-top_per_query, axis=1)[:, -top_per_query:]
        articles_indices[:] = np.take_along_axis(merged_indices, top_indices, axis=1)
        top_scores[:] = np.take_along_axis(merged_scores, top_indices, axis=1)

    order = np.argsort(top_scores, axis=1)[:, ::-1]
    articles_indices = np.take_along_axis(articles_indices, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)

    return articles_indices, top_scores
