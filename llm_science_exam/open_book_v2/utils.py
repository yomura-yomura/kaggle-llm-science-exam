from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm
import numpy as np

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
    df_valid,
    modified_texts,
    *,
    #     chunk_size = 100_000,
    chunk_size=10_000,
    top_per_chunk=10,
    top_per_query=10,
    n_articles_to_fit=500000,
):
    corpus_df_valid = df_valid.apply(
        lambda row: f'{row["prompt"]}\n{row["prompt"]}\n{row["prompt"]}\n{row["A"]}\n{row["B"]}\n{row["C"]}\n{row["D"]}\n{row["E"]}',
        axis=1,
    ).values
    vectorizer1 = TfidfVectorizer(
        ngram_range=(1, 2), token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'", stop_words=stop_words
    )
    vectorizer1.fit(corpus_df_valid)
    vocab_df_valid = vectorizer1.get_feature_names_out()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
        stop_words=stop_words,
        vocabulary=vocab_df_valid,
    )
    vectorizer.fit(modified_texts[:n_articles_to_fit])
    corpus_vector = vectorizer.transform(corpus_df_valid)

    print(f"length of vectorizer vocab is {len(vectorizer.get_feature_names_out())}")

    all_chunk_top_indices = []
    all_chunk_top_values = []

    for idx in tqdm.tqdm(range(0, len(modified_texts), chunk_size)):
        wiki_vectors = vectorizer.transform(modified_texts[idx : idx + chunk_size])
        cosine_scores = (corpus_vector * wiki_vectors.T).toarray()

        chunk_top_indices = cosine_scores.argpartition(-top_per_chunk, axis=1)[:, -top_per_chunk:]
        chunk_top_values = cosine_scores[np.arange(cosine_scores.shape[0])[:, np.newaxis], chunk_top_indices]

        all_chunk_top_indices.append(chunk_top_indices + idx)
        all_chunk_top_values.append(chunk_top_values)

    top_indices_array = np.concatenate(all_chunk_top_indices, axis=1)
    top_values_array = np.concatenate(all_chunk_top_values, axis=1)

    merged_top_scores = np.sort(top_values_array, axis=1)[:, -top_per_query:]
    merged_top_indices = top_values_array.argsort(axis=1)[:, -top_per_query:]
    articles_indices = top_indices_array[np.arange(top_indices_array.shape[0])[:, np.newaxis], merged_top_indices]

    return articles_indices, merged_top_scores
