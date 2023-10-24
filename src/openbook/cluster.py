import numpy as np
import plotly.express as px
import umap
from sklearn.cluster import KMeans

import llm_science_exam.data
import llm_science_exam.open_book

dataset = llm_science_exam.data.dataset.get_dataset(
    "train",
    dataset_config=llm_science_exam.data.dataset.DatasetConfig(
        prompt_id=-1,
        additional_datasets=["cdeotte/60k-data-with-context-v2/only-stem-generated-by-chatgpt"],
        train_test_split=False,
        test_size=1,
        with_context=True,
        context=dict(version=3, top_n_sentences=3),
    ),
)

model = llm_science_exam.open_book.model.get_model()

embeddings_dict = {}
for dateset_type in ["train", "valid"]:
    sentences = dataset[dateset_type].map(
        lambda row: {"": " ".join([row["prompt"], row[row["answer"]]])},
        num_proc=None,
    )[""]
    embeddings_dict[dateset_type] = llm_science_exam.open_book.predict.get_embeddings(model, sentences)


k_means = KMeans(n_clusters=3, random_state=0, n_init=1)
k_means.fit(embeddings_dict["train"])
df["cluster_id"] = k_means.predict(embeddings_dict["train"])


reducer = umap.UMAP(n_components=2)
x = reducer.fit_transform(embeddings_dict["valid"])


fig = px.scatter(
    x=x[:, 0],
    y=x[:, 1],
    # color=df["cluster_id"].astype(str),
    # color_discrete_sequence=px.colors.qualitative.Alphabet,
    # category_orders={"color": np.unique(df["cluster_id"]).astype(str).tolist()},
)
fig.show()


for cluster_id, gdf in df.groupby("cluster_id"):
    print(f"{cluster_id = }")
    print(gdf["prompt"].iloc[0])
    print()
    print(gdf["prompt"].iloc[1])
    print()
    print(gdf["prompt"].iloc[2])
    print()
