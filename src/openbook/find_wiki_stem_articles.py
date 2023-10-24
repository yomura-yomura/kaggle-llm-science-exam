import collections
import pathlib

import datasets
import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans

# fmt: off
target_articles = [
    'API gravity', 'Amplitude', 'Angular momentum', 'Antiferromagnetism', 'Astrochemistry', 'Baryogenesis',
    'Black hole', 'Bollard pull', 'Born reciprocity', 'Butterfly effect', 'C1 chemistry', 'Causality (physics)',
    'Cavitation', 'Clockwise', 'Coffee ring effect', 'Coherence (physics)', 'Coherent turbulent structure', 'Cold '
                                                                                                            'dark '
                                                                                                            'matter',
    "Commentary on Anatomy in Avicenna's Canon", 'Condensation cloud', 'Convection (heat transfer)', 'Copernican '
                                                                                                     'principle',
    'Critical Raw Materials Act', 'Crossover experiment (chemistry)', 'Crystallinity', 'Dark Matter',
    'Decay technique', 'Diffraction', 'Dimension', 'Droste effect', 'Dynamic scaling', "Earnshaw's theorem",
    'Ecological pyramid', 'Electric flux', 'Electrical resistivity and conductivity', 'Electrochemical gradient',
    'Electronic entropy', "Elitzur's theorem", 'Emissivity', 'Enthalpy', 'Environmental Science Center',
    'Erlangen program', 'Explicit symmetry breaking', "Fermat's principle", 'Ferromagnetism', 'Frame-dragging',
    'Free neutron decay', 'Galaxy', 'Geometric quantization', 'Gravitational wave', 'Gravity Probe B', 'Heart',
    'Heat treating', "Hesse's principle of transfer", 'History of geology', 'Hydrodynamic stability',
    'Improper rotation', 'Infectious tolerance', 'Inflation (cosmology)', 'Interstellar medium', 'James Webb Space '
                                                                                                 'Telescope',
    'Kutta-Joukowski theorem', 'Landau–Lifshitz–Gilbert equation', 'Leidenfrost effect', 'Light-year',
    'Linear time-invariant system', 'List of equations in classical mechanics', 'Lorentz covariance', 'Luminance',
    'Magnetic monopole', 'Magnetic resonance imaging', 'Magnetic susceptibility', 'Magnitude (astronomy)',
    'Main sequence', 'Mammary gland', 'Mass versus weight', 'Mass-to-charge ratio', 'Memristor', 'Minkowski space',
    'Modified Newtonian dynamics', 'Molecular cloud', 'Molecular symmetry', 'Morphology (biology)', 'Navier–Stokes '
                                                                                                    'equations',
    'Nebula', "Newton's law of universal gravitation", 'Nuclear fusion', 'Observable universe', 'Organography',
    'Paramagnetism', 'Parity (physics)', 'Phageome', 'Phase transition', 'Photophoresis', 'Planetary system',
    'Plant', 'Point groups in three dimensions', 'Probability amplitude', 'Probability density function',
    'Propagation constant', 'Pulsar', 'Pulsar-based navigation', 'QCD matter', 'Quantization (physics)',
    'Quartz crystal microbalance', 'Radiosity (radiometry)', 'Ramsauer–Townsend effect', 'Rayleigh scattering',
    'Reciprocal length', 'Redshift', 'Refractive index', 'Regular polytope', 'Relative density', 'Renormalization',
    'Ring-imaging Cherenkov detector', 'Scale (ratio)', 'Second law of thermodynamics', 'Self-organization in '
                                                                                        'cybernetics',
    'Shower-curtain effect', 'Signal-to-noise ratio', 'Spatial dispersion', 'Speed of light', 'Spin (physics)',
    'Spontaneous symmetry breaking', 'Standard Model', 'Stellar classification', 'Stochastic differential equation',
    'Superconductivity', 'Supermassive black hole', 'Supernova', 'Supersymmetric quantum mechanics', 'Supersymmetry',
    'Surface power density', 'Surgical pathology', 'Symmetry in biology', 'Symmetry of diatomic molecules',
    'The Ambidextrous Universe', 'Theorem', 'Theorem of three moments', 'Thermal equilibrium', 'Tidal force', 'Time',
    'Time standard', 'Total internal reflection', 'Triskelion', 'Ultraviolet catastrophe', 'Unified field theory',
    'Uniform tilings in hyperbolic plane', 'Vacuum', 'Virtual particle', 'Water hammer', 'Wigner quasiprobability '
                                                                                         'distribution',
    'Work function', 'Zero-point energy'
]
# fmt: on

# cohere_dataset = datasets.load_dataset("Cohere/wikipedia-22-12-en-embeddings", split="train")
# cohere_dataset.to_parquet("cohere-wikipedia-22-12-en-embeddings.parquet")
# first_paragraphs = cohere_dataset.filter(lambda example: example["paragraph_id"] == 0, num_proc=10)  # 5.7 million

# cohere_titles = cohere_dataset["title"]
# cohere_first_paragraph_titles = first_paragraphs["title"]

# docs_titles = first_paragraphs["title"]
# docs_embeddings = first_paragraphs["emb"]

npz_path = pathlib.Path("first_paragraph.npz")

df = pl.scan_parquet("cohere-wikipedia-22-12-en-embeddings.parquet")

if npz_path.exists():
    print(f"load {npz_path}")
    data = np.load(npz_path)
    docs_titles = data["title"]
    doc_embeddings_array = data["emb"]
    del data
else:
    print("first paragraph")
    first_paragraphs_df = df.filter(pl.col("paragraph_id") == 0)
    cohere_first_paragraph_df = first_paragraphs_df.select(pl.col(["title", "emb"])).collect()
    print("to numpy")
    docs_titles = cohere_first_paragraph_df["title"].to_numpy().astype(str)
    doc_embeddings_array = np.array([emb for emb in cohere_first_paragraph_df["emb"]], dtype="f4")
    del cohere_first_paragraph_df
    # del docs_embeddings
    np.savez(npz_path, title=docs_titles, emb=doc_embeddings_array)


print("clustering")

cluster_number_dict = {}
for num_clusters in [
    # 3,
    # 5,
    # 7,
    # 10,
    # 13,
    # 15,
    20,
]:
    print(f"{num_clusters}")
    cluster_path = pathlib.Path(f"cluster_id_{num_clusters}.npz")

    if cluster_path.exists():
        cluster_number_dict[num_clusters] = np.load(cluster_path)["arr_0"]
    else:
        k_means = KMeans(n_clusters=num_clusters, random_state=0, n_init=1)
        k_means.fit(doc_embeddings_array)
        cluster_number_dict[num_clusters] = k_means.predict(doc_embeddings_array)
        np.savez(cluster_path, cluster_number_dict[num_clusters])


# target_cluster_num = 20


counts_df_dict = {
    k: pd.value_counts([v[i] for i, title in enumerate(docs_titles) if title in target_articles])
    for k, v in cluster_number_dict.items()
}
target_cluster_id_dict = {k: v.index[0] for k, v in counts_df_dict.items()}

# all_counts_df_dict = {
#     k: pd.value_counts([v[i] for i, title in enumerate(docs_titles)]) for k, v in cluster_number_dict.items()
# }
#
# n_articles_df = {k: v.loc[target_cluster_id_dict[k]] for k, v in all_counts_df_dict.items()}


target_cluster_num = 20
cluster_numbers = cluster_number_dict[target_cluster_num]

# indices = np.random.choice(len(doc_embeddings_array), size=10_000, replace=False)
# samples = doc_embeddings_array[indices]
# sampled_cluster_id = cluster_numbers[indices]
#
# import umap
#
# print("UMAP")
# reducer = umap.UMAP(n_components=2)
# x = reducer.fit_transform(samples)
#
# import plotly.express as px
#
# fig = px.scatter(
#     x=x[:, 0],
#     y=x[:, 1],
#     color=sampled_cluster_id.astype(str),
#     color_discrete_sequence=px.colors.qualitative.Alphabet,
#     category_orders={"color": np.unique(sampled_cluster_id).astype(str).tolist()},
# )
# fig.show()


filtered_articles = [
    docs_titles[dx] for dx, cl_id in enumerate(cluster_numbers) if cl_id == target_cluster_id_dict[target_cluster_num]
]
np.savez("filtered_articles.npz", filtered_articles)

sel = np.isin(docs_titles, filtered_articles)
docs_embeddings_array_filtered = doc_embeddings_array[sel]
docs_titles_filtered = docs_titles[sel]

for num_clusters in [6]:
    k_means = KMeans(n_clusters=num_clusters, random_state=0, n_init=1)
    k_means.fit(docs_embeddings_array_filtered)
    cluster_numbers_filtered = k_means.predict(docs_embeddings_array_filtered)
    break

print(pd.value_counts(cluster_numbers_filtered))
# 0    97748
# 1    72908
# 5    72568
# 2    70839
# 3    69642
# 4    59631

counts_df_dict = {
    k: pd.value_counts([v[i] for i, title in enumerate(docs_titles) if title in target_articles])
    for k, v in cluster_number_dict.items()
}
sorted([(cluster_numbers_filtered[dx], i) for dx, i in enumerate(docs_titles_filtered) if i in target_articles])


print(
    pd.value_counts([cluster_numbers_filtered[dx] for dx, i in enumerate(docs_titles_filtered) if i in target_articles])
)
# 5    135
# 1      6
# 2      5
# 0      1

sel = np.isin(cluster_numbers_filtered, [5, 1, 2])
matched_df = pd.DataFrame(
    {
        "cluster_id": cluster_numbers_filtered[sel],
        "title": docs_titles_filtered[sel],
    }
)
matched_df.to_csv("final_matched.csv", index=False)

del docs_titles_filtered, docs_embeddings_array_filtered, docs_titles, doc_embeddings_array

matched_df = pd.read_csv("final_matched.csv")
matched_df = matched_df[matched_df["cluster_id"].isin([5, 1])]  # 5, 1 are related to STEM, 2 is not.

all_titles = np.load("../../all_titles.npz")["arr_0"]

matched_cohere_df = (
    df.filter(pl.col("title").is_in(matched_df["title"]))
    .filter(pl.col("title").is_in(all_titles).not_())
    .select(pl.col(["id", "title", "text", "url", "wiki_id", "views", "paragraph_id"]))
    .collect()
)
matched_cohere_df = matched_cohere_df.to_pandas()
# matched_cohere_df.to_parquet("additional_cohere.parquet")  # 250_165 of 2078_760
matched_cohere_df.to_csv("additional_cohere.csv", index=False)
