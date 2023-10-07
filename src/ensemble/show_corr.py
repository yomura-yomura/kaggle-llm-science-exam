import numpy as np
import pandas as pd

import llm_science_exam.pj_struct_paths
import llm_science_exam.score

paths = [
    # "../../src/deberta/models/deberta-v3-large/01-base/test-ml1500-context4-fl14-32bits-ts/best/prob-200.csv",
    # "../../src/deberta/models/deberta-v3-large/01-base/test-max-length-1024/best/prob-200.csv",
    # "../../src/deberta/models/deberta-v3-large/01-base/test-ml1500-context4-fl16-32bits/best/prob-200.csv",  # best
    # "../../src/deberta/models/deberta-v3-large/01-base/test-ml1500-context4-fl16-32bits/best/prob-200+c300-v3.csv",
    "../../data/deberta-probs.csv",
    "../../data/llama2-probs.csv",  # best
    # "../../src/llama2/models/llama2/07-16bits-after-270k/13b/r-512-270k/best/prob-200+c300.csv"
    "../../src/deberta/models/deberta-v3-large/01-base/test-v4-ml1500-context5-fl18-32bits/best/prob-200.csv",
]


stacked_probs = np.stack(
    [pd.read_csv(p).to_numpy() for p in paths],
    axis=0,
)

corr = np.corrcoef(stacked_probs.reshape(stacked_probs.shape[0], -1))
print(corr)


import llm_science_exam.data

dataset = llm_science_exam.data.dataset.get_dataset(
    "valid",
    llm_science_exam.data.dataset.DatasetConfig(
        with_context=False,
        test_size=1,
        train_test_split=False,
        valid_additional_datasets=[
            # "wuwenmin/llm-sci-eval300-gpt4-corrected"
            #
        ],
    ),
)["valid"]
df = dataset.to_pandas()

answers = df["answer"].to_numpy()


def prob_to_labels(probs):
    return np.take(list("ABCDE"), np.argsort(probs, axis=1)[:, ::-1])


for i, p in enumerate(paths):
    print(p)
    llm_science_exam.score.print_map_at_3(answers, prob_to_labels(stacked_probs[i]))
    print()


def get_label_with_th(weights):
    return prob_to_labels(np.average(stacked_probs, axis=0, weights=weights))


import pandas as pd
import tqdm

base_weight = pd.DataFrame(np.linspace(0, 1, 300, dtype="f4"))
weight = base_weight.copy()
for _ in tqdm.trange(len(paths) - 1):
    weight = weight.merge(base_weight, how="cross")
weight = weight.to_numpy()
weight = weight[np.sum(weight, axis=1) == 1]


scores = np.array(
    [llm_science_exam.score.map_at_3(answers, get_label_with_th(x)) for x in tqdm.tqdm(weight, desc="calc scores")]
)

import plotly.express as px

px.density_heatmap(x=weight[:, 1], y=weight[:, 2], z=scores).show()

fig = px.line(x=th_list, y=scores, labels={"x": "Weight", "y": "MAP@3"})
best_score_range = [th_list[np.argmax(scores)], th_list[len(scores) - np.argmax(scores[::-1]) - 1]]
fig.add_vrect(
    x0=best_score_range[0],
    x1=best_score_range[1],
    line_dash="dash",
    annotation=dict(
        text=f"best = {np.max(scores)} in [{best_score_range[0]:.2f}, {best_score_range[1]:.2f}]",
        yanchor="bottom",
        xanchor="left",
    ),
)
fig.show()


th_at_best = np.mean(best_score_range)
print(f"Threshold at best = {th_at_best:.5f}")

llm_science_exam.score.print_map_at_3(answers, get_label_with_th(th_at_best))
