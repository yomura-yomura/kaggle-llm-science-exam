import pathlib

import numpy as np
import pandas as pd

import llm_science_exam.pj_struct_paths
import llm_science_exam.score

paths = [
    # "../../src/deberta/models/deberta-v3-large/01-base/test-ml1500-context4-fl14-32bits-ts/best/prob-200.csv",
    # "../../src/deberta/models/deberta-v3-large/01-base/test-max-length-1024/best/prob-200.csv",
    # "../../src/deberta/models/deberta-v3-large/01-base/test-ml1500-context4-fl16-32bits/best/prob-200.csv",  # best
    # "../../src/deberta/models/deberta-v3-large/01-base/test-ml1500-context4-fl16-32bits/best/prob-200+c300-v3.csv",
    #
    # "../../data/deberta-new-probs-200+c300.csv",
    # "../../data/deberta-new-probs-v2-200+c300.csv",
    "../../data/deberta-new-ml1500-probs.csv",
    #
    # "../../data/deberta-old-probs-200+c300.csv",
    # "../../data/deberta-old-probs-v2-200+c300.csv",
    "../../data/deberta-old-ml1500-probs.csv",
    #
    "../../data/deberta-ts-ml1500-probs.csv",
    #
    # "../../data/llama2-new-probs-200+c300.csv",  # #11
    # "../../data/llama2-new-probs-200+c300-best.csv",
    # "../../data/llama2-old-probs-200+c300.csv",
    "../../data/llama2-bge-probs.csv",
    # "../../data/deberta-probs.csv",
    # "../../data/llama2-probs.csv",  # best
    # "../../src/llama2/models/llama2/07-16bits-after-270k/13b/r-512-270k/best/prob-200+c300.csv"
    # "../../src/deberta/models/deberta-v3-large/01-base/test-v4-ml1500-context5-fl18-32bits/best/prob-200.csv",
]


def prob_to_labels(probs):
    return np.take(list("ABCDE"), np.argsort(probs, axis=-1)[:, ::-1])


def get_label_with_th(weights):
    return prob_to_labels(np.average(stacked_probs, axis=0, weights=weights))


import numba
import pandas as pd
import polars as pl
import tqdm

options = "ABCDE"
indices = list(range(5))
option_to_index = {option: index for option, index in zip(options, indices)}
index_to_option = {index: option for option, index in zip(options, indices)}


@numba.njit
def grad_func_jit(weights):
    preds = stacked_probs

    preds_clip = np.minimum(1 - 1e-15, np.maximum(preds, 1e-15))
    gradients = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        a, b, c = target_values, preds_clip[i], np.zeros((preds.shape[1], preds.shape[2]))
        a = np.eye(5)[a]
        for j in range(preds.shape[0]):
            if j != i:
                c += weights[j] * preds_clip[j]
        gradients[i] = -np.mean(
            (-a * b + (b**2) * weights[i] + b * c)
            / ((b**2) * (weights[i] ** 2) + 2 * b * c * weights[i] - b * weights[i] + (c**2) - c)
        )
    return gradients


def calc_mtr(predicted):
    y_preds = np.argsort(-predicted, 1)
    map3 = llm_science_exam.score.map_at_3(target_values.reshape(-1, 1).astype(str), y_preds.reshape(-1, 5).astype(str))
    return map3


# def calc_loss(predicted):
#     score = F.cross_entropy(torch.tensor(predicted), torch.tensor(target_values)).numpy()
#     return score


def log_loss_numpy(y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -target_values_one_hot * np.log(y_pred)
    loss = np.sum(loss, axis=-1)
    return loss.mean()


def func_to_optimise(weights):
    pred_blend = np.tensordot(weights, stacked_probs, axes=((0), (0)))
    score = log_loss_numpy(pred_blend)
    return score


def func_to_map3(weights):
    pred_blend = np.tensordot(weights, stacked_probs, axes=((0), (0)))
    score = calc_mtr(pred_blend)
    return score


if __name__ == "__main__":
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
                "wuwenmin/llm-sci-eval300-gpt4-corrected"
                #
            ],
        ),
    )["valid"]
    df = dataset.to_pandas()

    answers = df["answer"].to_numpy()

    if len(answers) > 200:
        for i, p in enumerate(paths):
            print(f"Old CV {p}")
            llm_science_exam.score.print_map_at_3(answers[:200], prob_to_labels(stacked_probs[i][:200]))
            print()

    for i, p in enumerate(paths):
        print(p)
        llm_science_exam.score.print_map_at_3(answers, prob_to_labels(stacked_probs[i]))
        print()

    target_values = df["answer"].map(option_to_index).to_numpy(int)
    target_values_one_hot = np.eye(5)[target_values]

    from scipy.optimize import minimize

    base_weight = pd.DataFrame(np.linspace(0, 1, 101, dtype="f4"))
    weight = base_weight.copy()
    for _ in tqdm.trange(len(paths) - 1):
        weight = weight.merge(base_weight, how="cross")
        weight = weight[np.sum(weight, axis=1) <= 1]
    weight = weight.to_numpy()
    weight = weight[np.sum(weight, axis=1) == 1]

    # scores = np.array(
    #     [llm_science_exam.score.map_at_3(answers, get_label_with_th(x)) for x in tqdm.tqdm(weight, desc="calc scores")]
    # )

    import multiprocess

    with multiprocess.Pool(multiprocess.cpu_count()) as pool:
        scores = np.array(
            [
                score
                for score in tqdm.tqdm(
                    pool.imap(lambda x: llm_science_exam.score.map_at_3(answers, get_label_with_th(x)), weight),
                    total=len(weight),
                )
            ]
        )

    n_top = 3
    order = np.argsort(scores)[::-1][:n_top]
    # scores[order]

    import iminuit

    records = []
    for init_guess in weight[order]:
        tol = 1e-10
        # init_guess = [1 / stacked_probs.shape[0]] * stacked_probs.shape[0]
        # init_guess = w
        bnds = [(0, 1) for _ in range(stacked_probs.shape[0])]
        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1, "jac": lambda x: [1] * len(x)}
        # print("Inital Blend Loss:", func_to_optimise(init_guess))
        print("Inital Blend MAP@3:", func_to_map3(init_guess))

        m = iminuit.Minuit(lambda x: -func_to_map3(np.array([*x, 1 - np.sum(x)])), init_guess[:-1])
        m.limits = [(0, 1)] * len(init_guess[:-1])
        m.errors = [1 / 100] * len(init_guess[:-1])
        m.simplex().migrad()

        map_at_3 = -m.fval
        w = np.array([*m.values, 1 - np.sum(m.values)])
        # res_scipy = minimize(
        #     # fun=func_to_optimise,
        #     fun=lambda weights: -func_to_map3(weights),
        #     x0=init_guess,
        #     # method="SLSQP",
        #     tol=tol,
        #     bounds=bnds,
        #     # jac=grad_func_jit,
        #     constraints=cons,
        #     options={"disp": True, "maxiter": 1000},
        # )
        # map_at_3 = func_to_map3(res_scipy.x)
        print(
            # f"Optimised Blend Loss:", res_scipy.fun,
            "Optimised Blend MAP@3:",
            map_at_3,
        )
        # print("Optimised Weights:", res_scipy.x)
        # print("-" * 70)
        records.append({"MAP@3": map_at_3, "weight": w, "initial_MAP@3": func_to_map3(init_guess)})

    score_df = pd.DataFrame(records)
    score_df = score_df.sort_values("MAP@3", ascending=False)
    print(score_df)

    print(func_to_map3(np.round(score_df.iloc[0]["weight"], decimals=3)))
    for p, w in zip(paths, score_df.iloc[0]["weight"]):
        print(f"{pathlib.Path(p).name}: {w}")
    # import plotly.express as px
    # import plotly.graph_objects as go
    #
    # go.Heatmap(
    #     x=weight[:, 1],
    #     y=weight[:, 2],
    #     z=scores,
    # )
    #
    # # px.imshow(, text_auto=".1f").show()
    #
    # fig = px.line(x=th_list, y=scores, labels={"x": "Weight", "y": "MAP@3"})
    # best_score_range = [th_list[np.argmax(scores)], th_list[len(scores) - np.argmax(scores[::-1]) - 1]]
    # fig.add_vrect(
    #     x0=best_score_range[0],
    #     x1=best_score_range[1],
    #     line_dash="dash",
    #     annotation=dict(
    #         text=f"best = {np.max(scores)} in [{best_score_range[0]:.2f}, {best_score_range[1]:.2f}]",
    #         yanchor="bottom",
    #         xanchor="left",
    #     ),
    # )
    # fig.show()
    #
    # th_at_best = np.mean(best_score_range)
    # print(f"Threshold at best = {th_at_best:.5f}")
    #
    # llm_science_exam.score.print_map_at_3(answers, get_label_with_th(th_at_best))
