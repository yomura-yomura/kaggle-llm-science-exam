"""
Copied from https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code/notebook
"""
from typing import Literal

import numpy as np

from ..typing import NDArray


def ap_at_k(actual, predicted, k: int = 10) -> float:
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def map_at_k(actual, predicted, *, k: int = 10, reduction: Literal["average"] | None = "average") -> NDArray[np.float_]:
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    map_at_k_list = np.array([ap_at_k(a, p, k) for a, p in zip(actual, predicted)])
    if reduction == "average":
        return np.mean(map_at_k_list)
    return map_at_k_list


def map_at_3(actual, predicted, *, reduction: Literal["average"] | None = "average") -> NDArray[np.float_]:
    return map_at_k(actual, predicted, k=3, reduction=reduction)


# def precision_at_k(r, k):
#     """Precision at k"""
#     assert k <= len(r)
#     assert k != 0
#     return sum(int(x) for x in r[:k]) / k
#
#
# def map_at_3(true, predicted):
#     """Score is mean average precision at 3"""
#     U = len(predicted)
#     map_at_3 = 0.0
#     for u in range(U):
#         user_preds = predicted[u]
#         user_true = true[u]
#         user_results = [1 if item == user_true else 0 for item in user_preds]
#         for k in range(min(len(user_preds), 3)):
#             map_at_3 += precision_at_k(user_results, k + 1) * user_results[k]
#     return map_at_3 / U
