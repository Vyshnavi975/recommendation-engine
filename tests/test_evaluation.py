"""
Unit tests for evaluation metrics, using small hand-computed examples with
known correct answers.
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.evaluation import (
    rmse, mae, precision_at_k, recall_at_k, dcg_at_k, ndcg_at_k,
)


# --------------------------------------------------------------------------- #
# rating metrics
# --------------------------------------------------------------------------- #

def test_rmse_zero_error():
    assert rmse([4, 3, 5], [4, 3, 5]) == 0.0


def test_rmse_hand_computed():
    # errors: 1, -1, 2 -> squared: 1, 1, 4 -> mean = 2 -> sqrt(2)
    y_true = [3, 4, 3]
    y_pred = [4, 3, 5]
    expected = math.sqrt((1 ** 2 + 1 ** 2 + 2 ** 2) / 3)
    assert abs(rmse(y_true, y_pred) - expected) < 1e-9


def test_mae_hand_computed():
    y_true = [3, 4, 3]
    y_pred = [4, 3, 5]
    expected = (1 + 1 + 2) / 3
    assert abs(mae(y_true, y_pred) - expected) < 1e-9


def test_mae_zero_error():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0


# --------------------------------------------------------------------------- #
# precision@k / recall@k
# --------------------------------------------------------------------------- #

def test_precision_at_k_all_hits():
    recommended = ["a", "b", "c"]
    relevant = {"a", "b", "c", "d"}
    assert precision_at_k(recommended, relevant, k=3) == 1.0


def test_precision_at_k_no_hits():
    recommended = ["x", "y", "z"]
    relevant = {"a", "b"}
    assert precision_at_k(recommended, relevant, k=3) == 0.0


def test_precision_at_k_partial_hits():
    # top-5: a,b,x,y,c ; relevant = {a,b,c} -> 3 hits / k=5
    recommended = ["a", "b", "x", "y", "c"]
    relevant = {"a", "b", "c"}
    assert abs(precision_at_k(recommended, relevant, k=5) - 3 / 5) < 1e-9


def test_precision_at_k_truncates_to_k():
    # only first k=2 should count, even though item 3 ("c") is relevant
    recommended = ["a", "x", "c"]
    relevant = {"a", "c"}
    assert abs(precision_at_k(recommended, relevant, k=2) - 1 / 2) < 1e-9


def test_recall_at_k_all_found():
    recommended = ["a", "b", "c", "d"]
    relevant = {"a", "b"}
    assert recall_at_k(recommended, relevant, k=4) == 1.0


def test_recall_at_k_partial():
    # relevant = {a, b, c}; top-2 = [a, x] -> 1 hit / 3 relevant
    recommended = ["a", "x", "c"]
    relevant = {"a", "b", "c"}
    assert abs(recall_at_k(recommended, relevant, k=2) - 1 / 3) < 1e-9


def test_recall_at_k_no_relevant_items():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


# --------------------------------------------------------------------------- #
# ndcg@k (hand-computed against the standard formula)
# --------------------------------------------------------------------------- #

def test_dcg_at_k_hand_computed():
    # recommended order: item1 (rel=3), item2 (rel=2), item3 (rel=0)
    recommended = ["item1", "item2", "item3"]
    relevance = {"item1": 3, "item2": 2}
    expected = (
        (2 ** 3 - 1) / math.log2(2)  # rank 1
        + (2 ** 2 - 1) / math.log2(3)  # rank 2
        + (2 ** 0 - 1) / math.log2(4)  # rank 3, rel=0 (missing -> 0)
    )
    assert abs(dcg_at_k(recommended, relevance, k=3) - expected) < 1e-9


def test_ndcg_at_k_perfect_ranking_is_one():
    # if the model ranks items in exactly the ideal (relevance-descending)
    # order, NDCG must be exactly 1.0
    relevance = {"a": 5, "b": 3, "c": 1}
    recommended = ["a", "b", "c"]  # matches ideal order
    assert abs(ndcg_at_k(recommended, relevance, k=3) - 1.0) < 1e-9


def test_ndcg_at_k_worst_ranking_hand_computed():
    # worst-case ordering (ascending relevance) should score below 1 and
    # match a hand-derived value
    relevance = {"a": 5, "b": 3, "c": 1}
    recommended = ["c", "b", "a"]  # reversed / worst order

    dcg = (
        (2 ** 1 - 1) / math.log2(2)
        + (2 ** 3 - 1) / math.log2(3)
        + (2 ** 5 - 1) / math.log2(4)
    )
    idcg = (
        (2 ** 5 - 1) / math.log2(2)
        + (2 ** 3 - 1) / math.log2(3)
        + (2 ** 1 - 1) / math.log2(4)
    )
    expected = dcg / idcg
    result = ndcg_at_k(recommended, relevance, k=3)
    assert abs(result - expected) < 1e-9
    assert result < 1.0


def test_ndcg_at_k_no_relevant_items_is_zero():
    assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0


def test_ndcg_at_k_is_bounded_between_0_and_1():
    relevance = {"a": 4, "b": 1, "c": 5, "d": 2}
    recommended = ["b", "d", "a", "c"]
    result = ndcg_at_k(recommended, relevance, k=4)
    assert 0.0 <= result <= 1.0
