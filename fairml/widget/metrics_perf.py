# coding: utf-8

import numpy as np


# =====================================
# Metrics
# =====================================


# contingency_table
# -------------------------------------
# input: lists, not np.ndarray
'''
contingency_table_binary
    |         | hi = +1 | hi = -1 |
    | hj = +1 |    a    |    c    |
    | hj = -1 |    b    |    d    |
'''


def contingency_tab_binary(ha, hb):
    if len(ha) != len(hb):
        raise AssertionError(  # number of instances/samples
            "The shapes of two individual classifiers are different.")

    tem = np.concatenate([ha, hb]).tolist()
    vY, dY = judge_transform_need(tem)
    if dY > 2:
        raise AssertionError(
            "`contingency_table` works for binary classification only.")
    elif dY == 2:
        ha = [i * 2 - 1 for i in ha]
        hb = [i * 2 - 1 for i in hb]

    hi = np.array(ha, dtype=DTY_INT)
    hj = np.array(hb, dtype=DTY_INT)
    a = np.sum((hi == 1) & (hj == 1))
    b = np.sum((hi == 1) & (hj == -1))
    c = np.sum((hi == -1) & (hj == 1))
    d = np.sum((hi == -1) & (hj == -1))
    return a, b, c, d


'''
contingency_table_binary
    |         | hj = +1 | hj = -1 |
    | hi = +1 |    a    |    b    |
    | hi = -1 |    c    |    d    |

contingency_table_multiclass
    |         | hb == y | hb != y |
    | ha == y |    a    |    b    |
    | ha != y |    c    |    d    |
'''


def contingency_tab_multiclass(ha, hb, y):
    # Do NOT use this function to calcuate!
    a = np.sum(np.logical_and(
        np.equal(ha, y), np.equal(hb, y)))
    c = np.sum(np.logical_and(
        np.not_equal(ha, y), np.equal(hb, y)))
    b = np.sum(np.logical_and(
        np.equal(ha, y), np.not_equal(hb, y)))
    d = np.sum(np.logical_and(
        np.not_equal(ha, y), np.not_equal(hb, y)))
    return int(a), int(b), int(c), int(d)


'''
contingency_table_{?}
    |              | hb!=y, hb=-1 | hb==y, hb=1 |
    | ha!=y, ha=-1 |   d          |   c         |
    | ha==y, ha= 1 |   b          |   a         |

contingency_table_multi
    |               | hb= c_0 | hb= c_1 | hb= c_{n_c-1} |
    | ha= c_0       |  C_{00} |  C_{01} |  C_{0?}       |
    | ha= c_1       |  C_{10} |  C_{11} |  C_{1?}       |
    | ha= c_{n_c-1} |  C_{?0} |  C_{?1} |  C_{??}       |
'''


def contingency_tab_multi(hi, hj, y=list()):
    tem = np.concatenate([hi, hj, y]).tolist()
    vY, dY = judge_transform_need(tem)
    if dY == 1:
        dY = 2

    ha, hb = np.array(hi), np.array(hj)
    # construct a contingency table
    Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)

    for i in range(dY):
        for j in range(dY):
            Cij[i, j] = np.sum((ha == vY[i]) & (hb == vY[j]))
    return Cij.copy()  # np.ndarray


# =====================================
# Metrics
# =====================================
