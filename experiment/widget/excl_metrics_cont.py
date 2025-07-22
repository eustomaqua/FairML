# coding: utf-8

import numpy as np
# import numba
from fairml.widget.utils_const import (
    judge_transform_need, check_zero, DTY_INT)
from experiment.wp2_oracle.fetch_data import (
    calc_confusion, calc_PR)


def calc_PRF1_multi_lists(y, hx):
    vY, _ = judge_transform_need(y + hx)
    # dY = len(vY)

    P_list = []
    R_list = []
    TP_list = []
    FP_list = []
    FN_list = []

    for pos in vY:
        TP, FP, FN, TN = calc_confusion(y, hx, pos)
        P, R, _ = calc_PR(TP, FP, FN)  # _:F1

        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        P_list.append(P)
        R_list.append(R)

    N = len(y)
    return vY, N, P_list, R_list, TP_list, FP_list, FN_list


def calc_PRF1_multi_macro(P_list, R_list, beta=1):
    # macro_P = np.sum(P_list) / N
    # macro_R = np.sum(R_list) / N
    macro_P = np.mean(P_list).tolist()
    macro_R = np.mean(R_list).tolist()
    denom = check_zero(macro_P + macro_R)
    macro_F1 = 2 * macro_P * macro_R / denom

    beta2 = beta ** 2
    denom = beta2 * macro_P + macro_R
    fbeta = (1 + beta2) * macro_P * macro_R / check_zero(denom)
    # Notice there is a difference between mine and sklearn.
    # sklearn uses np.average([f1/fbeta])

    return macro_P, macro_R, macro_F1, fbeta


def calc_PRF1_multi_micro(TP_list, FP_list, FN_list, beta=1):
    TP_avg = np.mean(TP_list).tolist()
    FP_avg = np.mean(FP_list).tolist()
    FN_avg = np.mean(FN_list).tolist()

    micro_P = TP_avg / check_zero(TP_avg + FP_avg)
    micro_R = TP_avg / check_zero(TP_avg + FN_avg)
    denom = check_zero(micro_P + micro_R)
    micro_F1 = 2 * micro_P * micro_R / denom

    beta2 = beta ** 2
    denom = (beta**2 * micro_P) + micro_R
    fbeta = (1 + beta2) * micro_P * micro_R / check_zero(denom)

    return micro_P, micro_R, micro_F1, fbeta


# =====================================
# Contingency table
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


"""
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
"""


# =====================================
# Metrics
# =====================================
