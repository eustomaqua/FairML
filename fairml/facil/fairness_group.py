# coding: utf-8
#
# TARGET:
#   Oracle bounds concerning fairness for majority voting


import numpy as np
import numba
from fairml.widget.utils_const import (
    DTY_INT, judge_transform_need, check_zero)


# =====================================
# Oracle bounds for fairness

'''
marginalised groups
|      | h(xneg,gzero)=1 | h(xneg,gzero)=0 |
| y= 1 |    TP_{gzero}   |    FN_{gzero}   |
| y= 0 |    FP_{gzero}   |    TN_{gzero}   |
privileged group
|      | h(xneg,gones)=1 | h(xneg,gones)=0 |
| y= 1 |    TP_{gones}   |    FN_{gones}   |
| y= 0 |    FP_{gones}   |    TN_{gones}   |

instance (xneg,xpos) --> (xneg,xqtb)
        xpos might be `gzero` or `gones`
'''


def marginalised_contingency(y, hx, vY, dY):
    assert len(y) == len(hx), "Shapes do not match."
    Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)
    for i in range(dY):
        for j in range(dY):
            tmp = np.logical_and(
                np.equal(y, vY[i]), np.equal(hx, vY[j]))
            Cij[i, j] = np.sum(tmp).tolist()
            # Cij[i, j] = int(np.sum(tmp))
    return Cij  # np.ndarray


@numba.jit(nopython=True)
def marginalised_confusion(Cij, loc=1):
    Cm = np.zeros((2, 2), dtype=DTY_INT)
    # loca = vY.index(pos)  # [[TP,FN],[FP,TN]]

    Cm[0, 0] = Cij[loc, loc]
    Cm[0, 1] = np.sum(Cij[loc]) - Cij[loc, loc]
    Cm[1, 0] = np.sum(Cij[:, loc]) - Cij[loc, loc]

    # Cm[1, 1] = np.sum(Cij[:loca, :, loca])
    Cm[1, 1] = (np.sum(Cij) + Cij[loc, loc] 
                - np.sum(Cij[loc]) - np.sum(Cij[:, loc]))
    return Cm  # np.ndarray


def marginalised_pd_mat(y, hx, pos=1, idx_priv=list()):
    # y : not pd.DataFrame, is pd.core.series.Series
    # hx: not pd.DataFrame, is np.ndarray
    # tmp = y.to_numpy().tolist() + hx.tolist()

    if isinstance(y, list) or isinstance(hx, list):
        y, hx = np.array(y), np.array(hx)

    # y : np.ndarray, =pd.DataFrame.to_numpy().reshape(-1)
    # hx: np.ndarray
    tmp = y.tolist() + hx.tolist()
    vY, _ = judge_transform_need(tmp)
    dY = len(vY)

    gones_y_ = y[idx_priv].tolist()
    gzero_y_ = y[np.logical_not(idx_priv)].tolist()
    gones_hx = hx[idx_priv].tolist()
    gzero_hx = hx[np.logical_not(idx_priv)].tolist()

    g1_Cij = marginalised_contingency(gones_y_, gones_hx, vY, dY)
    g0_Cij = marginalised_contingency(gzero_y_, gzero_hx, vY, dY)
    loca = vY.index(pos)

    gones_Cm = marginalised_confusion(g1_Cij, loca)
    gzero_Cm = marginalised_confusion(g0_Cij, loca)
    # gones_Cm:  for privileged group
    # gzero_Cm:  for marginalised groups
    return g1_Cij, g0_Cij, gones_Cm, gzero_Cm  # np.ndarray


# =====================================
# Group fairness measures

''' Cm
|        | hx= pos | hx= neg |
| y= pos |    TP   |    FN   |
| y= neg |    FP   |    TN   |
'''


# 1) Demographic parity
# aka. (TP+FP)/N = P[h(x)=1]

def unpriv_group_one(gones_Cm, gzero_Cm):
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[1, 0]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[1, 0]) / N0
    return float(g1), float(g0)


# 2) Equality of opportunity
# aka. TP/(TP+FN) = recall
#                 = P[h(x)=1, y=1 | y=1]

def unpriv_group_two(gones_Cm, gzero_Cm):
    t1 = gones_Cm[0, 0] + gones_Cm[0, 1]
    t0 = gzero_Cm[0, 0] + gzero_Cm[0, 1]
    g1 = gones_Cm[0, 0] / check_zero(t1)
    g0 = gzero_Cm[0, 0] / check_zero(t0)
    return float(g1), float(g0)


# 3) Predictive (quality) parity
# aka. TP/(TP+FP) = precision
#                 = P[h(x)=1, y=1 | h(x)=1]

def unpriv_group_thr(gones_Cm, gzero_Cm):
    t1 = gones_Cm[0, 0] + gones_Cm[1, 0]
    t0 = gzero_Cm[0, 0] + gzero_Cm[1, 0]
    g1 = gones_Cm[0, 0] / check_zero(t1)
    g0 = gzero_Cm[0, 0] / check_zero(t0)
    return float(g1), float(g0)


# Assume different groups have the same potential
# aka. (TP+FN)/N = P[y=1]
def unpriv_unaware(gones_Cm, gzero_Cm):
    # aka. prerequisite
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[0, 1]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[0, 1]) / N0
    return float(g1), float(g0)


# Self-defined using accuracy
# aka. (TP+TN)/N = P[h(x)=y]
def unpriv_manual(gones_Cm, gzero_Cm):
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[1, 1]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[1, 1]) / N0
    return float(g1), float(g0)


# =====================================
