# coding: utf-8
#
# Target:
#   Existing diversity measures in ensemble learning
#


from copy import deepcopy
import numpy as np

from fairml.widget.utils_const import (
    check_zero, DTY_FLT, DTY_INT, judge_transform_need)
from fairml.widget.utils_remark import (
    PAIRWISE, NONPAIRWISE, AVAILABLE_NAME_DIVER)


# ==================================
#  General
# ==================================


# ----------------------------------
# Data Set
# ----------------------------------
#
# Instance  :  \mathcal{Y} \in \{c_1,...,c_{\ell}\} = {0,1,...,n_c-1}
# m (const) :  number of instances
# n (const) :  number of individual classifiers
# n_c (const:  number of classes / labels
#
# Data Set  :  \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^m
# Classifier:  \mathcal{H} = \{h_j\}_{j=1}^n
#


# ----------------------------------
# Pairwise Measures
# ----------------------------------
'''
contingency_table_binary
    |         | hi = +1 | hi = -1 |
    | hj = +1 |    a    |    c    |
    | hj = -1 |    b    |    d    |

contingency_table_multi
    |               | hb= c_0 | hb= c_1 | hb= c_{n_c-1} |
    | ha= c_0       |  C_{00} |  C_{01} |  C_{0?}       |
    | ha= c_1       |  C_{10} |  C_{11} |  C_{1?}       |
    | ha= c_{n_c-1} |  C_{?0} |  C_{?1} |  C_{??}       |

contingency_table_multiclass
    |         | hb == y | hb != y |
    | ha == y |    a    |    c    |
    | ha != y |    b    |    d    |
'''


def contingency_table_binary(hi, hj):
    if not (len(hi) == len(hj)):  # number of instances/samples
        raise AssertionError("These two individual classifiers have"
                             " two different shapes.")
    tem = np.concatenate([hi, hj])
    _, dY = judge_transform_need(tem)
    del tem
    if dY > 2:
        raise AssertionError("contingency_table only works for binary"
                             " classification.")  # works for only.
    elif dY == 2:
        hi = [i * 2 - 1 for i in hi]
        hj = [i * 2 - 1 for i in hj]
    #   #   #
    hi = np.array(hi, dtype=DTY_INT)
    hj = np.array(hj, dtype=DTY_INT)
    a = np.sum((hi == 1) & (hj == 1))
    b = np.sum((hi == 1) & (hj == -1))
    c = np.sum((hi == -1) & (hj == 1))
    d = np.sum((hi == -1) & (hj == -1))
    # return int(a), int(b), int(c), int(d)
    return a, b, c, d


def contingency_table_multi(hi, hj, y):
    tem = np.concatenate([hi, hj, y])
    # vY = np.unique(tem)
    # dY = len(vY)
    vY, dY = judge_transform_need(tem)
    del tem
    if dY == 1:
        dY = 2
    ha, hb = np.array(hi), np.array(hj)  # y=np.array(y)
    # construct a contingency table
    Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)
    for i in range(dY):
        for j in range(dY):
            Cij[i, j] = np.sum((ha == vY[i]) & (hb == vY[j]))
    #   #   #
    # return Cij.tolist()  # list
    return Cij.copy()  # Cij, np.ndarray


def contingency_table_multiclass(ha, hb, y):
    # construct a contingency table, Cij
    a = np.sum(np.logical_and(np.equal(ha, y), np.equal(hb, y)))
    c = np.sum(np.logical_and(np.equal(ha, y), np.not_equal(hb, y)))
    b = np.sum(np.logical_and(np.not_equal(ha, y), np.equal(hb, y)))
    d = np.sum(np.logical_and(np.not_equal(ha, y), np.not_equal(hb, y)))
    # a,b,c,d are `np.integer` (not `int`), a/b/c/d.tolist() gets `int`
    return int(a), int(b), int(c), int(d)


# ----------------------------------
# Non-Pairwise Measures
# ----------------------------------


def number_individuals_correctly(yt, y):
    # rho_x = np.sum(yt == y, axis=0)  # np.ndarray
    # return rho_x.copy()  # rho_x
    rho_x = np.sum(np.equal(yt, y), axis=0)
    return rho_x.tolist()


def number_individuals_fall_through(yt, y, nb_cls):
    # failed  # not failing down
    failing = np.sum(np.not_equal(yt, y), axis=0)  # yt!=y
    # nb_cls = len(yt)  # number of individual classifiers
    pi = []
    for i in range(nb_cls + 1):
        tem = np.mean(failing == i)  # np.sum()/m
        pi.append(float(tem))
    return pi


# ==================================
#  Pairwise Measures
#   [multi-class classification]
#   \citep{kuncheva2003diversity}
# ==================================


# ----------------------------------
# $Q$-Statistic
#   \in [-1, 1]
#   different / independent (=0) / similar predictions
#
# Q_ij = zero if hi and hj are independend;
# Q_ij is positive if hi and hj make similar predictions
# Q_ij is negative if hi and hj make different predictions
#

def Q_statistic_multiclass(ha, hb, y):
    a, b, c, d = contingency_table_multiclass(ha, hb, y)
    denominat = a * d + b * c  # 分母, denominator
    numerator = a * d - b * c  # 分子, numerator
    return numerator / check_zero(denominat)


def Q_Statistic_binary(hi, hj):
    a, b, c, d = contingency_table_binary(hi, hj)
    tem = a * d + b * c
    return (a * d - b * c) / check_zero(tem)


# self defined: more research needed
def Q_Statistic_multi(hi, hj, y):
    Cij = contingency_table_multi(hi, hj, y)
    # Cij --> np.ndarray
    # return:
    #   d  c
    #   b  a
    #
    # Cij = np.array(Cij)
    # # axd = np.prod(np.diag(Cij))  # np.diagonal
    mxn = np.shape(Cij)[0]  # mxn = Cij.shape[0]
    axd = [Cij[i][i] for i in range(mxn)]
    bxc = [Cij[i][mxn - 1 - i] for i in range(mxn)]
    axd = np.prod(axd)
    bxc = np.prod(bxc)
    return (axd - bxc) / check_zero(axd + bxc)


# ----------------------------------
# $\kappa$-Statistic
#   \in [-1, 1]?
#   =1, totally agree; =0, agree by chance;
#   <0, rare case, less than expected by chance
#
#   \kappa_p = \frac{ \Theta_1 - \Theta_2 }{ 1 - \Theta_2 }
#   \Theta_1 = \frac{a+d}{m}
#   \Theta_2 = \frac{(a+b)(a+c) + (c+d)(b+d)}{m^2}
#       \Theta_1 \in [0, 1]
#       \Theta_2 \in [0, 1]
#       \kappa_p \in [-1, 1] probably
#

def kappa_statistic_multiclass(ha, hb, y, m):
    a, b, c, d = contingency_table_multiclass(ha, hb, y)
    Theta_1 = (a + d) / float(m)
    numerator = (a + b) * (a + c) + (c + d) * (b + d)
    Theta_2 = numerator / float(m) ** 2
    denominat = 1. - Theta_2
    return (Theta_1 - Theta_2) / check_zero(denominat)


# (\Theta_1 - \Theta_2) \times m^2 = (a+d)(a+b+c+d) - (a+b)(a+c)-(c+d)(b+d)
#     method 1 = (a+b)[a+d-(a+c)] + (c+d)[a+d-(b+d)] = (a+b)(d-c)+(c+d)(a-b)
#              = ad-ac+bd-bc + ac-bc+ad-bd = 2(ad-bc)
#     method 2 = (a+c)[a+d-(a+b)] + (b+d)[a+d-(c+d)] = (a+c)(d-b)+(b+d)(a-c)
#              = ad-ab+cd-bc + ab-bc+ad-cd = 2(ad-bc)
# \Theta_1 - \Theta_2 = 2\frac{ad-bc}{m^2}
#
# (1 - \Theta_2) \times m^2 = m^2 - (a+b)(a+c)-(c+d)(b+d) = (a+b)(b+d)+(c+d)(a+c)
#     method 3 = (a+b+c+d)^2 - (a+b)(a+c) - (c+d)(b+d) = (a+b)(b+d)+(c+d)(a+c)
# 1 - \Theta_2 = \frac{ (a+b)(b+d) + (a+c)(c+d) }{m^2}
#
# \frac{\Theta_1 - \Theta_2}{1- \Theta_2} = 2\frac{ad-bc}{ (a+b)(b+d)+(a+c)(c+d) }
#     denominator = ad+ab+db+b^2 + ad+ac+dc+c^2 = 2ad+(a+d)(b+c)+b^2+c^2
#     numerator   = 2ad-2bc
# definitely, \kappa Statistic \in [-1, 1]?, even more narrow
#

def Kappa_Statistic_binary(hi, hj, m):
    a, b, c, d = contingency_table_binary(hi, hj)
    Theta_1 = float(a + d) / m
    Theta_2 = ((a + b) * (a + c) + (c + d) * (b + d)) / (float(m) ** 2)
    return (Theta_1 - Theta_2) / check_zero(1. - Theta_2)


# self defined: research needed
def Kappa_Statistic_multi(hi, hj, y, m):
    # m = len(y)  # number of instances / samples
    tem = np.concatenate([hi, hj, y])
    _, dY = judge_transform_need(tem)  # vY,
    del tem
    if dY == 1:
        dY = 2
    #   #
    Cij = np.array(contingency_table_multi(hi, hj, y))
    c_diagonal = [Cij[i, i] for i in range(dY)]
    theta1 = np.sum(c_diagonal) / float(m)
    c_row_sum = np.sum(Cij, axis=1)  # rows / float(m)
    c_col_sum = np.sum(Cij, axis=0)  # columns / float(m)
    theta2 = np.sum(c_row_sum * c_col_sum) / float(m) ** 2
    ans = (theta1 - theta2) / check_zero(1. - theta2)
    return ans, theta1, theta2


# ----------------------------------
# Disagreement
#   \in [0, 1]
#   the larger the value, the larger the diversity.
#

def disagreement_measure_multiclass(ha, hb, y, m):
    _, b, c, _ = contingency_table_multiclass(ha, hb, y)
    return (b + c) / float(m)


def Disagreement_Measure_binary(hi, hj, m):
    _, b, c, _ = contingency_table_binary(hi, hj)
    return float(b + c) / m


def Disagreement_Measure_multi(hi, hj, m):
    tem = np.sum(np.not_equal(hi, hj))  # np.sum(hi != hj)
    return float(tem) / m


# ----------------------------------
# Correlation Coefficient
#   \in [-1, 1]
#   |\rho_{ij}| \leqslant |Q_{ij}| with the same sign
#

def correlation_coefficient_multiclass(ha, hb, y):
    a, b, c, d = contingency_table_multiclass(ha, hb, y)
    numerator = a * d - b * c
    denominat = (a + b) * (a + c) * (c + d) * (b + d)
    denominat = np.sqrt(denominat)
    return numerator / check_zero(denominat)


# to see which one of Q_ij and \rho_{ij} is larger, compare
#       a * d + b * c =?= np.sqrt(np.prod([a+b, a+c, c+d, b+d]))
#       a^2d^2+2abcd+b^2c^2 =?= (a^2+ab+ac+bc)(bc+bd+cd+d^2)
#                             = (a^2+bc+ab+ac)(bc+d^2+bd+cd)
#                             = (a^2+bc)(bc+d^2) + ....
#   right= a^2bc+bcd^2+a^2d^2+b^2c^2 + (a^2+bc)(bd+cd)+(ab+ac)(bd+cd)+(ab+ac)(bc+d^2)
#   right-left= bc(a^2+d^2)-2abcd + ....
#             = bc(a-d)^2 +.... >= 0
#   0 <= left <= right
#   1/left >= 1/right
#       therefore, |Q_ij| \geqslant |\rho_ij|
#
#       0 =?= bc(a-d)^2 + ....
#   denominator of Q_ij is smaller, then abs(Q_ij) is larger
#   therefore, it should be |rho_{ij}| \leqslant |Q_{ij}|
#

def Correlation_Coefficient_binary(hi, hj):
    a, b, c, d = contingency_table_binary(hi, hj)
    denominator = (a + b) * (a + c) * (c + d) * (b + d)
    denominator = np.sqrt(denominator)
    return (a * d - b * c) / check_zero(denominator)


# self defined: more research needed
def Correlation_Coefficient_multi(hi, hj, y):
    Cij = np.array(contingency_table_multi(hi, hj, y))
    # list --> np.ndarray:  d  c
    #                       b  a
    mxn = Cij.shape[1]  # 主对角线,反对角线元素
    axd = np.prod([Cij[i, i] for i in range(mxn)])
    bxc = np.prod([Cij[i, mxn - 1 - i] for i in range(mxn)])
    C_row_sum = np.sum(Cij, axis=1)  # sum in the same row
    C_col_sum = np.sum(Cij, axis=0)  # sum in the same column
    denominator = np.multiply(C_col_sum, C_row_sum)  # element-wise
    # denominator = np_prod(denominator.tolist())
    denominator = np.prod(denominator)
    denominator = np.sqrt(denominator)
    return (axd - bxc) / check_zero(denominator)

# 这里发现了一个大 BUG！是 numpy 造成的
# numpy.prod([3886, 4440, 4964]) 结果是不对的
# 它输出为 -251284160，但实际上应为 85648061760
#
# 错因：是超出计算范围了，所以可能会得到 nan 的返回值结果


# ----------------------------------
# Double-Fault
#   \in [0, 1], should be
#

def double_fault_measure_multiclass(ha, hb, y, m):
    _, _, _, e = contingency_table_multiclass(ha, hb, y)
    # m = len(y)  # = a+b+c+d, number of instances
    # e = np.sum(np.logical_and(np.not_equal(ha,y), np.not_equal(hb,y)))
    return int(e) / float(m)


def Double_Fault_Measure_binary_multi(hi, hj, y, m):
    # np.ndarray
    ei = np.not_equal(hi, y)  # hi != y
    ej = np.not_equal(hj, y)  # hj != y
    e = np.sum(ei & ej)
    return float(e) / m


# ----------------------------------


# ==================================
#  Non-Pairwise Measures
#   [multi-class classification]
# ==================================

# m, nb_cls = len(y), len(yt)  # number of instances / individuals
#


# ----------------------------------
# Kohavi-Wolpert Variance
#   the larger the kw value, the larger the diversity
#
# (1) KWVar = \frac{ 1 - \sum_{y\in\mathcal{Y}} \mathbf{P}(y|\mathbf{x})^2 }{2}
# (2) KWVar = \frac{1}{mn^2} \sum_{k=1}^m \rho(\mathbf{x}_k)(n - \rho(\mathbf{x}_k))
#           = \frac{1}{mn^2} \sum_{k=1}^m [-(\rho(\mathbf{x}) -n/2)^2 + n^2/4]
#       because \rho(\mathbf{x}_k) \in [0, n] i.e., [0, T]
#       then -(\rho(\mathbf{x}_k) -n/2)^2 + n^2/4 in [0, n^2/4]
#       therefore KWVar \in [0, 1/4]
#

def Kohavi_Wolpert_variance_multiclass(yt, y, m, nb_cls):
    rho_x = number_individuals_correctly(yt, y)
    numerator = np.multiply(rho_x, np.subtract(nb_cls, rho_x))
    denominat = nb_cls ** 2 * float(m)
    return np.sum(numerator) / denominat


# KWVar = \frac{1}{mT^2} * \sum_{k\in[1,m]} -[\rho(x_k) - T/2]^2 + T^2/4
#       because of \rho(x_k) \in [0, T]
#       then \sum_{k}-[]^2+T^2/4 \in [0, T^2/4] \times m
#       thus KWVar \in [0, 1/4]
#


# ----------------------------------
# Inter-rater agreement
#   =1, totally agree; \leqslant 0, even less than what is expected by chance
#
# (1) numerator = \frac{1}{n} \sum_{k=1}^m \rho(x_k)(n - \rho(x_k))
# (2) denominator = m(n-1) \bar{p}(1 - \bar{p})
#     where \bar{p} = \frac{1}{mn} \sum_{i=1}^n \sum_{k=1}^m \mathbb{I}(h_i(x_k) = y_k)
# (3) \kappa = 1 - \frac{numerator}{denominator}
#
#   \rho(x_k)(n-\rho(x_k)) = -(\rho(x_k) - n/2)^2 + n^2/4 \in [0, n^2/4]
#   \bar{p} = \frac{1}{mn} \sum_{k=1}^m \rho(x_k) \in [0,1] since \rho(x_k) \in [0,n]
#   \bar{p}(1-\bar{p}) = -(\bar{p}-1/2)^2+1/4 \in [0, 1/4]
#       denominator \in m(n-1) [0, 1/4] i.e., [0, m(n-1)/4]
#       numerator \in m/n [0, n^2/4] i.e., [0, mn/4]
#       \frac{numerator}{denominator} \in [0, +\infty)
#       \kappa \in (-\infty, 1]
#

def interrater_agreement_multiclass(yt, y, m, nb_cls):
    rho_x = number_individuals_correctly(yt, y)
    p_bar = np.sum(rho_x) / float(m * nb_cls)
    numerator = np.multiply(rho_x, np.subtract(nb_cls, rho_x))
    numerator = np.sum(numerator) / float(nb_cls)
    denominat = m * (nb_cls - 1.) * p_bar * (1. - p_bar)
    return 1. - numerator / check_zero(denominat)


# Interrater agreement
#
# numerator \in \frac{1}{T} \times m \times [0, T^2/4] = \frac{m}{T} \times [0, T^2/4]
#     \bar{p}            \in \frac{1}{mT} [0, Tm] = [0, 1]
#     \bar{p}{1-\bar{p}} = -[\bar{p} - 1/2]^2+1/4 \in [0, 1/4]
# denominator \in m(T-1) [0, 1/4]
# \frac{numerator}{denominator} = \frac{1}{m(T-1)} [0, T^2/4] / [0, 1/4]
#                                          [0, +inf), [T^2, +inf)
#                              ~= \frac{1}{m(T-1)} [0, +inf) = [0, +inf)
# 1-\frac{numerator}{denominator} \in (-inf, 1]
#


# ----------------------------------
# Entropy
#
# Ent_cc= \frac{1}{m}\sum_{k=1}^m \sum_{y\in\mathcal{Y}}
#                                     -\mathbf{P}(y|x_k) \log(\mathbf{P(y|x_k)})
#   where \mathbf{P}(y|\mathbf{x}_k) =
#                        \frac{1}{n}\sum_{i=1}^n \mathbb{I}(h_i(\mathbf{x}) =y)
#
# the calculation doesn't require to know the correctness of individual classifiers.
#

def Entropy_cc_multiclass(yt, y):
    vY = np.concatenate([[y], yt]).reshape(-1)
    vY, _ = judge_transform_need(vY)
    ans = np.zeros_like(y, dtype=DTY_FLT)
    for i in vY:
        P_y_xk = np.mean(np.equal(yt, i), axis=0)  # np.sum(..)/nb_cls
        tem = list(map(check_zero, P_y_xk))
        tem = -1. * P_y_xk * np.log(tem)
        ans += tem
    ans = np.mean(ans)  # np.sum(..)/m
    return float(ans)


# ----------------------------------
# Entropy
#
# Ent_sk = \frac{1}{m}\sum_{k=1}^m \frac{ \min(\rho(x_k), n-\rho(x_k)) }{n-ceil(n/2)}
#   \in [0, 1]
#   the larger the value, the larger the diversity; =0, totally agree
#

def Entropy_sk_multiclass(yt, y, nb_cls):
    rho_x = number_individuals_correctly(yt, y)
    sub_x = np.subtract(nb_cls, rho_x).tolist()
    tmp = list(map(min, rho_x, sub_x))
    denominator = nb_cls - np.ceil(nb_cls / 2.)
    ans = np.mean(tmp) / check_zero(denominator)
    return float(ans)


# ----------------------------------
# Entropy [works for multi-class classification]
#
#     -p*np.log(p) \in [0, a], 0.35 < a < 0.4, if p\in (0, 1]
#     -p*np.log(p)-(1-p)*np.log(1-p) \in [0, a], a<=0.7?, if p\in (0,1] and q=1-p
# Entropy_cc \in \frac{1}{m} m \times \sum_{multi values of p} -p * np.log(p)
#              = \sum_{multi values of p} -p*np.log(p), and \sum_{values of p}p = 1
#            \in [0, 1]
#
#     \rho(x_k) \in [0, T]      T-\rho(x_k) \in [0, T]
#     \min(\rho(x_k), T-\rho(x_k)) \in [0, T/2]
# Entropy_sk \in \frac{1}{m* floor(T/2)} m \times [0, T/2]
#              = \frac{1}{floor(T/2)} [0, T/2] = \frac{1}{floor(T/2)} [0, floor(T/2)]
#            \in [0, 1]
#


# ----------------------------------
# Difficulty
#   the smaller the theta value, the larger the diversity
#
# Uniform distribution x\sim [a,b]
#   Expection E(x) = (a+b)/2
#   Variance D(x) = (b-a)^2/12
#
# [0, 1] --> np.var(x) \in
# Note: this is not an uniform distribution.
#       the number of x taking values may vary
#

def difficulty_multiclass(yt, y):
    X = np.mean(np.equal(yt, y), axis=0)
    ans = np.var(X)
    return float(ans)


# ----------------------------------
# Generalized diversity
#   \in [0, 1]
#   the diversity is minimized when gd=0
#   the diversity is maximized when gd=1
#
# gd = 1 - \frac{p(2)}{p(1)}
# p(2) = \sum_{i=1}^n \frac{i}{n} p_i
# p(1) = \sum_{i=1}^n \frac{i}{n} \frac{i-1}{n-1} p_i
#   p_i \in [0,1], \forall\, i \in {0,1,2,...,n}
#   \frac{i}{n} p_i \in [0,1]
#   \frac{i}{n} \frac{i-1}{n-1} = \frac{ (i-1/2)^2-1/4 }{n(n-1)} \in [0,1]
#   therefore, p(1) \in [0,n], p(2) \in [0,n], p(2)/p(1) \in [0,+\infty)
#           gd = 1- p(2)/p(1) \in (-\infty, 1]
#

def generalized_diversity_multiclass(yt, y, nb_cls):
    pi = number_individuals_fall_through(yt, y, nb_cls)
    p_1, p_2 = 0., 0.
    for i in range(1, nb_cls + 1):
        p_1 += pi[i] * i / float(nb_cls)
        # p_2 += pi[i] * (i * (i - 1.) / nb_cls / (nb_cls - 1.))
        p_2 += pi[i] * (i * (i - 1.) / nb_cls / check_zero(nb_cls - 1.))
    return 1. - p_2 / check_zero(p_1)


#   \frac{i-1}{T-1} \in [0, 1], due to i \in [1, T]
#   \frac{i}{T}   \in [1/T, 1], due to i \in [1, T]
#   0 <= \frac{i}{T} \frac{i-1}{T-1} <= \frac{i}{T} <= 1
#   p_i \in [0, 1], i = {0, 1, ..., T}
# p(2)/p(1) \in [0, 1], then gd \in [0, 1]
#

def Generalized_Diversity_multi(yt, y, m, nb_cls):
    failing = np.sum(np.not_equal(yt, y), axis=0)  # yt!=y
    pi = [-1.]
    for i in range(1, nb_cls + 1):
        tem = np.sum(failing == i) / float(m)
        pi.append(float(tem))
    p_1, p_2 = 0., 0.
    for i in range(1, nb_cls + 1):
        p_1 += pi[i] * i / nb_cls
        p_2 += pi[i] * (i * (i - 1.) / nb_cls / (nb_cls - 1.))
    return 1. - p_2 / check_zero(p_1)


# ----------------------------------
# Coincident failure
#   when all individuals are the same, cfd=0
#   when each one is different from each other, cfd=1
#   \in [0, 1] ?
#
# cfd =| 0 , if p_0 = 1
#      | \frac{1}{1-p_0} \sum_{i=1}^n \frac{n-i}{n-1} p_i
#

def coincident_failure_multiclass(yt, y, nb_cls):
    pi = number_individuals_fall_through(yt, y, nb_cls)
    if pi[0] == 1.:
        return 0.
    if pi[0] < 1.:
        ans = 0.
        for i in range(1, nb_cls + 1):
            # ans += pi[i] * (nb_cls - i) / (nb_cls - 1.)
            ans += pi[i] * (nb_cls - i) / check_zero(nb_cls - 1.)
        return ans / check_zero(1. - pi[0])
    return 0.


# ----------------------------------


# ==================================
#  General
# ==================================
#
# zhou2012ensemble     : binary (multi: self defined)
# kuncheva2003diversity: multiclass
#


# ----------------------------------
# Pairwise Measure


def pairwise_measure_item_multiclass(name_div, ha, hb, y, m):
    if name_div == "Disag":  # "Disagreement":
        ans = disagreement_measure_multiclass(ha, hb, y, m)
    elif name_div == "QStat":  # "Q_statistic":
        ans = Q_statistic_multiclass(ha, hb, y)
    elif name_div == "Corre":  # "Correlation":
        ans = correlation_coefficient_multiclass(ha, hb, y)
    elif name_div == "KStat":  # "K_statistic":
        ans = kappa_statistic_multiclass(ha, hb, y, m)
    elif name_div == "DoubF":  # "Double_fault":
        ans = double_fault_measure_multiclass(ha, hb, y, m)
    elif name_div not in PAIRWISE.keys():  # .values():
        raise ValueError("Pairwise-Measure doesn't work for"
                         " `name_div` =", name_div)
    return ans


def pairwise_measure_gather_multiclass(name_div, yt, y, m, nb_cls):
    ans = 0.
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            tem = pairwise_measure_item_multiclass(
                name_div, yt[i], yt[j], y, m)
            ans += tem
    return ans * 2. / check_zero(nb_cls * (nb_cls - 1.))


def pairwise_measure_item_binary(name_div, hi, hj, y, m):
    if name_div == "Disagreement":
        ans = Disagreement_Measure_binary(hi, hj, m)
    elif name_div == "Q_statistic":
        ans = Q_Statistic_binary(hi, hj)
    elif name_div == "Correlation":
        ans = Correlation_Coefficient_binary(hi, hj)
    elif name_div == "K_statistic":
        ans = Kappa_Statistic_binary(hi, hj, m)
    elif name_div == "Double_fault":
        ans = Double_Fault_Measure_binary_multi(hi, hj, y, m)
    else:
        raise UserWarning("LookupError! Check the `name_diver`"
                          " for pairwise_measure.")
    return ans


def pairwise_measure_item_multi(name_div, hi, hj, y, m):
    if name_div == "Disagreement":
        ans = Disagreement_Measure_multi(hi, hj, m)
    elif name_div == "Double_fault":
        ans = Double_Fault_Measure_binary_multi(hi, hj, y, m)
    #   #   #
    # three self defined: more research needed
    elif name_div == "Q_statistic":
        ans = Q_Statistic_multi(hi, hj, y)
    elif name_div == "Correlation":
        ans = Correlation_Coefficient_multi(hi, hj, y)
    elif name_div == "K_statistic":
        ans, _, _ = Kappa_Statistic_multi(hi, hj, y, m)
    else:
        raise UserWarning("LookupError! Check the `name_diver`"
                          " for pairwise_measure.")
    return ans


def pairwise_measure_whole_binary(name_div, yt, y, m, nb_cls):
    ans = 0.
    for i in range(nb_cls - 1):
        hi = yt[i]
        for j in range(i + 1, nb_cls):
            hj = yt[j]
            ans += pairwise_measure_item_binary(
                name_div, hi, hj, y, m)
    ans = ans * 2. / check_zero(nb_cls * (nb_cls - 1.))
    return float(ans)


def pairwise_measure_whole_multi(name_div, yt, y, m, nb_cls):
    ans = 0.
    for i in range(nb_cls - 1):
        hi = yt[i]
        for j in range(i + 1, nb_cls):
            hj = yt[j]
            ans += pairwise_measure_item_multi(
                name_div, hi, hj, y, m)
    ans = ans * 2. / check_zero(nb_cls * (nb_cls - 1.))
    return float(ans)


# ----------------------------------
# Non-Pairwise Measure


def nonpairwise_measure_gather_multiclass(name_div, yt, y, m, nb_cls):
    if name_div == "KWVar":  # "KWVariance":
        ans = Kohavi_Wolpert_variance_multiclass(yt, y, m, nb_cls)
    elif name_div == "Inter":  # "Interrater":
        ans = interrater_agreement_multiclass(yt, y, m, nb_cls)
    elif name_div == "EntCC":  # "EntropyCC":
        ans = Entropy_cc_multiclass(yt, y)
    elif name_div == "EntSK":  # "EntropySK":
        ans = Entropy_sk_multiclass(yt, y, nb_cls)
    elif name_div == "Diffi":  # "Difficulty":
        ans = difficulty_multiclass(yt, y)
    elif name_div == "GeneD":  # "Generalized":
        ans = generalized_diversity_multiclass(yt, y, nb_cls)
    elif name_div == "CFail":  # "CoinFailure":
        ans = coincident_failure_multiclass(yt, y, nb_cls)
    else:
        raise ValueError("Non-Pairwise-Measure doesn't work for"
                         " `name_div` =", name_div)
    return ans


def nonpairwise_measure_item_multiclass(name_div, ha, hb, y, m):
    yt = [ha, hb]  # yt = np.vstack([ha, hb])  # nb_cls = 2
    return nonpairwise_measure_gather_multiclass(name_div, yt, y, m, 2)


# ----------------------------------
# General Overall


def contrastive_diversity_gather_multiclass(name_div, y, yt):
    m = len(y)  # number of instances
    nb_cls = len(yt)  # number of individuals
    assert name_div in AVAILABLE_NAME_DIVER
    if name_div in PAIRWISE.keys():
        return pairwise_measure_gather_multiclass(
            name_div, yt, y, m, nb_cls)
    elif name_div in NONPAIRWISE.keys():
        return nonpairwise_measure_gather_multiclass(
            name_div, yt, y, m, nb_cls)
    raise ValueError(
        "LookupError! Double check the `name_div` please.")


def contrastive_diversity_item_multiclass(name_div, y, ha, hb):
    m = len(y)  # if m is None else m
    # number of individual classifiers
    assert name_div in AVAILABLE_NAME_DIVER
    if name_div in PAIRWISE.keys():
        return pairwise_measure_item_multiclass(
            name_div, ha, hb, y, m)
    elif name_div in NONPAIRWISE.keys():
        return nonpairwise_measure_item_multiclass(
            name_div, ha, hb, y, m)
    raise ValueError(
        "LookupError! Double check the `name_div` please.")


def contrastive_diversity_by_instance_multiclass(name_div, y, yt):
    # nb_cls = len(yt)  # number of individual / weak classifiers
    nb_inst = len(y)  # =m, number of instances/samples in the data set
    answer = []
    for k in range(nb_inst):
        h = [y[k]]
        ht = [[fx[k]] for fx in yt]
        ans = contrastive_diversity_gather_multiclass(name_div, h, ht)
        answer.append(ans)
    return deepcopy(answer)


# ----------------------------------


def contrastive_diversity_whole_binary(name_div, y, yt):
    m, nb_cls = len(y), len(yt)
    if name_div in PAIRWISE.keys():
        return pairwise_measure_whole_binary(
            PAIRWISE[name_div], yt, y, m, nb_cls)
    elif name_div in NONPAIRWISE.keys():
        return nonpairwise_measure_gather_multiclass(
            name_div, yt, y, m, nb_cls)
    raise ValueError("Incorrect `name_div`.")


def contrastive_diversity_whole_multi(name_div, y, yt):
    m, nb_cls = len(y), len(yt)
    if name_div in PAIRWISE.keys():
        return pairwise_measure_whole_multi(
            PAIRWISE[name_div], yt, y, m, nb_cls)
    elif name_div in NONPAIRWISE.keys():
        return nonpairwise_measure_gather_multiclass(
            name_div, yt, y, m, nb_cls)
    raise ValueError("Incorrect `name_div`.")


# ----------------------------------


def div_inst_item_cont_tab(ha, hb, vY, dY, change="mu"):
    if (change == "bi") or (dY == 2):  # dY == 2
        ha = ha * 2 - 1  # ha = [i * 2 - 1 for i in ha]
        hb = hb * 2 - 1  # hb = [i * 2 - 1 for i in hb]
    if (change in ["bi", "tr"]) or (dY in [1, 2]):
        a = np.sum(np.equal(ha, 1) & np.equal(hb, 1))
        b = np.sum(np.equal(ha, 1) & np.equal(hb, -1))
        c = np.sum(np.equal(ha, -1) & np.equal(hb, 1))
        d = np.sum(np.equal(ha, -1) & np.equal(hb, -1))
        return a, b, c, d
    elif (change == "mu") or (dY >= 3):
        Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)
        for i in range(dY):
            for j in range(dY):
                Cij[i, j] = np.sum(
                    np.equal(ha, vY[i]) & np.equal(hb, vY[j]))
        return Cij.copy()
    raise ValueError(
        "Check `change`, it should belong to {tr,bi,mu}.")


def div_inst_item_pairwise(name_div, h, ha, hb, vY, dY):
    # if change in ["tr", "bi"]:
    change = "tr" if dY == 1 else "bi"
    a, b, c, d = div_inst_item_cont_tab(ha, hb, vY, dY, change)
    # if change != "mu":
    #   #   #
    if name_div == "QStat":
        return (a * d - b * c) / check_zero(a * d + b * c)
    elif name_div == "KStat":
        m = a + b + c + d
        Theta_1 = float(a + d) / m
        Theta_2 = (
            (a + b) * (a + c) + (c + d) * (b + d)) / float(m ** 2)
        return (Theta_1 - Theta_2) / check_zero(1. - Theta_2)
    elif name_div == "Disag":
        return float(b + c) / (a + b + c + d)
    elif name_div == "Corre":
        denominator = (a + b) * (a + c) * (c + d) * (b + d)
        denominator = np.sqrt(denominator)
        return (a * d - b * c) / check_zero(denominator)
    elif name_div == "DoubF":
        e = np.sum(np.not_equal(ha, h) & np.not_equal(hb, h))
        return float(e) / (a + b + c + d)
    #   #
    raise ValueError(
        "Check `name_div`, not a pairwise measure of diversity.")


# def contrastive_diversity_institem_mubi(name_div, y, ha, hb, change="mu"):
def div_inst_item_mubi(name_div, h, ha, hb, vY, dY):
    # elif change == "mu":
    Cij = div_inst_item_cont_tab(ha, hb, vY, dY, change="mu")
    m = 1  # m = len(h)
    #   #   #
    if name_div == "QStat":
        axd = np.prod([Cij[i][i] for i in range(dY)])
        bxc = np.prod([Cij[i][dY - 1 - i] for i in range(dY)])
        return (axd - bxc) / check_zero(axd + bxc)
    elif name_div == "KStat":
        Theta_1 = np.sum([Cij[i, i] for i in range(dY)]) / float(m)
        Theta_2 = np.sum(Cij, axis=1) * np.sum(Cij, axis=0)
        Theta_2 = np.sum(Theta_2) / float(m ** 2)
        return (Theta_1 - Theta_2) / check_zero(1. - Theta_2)
    elif name_div == "Disag":
        # return np.sum(np.not_equal(ha, hb)) / float(a + b + c + d)
        return np.sum(np.not_equal(ha, hb)) / float(m)
    elif name_div == "Corre":
        axd = np.prod([Cij[i, i] for i in range(dY)])
        bxc = np.prod([Cij[i, dY - 1 - i] for i in range(dY)])
        denominator = np.multiply(
            np.sum(Cij, axis=1), np.sum(Cij, axis=0))
        denominator = np.sqrt(np.prod(denominator))
        # denominator = np.sqrt(np_prod(denominator.tolist()))
        return (axd - bxc) / check_zero(denominator)
    elif name_div == "DoubF":
        e = np.sum(np.not_equal(ha, h) & np.not_equal(hb, h))
        return float(e) / m
    #   #
    # if name_div not in PAIRWISE:  # i.e., PAIRWISE.keys()
    raise ValueError(
        "Check `name_div`, it should be a pairwise measure.")


def contrastive_diversity_instance_mubi(name_div, y, yt):  # , change="mu"
    vY = np.concatenate(
        [[y], yt], axis=0).reshape(-1).tolist()
    vY, dY = judge_transform_need(vY)
    change = "mu" if dY >= 3 else ("bi" if dY == 2 else "tr")  # dY==1
    nb_inst, nb_cls, answer = len(y), len(yt), []
    for k in range(nb_inst):
        h = y[k]  # h = [y[k]]
        ht = [fx[k] for fx in yt]  # ht = [[fx[k]] for fx in yt]
        # if change == "mu":
        #     ans = contrastive_diversity_whole_multi(name_div, h, ht)
        # elif change in ["bi", "tr"]:
        #     ans = contrastive_diversity_whole_binary(name_div, h, ht)
        # else:
        #     raise ValueError("Check the `change` parameter please.")
        if name_div in PAIRWISE:
            res = 0.
            if change in ["tr", "bi"]:
                for ia in range(nb_cls - 1):
                    for ib in range(ia + 1, nb_cls):
                        res += div_inst_item_pairwise(
                            name_div, h, ht[ia], ht[ib], vY, dY)
            elif change in ["mu"]:
                for ia in range(nb_cls - 1):
                    for ib in range(ia + 1, nb_cls):
                        res += div_inst_item_mubi(
                            name_div, h, ht[ia], ht[ib], vY, dY)
            else:
                raise ValueError(
                    "Check `change`, it should belong to {tr,bi,mu}.")
            ans = res * 2. / check_zero(nb_cls * (nb_cls - 1.))
        elif name_div in NONPAIRWISE:
            # i.e.,  contrastive_diversity_whole_multi(name_div, h, ht)
            ht = [[fx] for fx in ht]
            ans = contrastive_diversity_gather_multiclass(
                name_div, [h], ht)
            # NOTICE: 'GeneD', 'CFail/CoinF' might have bug:
            #       ## ZeroDivisionError: float division by zero
        else:
            raise ValueError("Check `name_div`, pairwise/nonpairwise?")
        answer.append(ans)
    return deepcopy(answer)
