# coding: utf-8
# Author: Yijun
#
# TARGET:
#   Oracle bounds concerning fairness for weighted voting


# import pdb
import numpy as np
from pyfair.marble.data_distance import KL_div  # import scipy
from pyfair.facil.utils_const import check_zero

# from pyfair.facil.utils_timer import fantasy_timer
# from pyfair.facil.ensem_voting import weighted_voting
# from fairml.discriminative_risk import (
#     hat_L_fair, hat_L_loss, tandem_fair, E_rho_L_loss_f)


# =====================================
# Oracle bounds for fairness
# =====================================


# top-1 voting margin

def top1_vot_mrg(yt, wgt, dY=2):
    if not dY:
        vY = np.unique(yt).tolist()
    else:
        vY = np.arange(dY).tolist()
    coef = np.array([wgt]).T

    weig = [np.sum(coef * np.equal(
        yt, i), axis=0).tolist() for i in vY]
    weig = np.array(weig)
    # loca = weig.argmax(axis=0).tolist()  # top1

    # sorting = np.argsort(-weig.T, axis=1).T
    # sorting = sorting.T.tolist()
    sorting = np.argsort(-weig, axis=0)
    # sorting = weig.argsort(axis=0).T.tolist()
    top0 = sorting[0].tolist()  # [i.index(0) for i in sorting]
    top1 = sorting[1].tolist()  # [i.index(1) for i in sorting]

    coef = weig.T.tolist()  # n = len(yt)
    q_star = [wj[i] for i, wj in zip(top0, coef)]
    q_top1 = [wj[i] for i, wj in zip(top1, coef)]
    gamma = [q - qp for q, qp in zip(q_star, q_top1)]
    # pdb.set_trace()
    return gamma  # i.e. gamma_rho


# (Pdb) weig[:, 1]
# array([0.38081275, 0.25457612, 0.36461113])
# (Pdb) weig[:, 1].argsort()
# array([1, 2, 0])


# -------------------------------------
# Oracle bounds


def Erho_intermediate(yt, yq, wgt, nc=2):
    coef = np.array([wgt]).T
    delta = np.sum(coef * np.not_equal(
        yt, yq), axis=0).tolist()
    gamma = top1_vot_mrg(yt, wgt, nc)
    return delta, gamma


# Theorem 3.1.
# First-order oracle bound


# def E_rho_mrg1order(yt, yq, wgt, nc=2):
#     # def E_rho_L_fair_f():
#     # E_rho = [hat_L_fair(
#     #     p, q)[0] for p, q in zip(yt, yq)
#     # ]
#     # tmp = np.sum(np.multiply(wgt, E_rho))
#     # alt = float(tmp)

def E_rho_mrg1order(delta, gamma):
    tmp = [i / check_zero(
        j) for i, j in zip(delta, gamma)]
    # pdb.set_trace()
    # return tmp, delta, gamma
    bnd = np.mean(tmp).tolist()  # * 2.
    return bnd, tmp


# Theorem 3.2.
# Second-order oracle bound

def E_rho_mrg2order(delta, gamma):
    tmp = [i**2 / check_zero(
        j**2) for i, j in zip(delta, gamma)]
    bnd = np.mean(tmp).tolist()  # * 4.
    return bnd, tmp


# Theorem 3.3.
# C-tandem oracle bound


# def E_rho_Ctandem(delta, gamma):
#     mrg1 = [i / check_zero(
#         j) for i, j in zip(delta, gamma)]
#     mrg2 = [i**2 / check_zero(
#         j**2) for i, j in zip(delta, gamma)]
#     bnd1 = np.mean(mrg1).tolist()
#     bnd2 = np.mean(mrg2).tolist()
#     tmp = bnd2 - bnd1 ** 2

def E_rho_Ctandem(bnd1, bnd2):
    tmp = bnd2 - bnd1 ** 2
    return tmp / check_zero(tmp + 1. / 4)


def E_rho_summary(yt, yq, wgt, nc=2):
    delta, gamma = Erho_intermediate(yt, yq, wgt, nc)
    bnd_1, _ = E_rho_mrg1order(delta, gamma)
    bnd_2, _ = E_rho_mrg2order(delta, gamma)
    bnd_3 = E_rho_Ctandem(bnd_1, bnd_2)
    # pdb.set_trace()
    return bnd_1, bnd_2, bnd_3


# -------------------------------------
# Relaxation of these oracle bounds
# 把“绝大多数点上投票很 confident”这件事写成一个可用的条件。
# 可以在验证集上直接估计 gamma_rho 的分位数，比如取
# gamma_0 为 5% 分位数，那么 eta \approx 0.05 。

def E_rl_intermediate(gamma, eta=0.05):
    r0 = np.percentile(gamma, eta).tolist()
    return r0  # gamma_0


def Erl_mrg1order(yt, yq, wgt, eta, r0):
    E_rho = np.not_equal(yt, yq).mean(axis=1)
    tmp = np.sum(np.multiply(wgt, E_rho)).tolist()  # ,ED)
    return tmp * 2. / check_zero(r0) + eta, tmp


def lemma_RHS(yt, yq, wgt):
    tmt = np.not_equal(yt, yq)  # tandem
    nf = len(wgt)               # i.e. nb_cls
    L_f_fp = [[tmt[i] & tmt[j] for j in range(
        nf)] for i in range(nf)]
    L_f_fp = np.mean(L_f_fp, axis=2)

    E_rho2 = np.sum(np.multiply(L_f_fp, wgt), axis=1)
    E_rho2 = np.sum(np.multiply(E_rho2, wgt), axis=0)
    # pdb.set_trace()
    return E_rho2.tolist()


def lemma_LHS(yt, yq, wgt):
    coef = np.array([wgt]).T
    I_f = np.not_equal(yt, yq)
    Erho = np.sum(coef * I_f, axis=0)
    ED = np.mean(Erho * Erho)
    return ED.tolist()


def Erl_mrg2order(yt, yq, wgt, eta, r0):
    tmp = lemma_RHS(yt, yq, wgt)
    return tmp * 4. / check_zero(r0**2) + eta, tmp


def Erl_summary(yt, yq, wgt, eta=.05, nc=2):
    _, gamma = Erho_intermediate(yt, yq, wgt, nc)
    r0 = E_rl_intermediate(gamma, eta)
    bnd_1, _ = Erl_mrg1order(yt, yq, wgt, eta, r0)
    bnd_2, rhs = Erl_mrg2order(yt, yq, wgt, eta, r0)
    lhs = lemma_LHS(yt, yq, wgt)
    return bnd_1, bnd_2, rhs, lhs


# -------------------------------------
# PAC bounds


def pac_indiv(nb_trn, delt):
    tmp = -np.log(delt)  # ln()
    tmp = tmp / (2. * nb_trn)
    tmp = np.sqrt(tmp)
    return tmp.tolist()


def pac_ensem(nb_trn, delt, nf):
    tmp = np.log(nf) - np.log(delt)
    tmp = tmp / (2. * nb_trn)
    tmp = np.sqrt(tmp)
    return tmp.tolist()


def pac_kl_gibbs(nb_trn, delt, coef):
    nf = len(coef)
    wgt0 = np.ones(nf) / nf
    kl = KL_div(coef, wgt0.tolist())
    # kl = scipy.stats.entropy(coef, wgt0)

    n = float(nb_trn)
    delt = check_zero(delt)
    ep_c = np.log((1 + 4. * n) / delt)
    ep_c = (kl + ep_c) / check_zero(2. * n)
    ep_c = np.sqrt(ep_c).tolist()

    ep_d = np.log(2. * np.sqrt(n) / delt)
    ep_d = (kl + ep_d) / check_zero(2. * n)
    ep_d = np.sqrt(ep_d).tolist()
    return kl, ep_c, ep_d


# -------------------------------------


# =====================================
# Oracle bounds for fairness
# =====================================
