# coding: utf-8
# Author: Yijun
#
# TARGET:
#   Oracle bounds concerning fairness for majority voting

import numpy as np
import numba


# =====================================
# Oracle bounds for fairness


def ell_fair_x(fxp, fxq):
    # both are: list, shape (nb_inst,)
    # function is symmetrical
    # value belongs to {0,1}, set
    return np.not_equal(fxp, fxq).tolist()


def ell_loss_x(fxp, y):
    # both are: list, shape (nb_inst,)
    return np.not_equal(fxp, y).tolist()


def hat_L_fair(fxp, fxq):
    # both: list, shape (nb_inst,)
    # function is symmetrical
    # value belongs to [0,1], interval
    Lfair = ell_fair_x(fxp, fxq)
    return np.mean(Lfair).tolist()  # float


def hat_L_loss(fxp, y):
    # both: list, shape (nb_inst,)
    Lloss = ell_loss_x(fxp, y)
    return np.mean(Lloss).tolist()  # float


def tandem_fair(fa_p, fa_q, fb_p, fb_q):
    # whole: list, shape (nb_inst,)
    ha = np.not_equal(fa_p, fa_q)
    hb = np.not_equal(fb_p, fb_q)
    tmp = np.logical_and(ha, hb)
    return np.mean(tmp).tolist()  # float


def tandem_loss(fa, fb, y):
    # whole: list, shape (nb_inst,)
    ha = np.not_equal(fa, y)
    hb = np.not_equal(fb, y)
    tmp = np.logical_and(ha, hb)
    return np.mean(tmp).tolist()  # float


def hat_L_objt(fxp, fxq, y, lam):
    l_fair = hat_L_fair(fxp, fxq)
    l_acc_p = hat_L_loss(fxp, y)
    return lam * l_fair + (1. - lam) * l_acc_p


def tandem_objt(fa, fa_q, fb, fb_q, y, lam):
    l_fair = tandem_fair(fa, fa_q, fb, fb_q)
    l_acc_p = hat_L_loss(fa, y)
    l_acc_q = hat_L_loss(fb, y)
    # l_acc = (1. - lam) * (l_acc_p + l_acc_q) / 2.
    # return lam * l_fair + l_acc
    l_acc = (l_acc_p + l_acc_q) / 2.
    return lam * l_fair + (1. - lam) * l_acc


def cal_L_obj_v1(yt, yq, y, wgt, lam=.5):
    L_fair = Erho_sup_L_fair(yt, yq, wgt)
    L_acc = E_rho_L_loss_f(yt, y, wgt)
    return lam * L_fair + (1. - lam) * L_acc


def cal_L_obj_v2(yt, yq, y, wgt, lam=.5):
    nb_cls = len(wgt)

    ans = []
    for i in range(nb_cls):
        tmp = [tandem_objt(yt[i], yq[i],
                           yt[j], yq[j],
                           y, lam) for j in range(nb_cls)]
        ans.append(tmp)

    res = np.sum(np.multiply(ans, wgt), axis=1)
    res = np.sum(np.multiply(res, wgt), axis=0)
    return res.tolist()


# =====================================
# Theorems


def L_fair_MV_rho(MVrho, MVpmo):
    return hat_L_fair(MVrho, MVpmo)


def L_loss_MV_rho(MVrho, y):
    return hat_L_loss(MVrho, y)


# -------------------------------------
# Theorem 3.1.
# First-order oracle bound
# -------------------------------------
# $ L(MV_\rho) \leqslant 2Exp_\rho[ L(h)] $
#
# \begin{align}
#   cal L_{loss}(MV_\rho) <= 2Exp_\rho[cal L_{loss}(f)]
#   cal L_{fair}(MV_\rho) <= 2Exp_\rho[cal L_{fair}(f)]
# \end{align}


def E_rho_L_fair_f(yt, yq, wgt):
    E_rho = [hat_L_fair(
        p, q) for p, q in zip(yt, yq)
    ]  # list, shape (nb_cls,)
    tmp = np.sum(np.multiply(wgt, E_rho))
    return tmp.tolist()  # float


def E_rho_L_loss_f(yt, y, wgt):
    E_rho = [hat_L_loss(p, y) for p in yt]
    # return np.mean(E_rho).tolist()
    tmp = np.sum(np.multiply(wgt, E_rho))
    return tmp.tolist()  # float


# -------------------------------------
# Theorem 3.3.
# Second-order oracle bound
# -------------------------------------
# $ L(MV_\rho) \leqslant 4Exp_{\rho^2}[ L(h,h')] $
#
# \begin{align}
#   L_{loss}(MV_\rho) <= 4Exp_{\rho^2}[ L_{loss}( f,f')]
#   L_{fair}(MV_\rho) <= 4Exp_{\rho^2}[ L_{fair}( f,f')]
# \end{align}


def Erho_sup_L_fair(yt, yq, wgt, nb_cls=None):
    if not nb_cls:
        nb_cls = len(wgt)  # number of weights

    L_f_fp = []
    for p in range(nb_cls):
        tmp = [tandem_fair(
            yt[p], yq[p],
            yt[i], yq[i]) for i in range(nb_cls)]
        L_f_fp.append(tmp)
    # L_f_fp: list, shape (nb_cls, nb_cls)

    E_rho2 = np.sum(np.multiply(L_f_fp, wgt), axis=1)
    E_rho2 = np.sum(np.multiply(E_rho2, wgt), axis=0)
    # E_rho2: list, shape (nb_cls,)
    # E_rho2: dtype('float64'), shape ()

    return E_rho2.tolist()  # float


def Erho_sup_L_loss(yt, y, wgt, nb_cls=None):
    if not nb_cls:
        nb_cls = len(wgt)  # length of weights

    L_f_fp = []
    for p in range(nb_cls):
        tmp = [tandem_loss(
            yt[p],
            yt[i], y) for i in range(nb_cls)]
        L_f_fp.append(tmp)
    # L_f_fp: list, shape (nb_cls, nb_cls)

    E_rho2 = np.sum(np.multiply(L_f_fp, wgt), axis=1)
    E_rho2 = np.sum(np.multiply(E_rho2, wgt), axis=0)
    return E_rho2.tolist()  # float


# -------------------------------------
# Lemma 3.2.
# -------------------------------------
# $ E_D[E_\rho[
#         \mathbb{I}(h(X) \neq y)
#     ]^2] = E_{\rho^2}[ L(h,h')] $
#
# \begin{align}
#   E_\mathcal{D}[ E_\rho[
#     \mathbb{I}( f(bmx) \neq y)
#   ]^2] = E_{\rho^2}[ L_{loss}( f,f')]
#
#   E_\mathcal{D}[ E_\rho[
#     \mathbb{I}( f(xneg,xpos) \neq f(xneg,xqtb))
#   ]^2] = E_{\rho^2}[ L_{fair}( f,f')]
# \end{align}


def ED_Erho_I_fair(yt, yq, wgt):
    wt = np.array([wgt]).T

    I_f = np.not_equal(yt, yq)
    # I_f : ndarray, shape (nb_cls, nb_inst)

    Erho = np.sum(wt * I_f, axis=0)
    # Erho: ndarray, shape (nb_inst,)

    ED = np.mean(Erho * Erho)
    # ED  : dtype('float64'), shape ()
    return ED.tolist()  # float


def ED_Erho_I_loss(yt, y, wgt):
    wt = np.array([wgt]).T
    I_f = np.not_equal(yt, y)
    Erho = np.sum(wt * I_f, axis=0)
    ED = np.mean(Erho * Erho)
    return ED.tolist()  # float


# -------------------------------------
# Theorem 3.4.
# C-tandem oracle bound
# -------------------------------------
# If $E_\rho[ L(h)] < 1/2$, then
# \begin{equation}
#   L(MV_\rho) \leq
#   \frac{
#     E_{\rho^2}[ L(h,h')] - E_\rho[ L(h)]^2
#   }{
#     E_{\rho^2}[ L(h,h')] - E_\rho[ L(h)] +1/4
#   }
# \end{equation}
#

# If $Exp_\rho[ L_{loss}] <1/2$, then
# \begin{equation}
#   L_{loss}(MV_\rho) \leqslant \frac{
#       Exp_{\rho^2}[ L_{loss}(f,f')]
#     - Exp_\rho[ L_{loss}(f)]^2
#   }{
#       Exp_{\rho^2}[ L_{loss}(f,f')]
#     - Exp_\rho[ L_{loss}(f)]
#     + 1/4
#   }
# \end{equation}
#
# If $Exp_\rho[ L_{fair}] <1/2$, then
# \begin{equation}
#   L_{fair}(MV_\rho) \leqslant \frac{
#       Exp_{\rho^2}[ L_{fair}(f,f')]
#     - Exp_\rho[ L_{fair}(f)]^2
#   }{
#       Exp_{\rho^2}[ L_{fair}(f,f')]
#     - Exp_\rho[ L_{fair}(f)]
#     + 1/4
#   }
# \end{equation}
#


# -------------------------------------
# Theorem 3.5.
# -------------------------------------


# =====================================
