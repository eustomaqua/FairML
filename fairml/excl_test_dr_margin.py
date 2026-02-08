# coding: utf-8


import numpy as np
# import pdb
from pyfair.facil.ensem_voting import weighted_voting
from pyfair.facil.utils_const import check_equal, check_zero

from fairml.dr_voting_margin import (
    top1_vot_mrg,  # E_rho_L_fair_f,
    Erho_intermediate, E_rho_mrg1order, E_rho_mrg2order,
    E_rho_Ctandem, E_rho_summary, Erl_summary,
    E_rl_intermediate, Erl_mrg1order, Erl_mrg2order,
    pac_ensem, pac_indiv, pac_kl_gibbs)

from fairml.discriminative_risk import (
    hat_L_fair, tandem_fair, ell_fair_x)
# from pyfair.marble.data_distance import KL_div_ver1, KL_div_ver2


n, nc, nf = 20, 3, 5
# n = 4
y = np.random.randint(nc, size=n)  # y_set
y_hat = np.random.randint(nc, size=(nf, n))
wgt = np.random.rand(nf)
wgt /= np.sum(wgt)
wgt = wgt.tolist()

y = y.tolist()
# y_hat = y_hat.tolist()
# wgt = [1. / nf for _ in range(nf)]
yp = weighted_voting(y_hat, wgt)
y_qtb = np.random.randint(nc, size=(nf, n))
eta = .5  # 0.1


def test_voting_margin():
    top1_vot_mrg(y_hat, wgt, nc)
    # pdb.set_trace()

    # E_rho_L_fair_f(y_hat, y_qtb, wgt, nc)
    delta, gamma = Erho_intermediate(y_hat, y_qtb, wgt, nc)
    mrg_1 = E_rho_mrg1order(delta, gamma)
    mrg_2 = E_rho_mrg2order(delta, gamma)
    mrg_3 = E_rho_Ctandem(mrg_1[0], mrg_2[0])
    mrg_4 = E_rho_summary(y_hat, y_qtb, wgt, nc)

    r0 = E_rl_intermediate(gamma, eta)
    mrg_1 = Erl_mrg1order(y_hat, y_qtb, wgt, eta, r0)
    mrg_2 = Erl_mrg2order(y_hat, y_qtb, wgt, eta, r0)
    tmp = Erl_summary(y_hat, y_qtb, wgt, eta, nc)
    # pdb.set_trace()
    return


def test_previous_dr():
    h_hat = weighted_voting(y_hat, wgt)
    h_qtb = weighted_voting(y_qtb, wgt)
    ans_l = hat_L_fair(h_hat, h_qtb)  # [0]

    # Theorem 3.1
    tmp = [hat_L_fair(i, j) for i, j in zip(y_hat, y_qtb)]
    # tmp = [hat_L_fair(i, j)[0] for i, j in zip(y_hat, y_qtb)]
    ans_r1 = np.sum(np.multiply(wgt, tmp)).tolist()
    assert ans_l <= 2. * ans_r1
    delt, gamm = Erho_intermediate(y_hat, y_qtb, wgt, nc=nc)
    bnd_1, btmp = E_rho_mrg1order(delt, gamm)
    assert ans_l <= 2. * bnd_1
    r0 = E_rl_intermediate(gamm, eta)
    brl_1, btmp = Erl_mrg1order(y_hat, y_qtb, wgt, eta, r0)
    assert ans_l <= 2. * brl_1
    # pdb.set_trace()

    # Theorem 3.3
    tmp = [[tandem_fair(y_hat[i], y_qtb[i], y_hat[j], y_qtb[j]
                        ) for j in range(nf)] for i in range(nf)]
    ans_r3 = np.sum(np.multiply(tmp, wgt), axis=1)
    ans_r3 = np.sum(np.multiply(ans_r3, wgt), axis=0).tolist()
    assert ans_l <= 4. * ans_r3
    bnd_2, btmp = E_rho_mrg2order(delt, gamm)
    assert ans_l <= 4. * bnd_2
    brl_2, btmp = Erl_mrg2order(y_hat, y_qtb, wgt, eta, r0)
    assert ans_l <= 4. * brl_2
    # pdb.set_trace()

    # Lemma 3.2
    wgt_new = np.array([wgt]).T
    tmp = [ell_fair_x(i, j) for i, j in zip(y_hat, y_qtb)]
    ans_lp = np.sum(wgt_new * np.array(tmp), axis=0)
    ans_lp = np.mean(ans_lp ** 2).tolist()
    assert check_equal(ans_lp, ans_r3)  # ans_l == ans_r
    brl1, brl2, brl_l, brl_r = Erl_summary(y_hat, y_qtb, wgt, eta, nc)
    assert check_equal(brl1, brl_1) and check_equal(brl2, brl_2)
    assert check_equal(brl_l, brl_r) and check_equal(brl_r, ans_lp)

    # Theorem 3.4
    tmp = check_zero(ans_r3 - ans_r1 + 1. / 4)
    ans_r4 = (ans_r3 - ans_r1**2) / tmp
    # tmp = ans_r3 - ans_r1 ** 2
    # tmp = tmp / check_zero(tmp + 1. / 4)
    bnd_3 = E_rho_Ctandem(bnd_1, bnd_2)
    assert ans_l <= bnd_3  # bug! 0.75<=0.7117825668639496
    bnd1, bnd2, bnd3 = E_rho_summary(y_hat, y_qtb, wgt, nc)
    assert check_equal(bnd_1, bnd1)
    assert check_equal(bnd_2, bnd2)
    assert check_equal(bnd_3, bnd3)

    pac_1 = pac_indiv(n, 0.1)
    pac_2 = pac_ensem(n, 0.1, nf)
    assert 0 <= pac_1 <= 1 and 0 <= pac_2 <= 1
    pac_3 = pac_kl_gibbs(n, 0.1, wgt)
    assert pac_3[2] <= pac_3[1]  # tighter bound
    # pdb.set_trace()
    return
