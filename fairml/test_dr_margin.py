# coding: utf-8


import numpy as np
import pdb
from pyfair.facil.ensem_voting import weighted_voting

from fairml.dr_voting_margin import (
    top1_vot_mrg,  # E_rho_L_fair_f,
    Erho_intermediate, E_rho_mrg1order, E_rho_mrg2order,
    E_rho_Ctandem, E_rho_summary, Erl_summary,
    E_rl_intermediate, Erl_mrg1order, Erl_mrg2order)

from fairml.discriminative_risk import (
    hat_L_fair, tandem_fair, ell_fair_x)


n, nc, nf = 20, 3, 5
n = 4
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
