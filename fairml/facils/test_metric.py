# coding: utf-8

import sklearn.metrics as metrics
import numpy as np
# import pdb
from fairml.widget.utils_const import (
    check_equal, synthetic_dat, synthetic_clf)

nb_spl, nb_feat, nb_lbl = 121, 7, 3
X_trn, y_trn = synthetic_dat(nb_lbl, nb_spl, nb_feat)
y_hat, hx_qtb = synthetic_clf(y_trn, 2, err=.2, prng=None)
z_hat, hz_qtb = synthetic_clf(y_trn, 2, err=.3, prng=None)


# =====================================
# discriminative risk


def test_my_DR():
    from fairml.discriminative_risk import (
        hat_L_fair, hat_L_loss, tandem_fair, tandem_loss,
        hat_L_objt, tandem_objt)

    ans = hat_L_fair(y_hat, hx_qtb)
    res = hat_L_loss(y_hat, y_trn)
    err = float(1. - np.mean(np.equal(y_trn, y_hat)))
    assert check_equal(err, res)

    ans = tandem_fair(y_hat, hx_qtb, z_hat, hz_qtb)
    res = tandem_loss(y_hat, z_hat, y_trn)
    lam = .5
    ans = hat_L_objt(y_hat, hx_qtb, y_trn, lam)
    assert 0. <= ans <= 1.
    ans = tandem_objt(y_hat, hx_qtb, z_hat, hz_qtb, y_trn, lam)
    assert 0. <= ans <= 1.

    # pdb.set_trace()
    return


# =====================================
# metric_perf.py


# binary classification
_, z_trn = synthetic_dat(2, nb_spl, nb_feat)
z_hat, _ = synthetic_clf(z_trn, 2, err=.2, prng=None)

y_trn, y_hat = np.array(y_trn), np.array(y_hat)
z_trn, z_hat = np.array(z_trn), np.array(z_hat)


def test_contingency():
    # from fairml.facils.metric_cont import (
    from fairml.widget.metric_cont import (
        contingency_tab_bi, contg_tab_mu_type3)

    ans = metrics.cluster.contingency_matrix(y_trn, y_hat)
    res = contg_tab_mu_type3(y_trn, y_hat, list(range(nb_lbl)))
    assert np.all(np.equal(ans, res))
    ans = metrics.cluster.contingency_matrix(z_trn, z_hat)
    res = contg_tab_mu_type3(z_trn, z_hat, [0, 1])
    assert np.all(np.equal(ans, res))

    res = contingency_tab_bi(z_trn, z_hat, pos=1)
    assert res[0] == ans[1, 1]
    assert res[-1] == ans[0, 0]
    assert ans[0, 1] == res[1]  # fp
    assert ans[1, 0] == res[2]  # fn

    # pdb.set_trace()
    return


def test_performance():
    from fairml.widget.metric_cont import contingency_tab_bi
    from fairml.facils.metric_perf import (
        calc_accuracy, calc_error_rate, calc_precision, calc_recall,
        calc_f1_score, calc_f_beta)

    mid = contingency_tab_bi(z_trn, z_hat, pos=1)
    res = calc_accuracy(*mid)
    assert res == metrics.accuracy_score(z_trn, z_hat)
    assert check_equal(res, 1. - calc_error_rate(*mid))

    p = calc_precision(*mid)
    assert p == metrics.precision_score(z_trn, z_hat)
    r = calc_recall(*mid)
    assert r == metrics.recall_score(z_trn, z_hat)
    res = calc_f1_score(*mid)
    assert check_equal(res, metrics.f1_score(z_trn, z_hat))

    res = calc_f_beta(p, r, beta=1)
    assert res == metrics.fbeta_score(z_trn, z_hat, beta=1)
    res = calc_f_beta(p, r, beta=2)
    assert res == metrics.fbeta_score(z_trn, z_hat, beta=2)
    # pdb.set_trace()
    return


# =====================================
