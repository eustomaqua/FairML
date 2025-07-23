# coding: utf-8

import sklearn.metrics as metrics
import numpy as np
import pdb
from fairml.widget.utils_const import (
    check_equal, synthetic_dat, synthetic_clf)

# test_fairness.py
#
# from fairml.widget.utils_const import (
#     synthetic_clf, synthetic_set, judge_transform_need)
# import numpy as np
from fairml.widget.utils_const import (
    synthetic_set, judge_transform_need)


# =====================================
# discriminative risk


nb_spl, nb_feat, nb_lbl = 121, 7, 3
X_trn, y_trn = synthetic_dat(nb_lbl, nb_spl, nb_feat)
y_hat, hx_qtb = synthetic_clf(y_trn, 2, err=.2, prng=None)
z_hat, hz_qtb = synthetic_clf(y_trn, 2, err=.3, prng=None)


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
    assert check_equal(
        res, metrics.fbeta_score(z_trn, z_hat, beta=1))
    res = calc_f_beta(p, r, beta=2)
    assert check_equal(
        res, metrics.fbeta_score(z_trn, z_hat, beta=2))
    # assert res == metrics.fbeta_score(z_trn, z_hat, beta=2)
    # pdb.set_trace()
    return


# =====================================
# metric_fair.py


nb_spl, nb_lbl, nb_clf = 371, 3, 2  # nb_clf=7
y_bin, _, _ = synthetic_set(2, nb_spl, nb_clf)
y_non, _, _ = synthetic_set(nb_lbl, nb_spl, nb_clf)
ht_bin = synthetic_clf(y_bin, nb_clf, err=.4)
ht_non = synthetic_clf(y_non, nb_clf, err=.4)

idx_priv = np.random.randint(2, size=nb_spl, dtype='bool')
idx_Sjs = [idx_priv == 1, idx_priv == 0]
A_j = np.random.randint(3, size=nb_spl, dtype='int')
Sjs_bin = [idx_priv == 1, idx_priv != 1]
Sjs_non = [idx_priv == 1, idx_priv == 0, idx_priv == 2]


def test_group_fair():
    from fairml.facils.metric_fair import (
        # unpriv_grp_one, unpriv_grp_two, unpriv_grp_thr,
        unpriv_group_one, unpriv_group_two, unpriv_group_thr,
        marginalised_np_mat, unpriv_unaware, unpriv_manual,
        calc_fair_group, StatsParity_sing,  # StatsParity_mult,
        zero_division,  # alterGrps_sing, alterGroups_pl,
        extGrp1_DP_sing,  # extGrp2_EO_sing, extGrp3_PQP_sing,
        # extGrp1_DP_pl, extGrp2_EO_pl, extGrp3_PQP_pl,
        DPext_alterSP,  # extDP_SPalter, 
        marginalised_pd_mat, prev_unpriv_manual, prev_unpriv_unaware,
        prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr)

    def subroutine(y, hx, pos, A_j, Sjs_bin, Sjs_non):
        vY, _ = judge_transform_need(y)  # ,dY
        vY = vY[:: -1]
        z, ht = np.array(y), np.array(hx)  # priv=Sjs_bin[0]
        g1M, g0M = marginalised_np_mat(z, ht, pos, Sjs_non[0])
        _, _, c1, c0 = marginalised_pd_mat(z, ht, pos, Sjs_non[0])

        just_one = unpriv_group_one(g1M, g0M)
        just_two = unpriv_group_two(g1M, g0M)
        just_thr = unpriv_group_thr(g1M, g0M)
        just_zero = unpriv_unaware(g1M, g0M)
        just_four = unpriv_manual(g1M, g0M)
        assert check_equal(just_one, prev_unpriv_grp_one(c1, c0))
        assert check_equal(just_two, prev_unpriv_grp_two(c1, c0))
        assert check_equal(just_thr, prev_unpriv_grp_thr(c1, c0))
        assert check_equal(just_zero, prev_unpriv_unaware(c1, c0))
        assert check_equal(just_four, prev_unpriv_manual(c1, c0))

        assert zero_division(0., 0.) == 0.
        assert zero_division(1., 0.) == 10
        assert zero_division(1.5, 0.2) == 7.5
        ans = calc_fair_group(*just_one)
        res = StatsParity_sing(ht, Sjs_bin, pos)[0]
        tmp_1 = extGrp1_DP_sing(z, ht, Sjs_bin, pos)[0]
        assert check_equal(res, tmp_1[:-1])  # tmp[: -1])
        res = StatsParity_sing(ht, Sjs_non, pos)[0]
        tmp_2 = extGrp1_DP_sing(z, ht, Sjs_non, pos)[0]
        assert check_equal(res, tmp_2[:-1])  # tmp[: -1])

        # res = extDP_SPalter(z, ht, idx_Sjs, pos)[0]
        tmp_1 = DPext_alterSP(z, ht, Sjs_bin, pos)[0]
        tmp_2 = DPext_alterSP(z, ht, Sjs_non, pos)[0]
        assert check_equal(ans, [
            tmp_2[-1][0], tmp_1[-1][0], tmp_1[0], tmp_1[1], ])
        # pdb.set_trace()
    # subroutine(y_bin, ht_bin[0], 1, idx_priv)
    # subroutine(y_non, ht_non[0], 1, idx_priv)
    subroutine(y_bin, ht_bin[0], 1, A_j, Sjs_bin, Sjs_non)
    return


def test_group_prev():
    # from fairml.metrics.group_fair import (
    '''
    from fairml.facils.fairness_group import (
        marginalised_contingency, marginalised_confusion,
        unpriv_group_one, unpriv_group_two, unpriv_group_thr,
        marginalised_pd_mat, unpriv_unaware, unpriv_manual)
    '''

    from fairml.facils.metric_fair import (
        marginalised_contingency, marginalised_confusion,
        prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr,
        marginalised_pd_mat, prev_unpriv_manual, prev_unpriv_unaware)
    unpriv_group_one = prev_unpriv_grp_one
    unpriv_group_two = prev_unpriv_grp_two
    unpriv_group_thr = prev_unpriv_grp_thr
    unpriv_unaware = prev_unpriv_unaware
    unpriv_manual = prev_unpriv_manual

    def subroutine(y, pos, priv):
        vY, dY = judge_transform_need(y)  # + hx)
        vY = vY[:: -1]
        hx = np.random.randint(dY, size=nb_spl).tolist()
        Cij = marginalised_contingency(y, hx, vY, dY)
        Cm = marginalised_confusion(Cij, vY.index(pos))
        assert np.sum(Cm) == np.sum(Cij) == len(y)

        g1M, g0M, g1, g0 = marginalised_pd_mat(y, hx, pos, priv)
        assert np.sum(g1M) == np.sum(g1)
        assert np.sum(g0M) == np.sum(g0)
        assert np.sum(g1M) + np.sum(g0M) == len(y)

        just_one = unpriv_group_one(g1, g0)
        just_two = unpriv_group_two(g1, g0)
        just_thr = unpriv_group_thr(g1, g0)
        just_zero = unpriv_unaware(g1, g0)
        just_four = unpriv_manual(g1, g0)
        assert all([
            0 <= i <= 1 for i in just_one + just_two + just_thr])
        assert all([0 <= i <= 1 for i in just_zero + just_four])
        # pdb.set_trace()

    # idx_priv = np.random.randint(2, size=nb_spl, dtype='bool')
    subroutine(y_bin, 1, idx_priv)  # ht_bin[0],
    subroutine(y_non, 1, idx_priv)  # ht_non[0],
    return


# def test_my_DR():
#     return
