# coding: utf-8


from fairml.widget.utils_const import (
    synthetic_clf, synthetic_set, judge_transform_need)
import numpy as np
import pdb
nb_spl, nb_lbl, nb_clf = 371, 3, 2  # nb_clf=7
y_bin, _, _ = synthetic_set(2, nb_spl, nb_clf)
y_non, _, _ = synthetic_set(nb_lbl, nb_spl, nb_clf)
ht_bin = synthetic_clf(y_bin, nb_clf, err=.4)
ht_non = synthetic_clf(y_non, nb_clf, err=.4)


def test_group_fair():
    # from fairml.metrics.group_fair import (
    from fairml.facil.fairness_group import (
        marginalised_contingency, marginalised_confusion,
        unpriv_group_one, unpriv_group_two, unpriv_group_thr,
        marginalised_pd_mat, unpriv_unaware, unpriv_manual)

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

    idx_priv = np.random.randint(2, size=nb_spl, dtype='bool')
    subroutine(y_bin, 1, idx_priv)  # ht_bin[0],
    subroutine(y_non, 1, idx_priv)  # ht_non[0],
    return


def test_my_DR():
    return
