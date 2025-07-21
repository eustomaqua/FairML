# coding: utf-8

import numpy as np
# import pdb


# -------------------------------------
# fairml/widget/data_split.py


pr_trn = .8
pr_tst = .1
nb_iter = 5

nb_lbl = 4
nb_spl = nb_lbl * 10
nb_feat = 4

y_trn = np.random.randint(nb_lbl, size=nb_spl).tolist()
y_val = np.random.randint(nb_lbl, size=nb_spl).tolist()
y_tst = np.random.randint(nb_lbl, size=nb_spl).tolist()
y = np.concatenate([y_trn, y_val, y_tst], axis=0).tolist()

X_trn = np.random.rand(nb_spl, nb_feat)  # .tolist()
X_val = np.random.rand(nb_spl, nb_feat)  # .tolist()
X_tst = np.random.rand(nb_spl, nb_feat)  # .tolist()


def test_sp():
    from fairml.widget.data_split import (
        situation_split2, situation_split3)
    for nb_iter in [2, 3, 5]:

        split_idx = situation_split2(pr_trn, nb_iter, y_trn)
        for i_trn, i_val in split_idx:
            z_trn = set(i_trn)
            z_val = set(i_val)
            assert len(i_trn) == len(z_trn) >= 1
            assert len(i_val) == len(z_val) >= 1
            assert len(i_trn) + len(i_val) == nb_spl
        split_tmp = situation_split2(pr_trn, nb_iter, y_trn)
        assert id(split_tmp) != id(split_idx)

        split_idx = situation_split3(pr_trn, pr_tst, nb_iter, y_trn)
        for i_trn, i_val, i_tst in split_idx:
            z_trn = set(i_trn)
            z_val = set(i_val)
            z_tst = set(i_tst)
            assert len(i_trn) == len(z_trn) >= 1
            assert len(i_val) == len(z_val) >= 1
            assert len(i_tst) == len(z_tst) >= 1
            assert len(i_trn) + len(i_val) + len(i_tst) == nb_spl
        split_tmp = situation_split3(pr_trn, pr_tst, nb_iter, y_trn)
        assert id(split_idx) != id(split_tmp)


def test_cv():
    from fairml.widget.data_split import (
        situation_cross_validation, situation_split1)
    y = np.concatenate([y_trn, y_val, y_tst], axis=0).tolist()
    for nb_iter in [2, 3, 5]:

        split_type = "cross_valid_v2"
        split_idx = situation_cross_validation(nb_iter, y, split_type)
        for i_trn, i_tst in split_idx:
            z_trn, z_tst = set(i_trn), set(i_tst)
            assert len(z_trn) + len(z_tst) == nb_spl * 3
            assert len(i_trn) == len(z_trn) >= 1
            assert len(i_tst) == len(z_tst) >= 1
        split_tmp = situation_cross_validation(nb_iter, y, split_type)
        assert id(split_idx) != id(split_tmp)

        split_type = "cross_valid_v3"
        split_idx = situation_cross_validation(nb_iter, y, split_type)
        for i_trn, i_val, i_tst in split_idx:
            z_trn, z_val, z_tst = set(i_trn), set(i_val), set(i_tst)
            assert len(z_trn) + len(z_val) + len(z_tst) == nb_spl * 3
            assert len(i_trn) == len(z_trn) >= 1
            assert len(i_val) == len(z_val) >= 1
            assert len(i_tst) == len(z_tst) >= 1
        split_tmp = situation_cross_validation(nb_iter, y, split_type)
        assert id(split_idx) != id(split_tmp)

    y = np.random.randint(4, size=21)
    split_idx = situation_split1(y, 0.6, 0.2)
    split_idx = situation_split1(y, 0.8, 0.1)
    split_idx = situation_split1(y, 0.6)
    split_idx = situation_split1(y, 0.8)
    split_idx = situation_split1(y, 0.9)


def test_re():
    from fairml.widget.data_split import (
        manual_repetitive, manual_cross_valid,
        scale_normalize_helper, scale_normalize_dataset)

    X = np.concatenate([X_trn, X_val, X_tst], axis=0).tolist()
    for nb_iter in [2, 3, 5]:
        split_idx = manual_repetitive(nb_iter, y)
        assert all([len(i) == nb_spl * 3 for i in split_idx])
        split_idx = manual_repetitive(nb_iter, y, True)
        assert all([len(i) == nb_spl * 3 for i in split_idx])
        split_idx = manual_cross_valid(nb_iter, y)  # , X)
        assert all([
            len(xy) + len(z) == nb_spl * 3 for xy, z in split_idx])

    Xl_trn = X_trn.tolist()
    Xl_val = X_val.tolist()
    Xl_tst = X_tst.tolist()
    for scale_type in ["standard", "min_max", "normalize"]:
        scaler = scale_normalize_helper(scale_type)
        scatmp = scale_normalize_helper(scale_type)
        assert id(scaler) != id(scatmp)
        resler, X1, X2, X3 = scale_normalize_dataset(
            scaler, Xl_trn, Xl_val, Xl_tst)  # scale_type
        restmp, X4, X5, X6 = scale_normalize_dataset(
            scatmp, Xl_trn, Xl_val, Xl_tst)  # scale_type
        assert id(resler) != id(restmp)
        assert len(set(map(id, [X1, X2, X3, X4, X5, X6]))) == 6
        assert id(scaler) == id(resler)
        assert id(scatmp) == id(restmp)
