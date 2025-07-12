# coding: utf-8
#
# Target:
#   Split data for experiments
#   Split one dataset into "training / validation / test" dataset
#
#   Oracle bounds regarding fairness for majority voting
#


from copy import deepcopy
import gc
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection


# sklearn
# https://scikit-learn.org/stable/modules/ensemble.html
# https://scikit-learn.org/0.24/modules/cross_validation.html
# https://scikit-learn.org/0.24/model_selection.html

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold  # , RepeatedKFold
from sklearn.model_selection import StratifiedKFold

gc.enable()


# ==================================
# different ways to split data
# ==================================


# ----------------------------------
# Split situation 2
# ----------------------------------

def situation_split2(pr_trn, nb_iter, y_trn):
    vY = np.unique(y_trn).tolist()
    dY = len(vY)
    iY = [np.where(np.array(y_trn) == j)[0] for j in vY]  # indexes,
    # iY = [np.where(np.equal(y_trn, j))[0] for j in vY]  # np.ndarray
    lY = [len(j) for j in iY]  # length
    sY = [int(
        np.max([np.round(j * pr_trn), 1])) for j in lY]  # split_loca
    tem_idx = [np.arange(j) for j in lY]
    # del pr_trn, y_trn

    nb_trn = len(y_trn)
    nb_val = int(np.round(nb_trn * (1. * pr_trn)))
    nb_val = min(max(nb_val, 1), nb_trn - 1)

    split_idx = []
    for k in range(nb_iter):
        for i in tem_idx:
            np.random.shuffle(i)
        i_trn = [iY[i][tem_idx[i][:sY[i]]] for i in range(dY)]
        i_val = [iY[i][tem_idx[i][sY[i]:]] for i in range(dY)]
        i_trn = np.concatenate(i_trn, axis=0).tolist()
        i_val = np.concatenate(i_val, axis=0).tolist()
        if len(i_val) == 0:
            i_val = list(set(np.random.randint(nb_trn, size=nb_val)))
        tem = (deepcopy(i_trn), deepcopy(i_val))  # .copy()
        split_idx.append(deepcopy(tem))

        del i_trn, i_val, tem, i
    del k, tem_idx, sY, lY, dY, vY  # nb_iter
    gc.collect()
    return deepcopy(split_idx)


# ----------------------------------
# Split situation 3
# ----------------------------------

def situation_split3(pr_trn, pr_tst, nb_iter, y):
    pr_val = 1. - pr_trn - pr_tst

    vY = np.unique(y).tolist()
    dY = len(vY)
    iY = [np.where(np.array(y) == j)[0] for j in vY]  # indexes,
    # iY = [np.where(np.equal(y, j))[0] for j in vY]  # np.ndarray
    lY = [len(j) for j in iY]  # length
    sY = [[int(np.max([np.round(j * i), 1])) for i in (
        # pr_trn, pr_trn + pr_val)] for j in lY]
        pr_trn, pr_trn + pr_tst)] for j in lY]
    tem_idx = [np.arange(j) for j in lY]

    nb_y = len(y)
    nb_trn = int(np.round(nb_y * pr_trn))
    nb_tst = int(np.round(nb_y * pr_tst))
    # nb_val = int(np.round(nb_y * pr_val))
    nb_trn = min(max(nb_trn, 1), nb_y)
    nb_tst = min(max(nb_tst, 1), nb_y - 1)
    nb_val = nb_y - nb_trn - nb_tst
    nb_val = min(max(nb_val, 1), nb_y - 1)
    del pr_val  # pr_trn, pr_tst, y

    split_idx = []
    for k in range(nb_iter):
        for i in tem_idx:
            np.random.shuffle(i)
        i_trn = [iY[i][tem_idx[i][: sY[i][0]]] for i in range(dY)]
        i_tst = [iY[i][tem_idx[i][
            sY[i][0]: sY[i][1]]] for i in range(dY)]
        i_val = [iY[i][tem_idx[i][sY[i][1]:]] for i in range(dY)]
        i_trn = np.concatenate(i_trn, axis=0).tolist()
        i_val = np.concatenate(i_val, axis=0).tolist()
        i_tst = np.concatenate(i_tst, axis=0).tolist()
        if len(i_tst) == 0:
            i_tst = list(set(np.random.randint(nb_y, size=nb_tst)))
        if len(i_val) == 0:
            i_val = list(set(np.random.randint(nb_y, size=nb_val)))
        tem = (deepcopy(i_trn), deepcopy(i_val), deepcopy(i_tst))
        split_idx.append(deepcopy(tem))

        del i_trn, i_val, i_tst, tem, i
    del k, tem_idx, sY, lY, iY, dY, vY  # nb_iter
    gc.collect()
    return deepcopy(split_idx)


# ----------------------------------
# Cross-Validation
# ----------------------------------

def situation_cross_validation(nb_iter, y,
                               split_type='cross_valid_v3'):
    if split_type not in ["cross_valid_v3", "cross_valid_v2",
                          "cross_validation", "cross_valid"]:
        raise UserWarning("Check the number of datasets in the "
                          "cross-validation please.")

    y = np.array(y)
    vY = np.unique(y).tolist()
    dY = len(vY)
    iY = [np.where(y == j)[0] for j in vY]  # indexes
    lY = [len(j) for j in iY]  # length

    # tY = [np.arange(j) for j in lY]  # tmp_index
    tY = [np.copy(j) for j in iY]
    for j in tY:
        np.random.shuffle(j)
    sY = [int(np.floor(j / nb_iter)) for j in lY]  # split length
    if nb_iter == 2:
        sY = [int(np.floor(j / (nb_iter + 1))) for j in lY]
    elif nb_iter == 3:
        sY = [int(np.floor(j / (nb_iter + 1))) for j in lY]
    elif nb_iter == 1:
        sY = [int(np.floor(j / (nb_iter + 1))) for j in lY]

    split_idx = []
    for k in range(1, nb_iter + 1):
        i_tst, i_val, i_trn = [], [], []

        for i in range(dY):
            k_former = sY[i] * (k - 1)
            k_middle = sY[i] * k
            k_latter = sY[i] * (k + 1) if k != nb_iter else sY[i]

            i_tst.append(tY[i][k_former: k_middle])
            if k != nb_iter:
                i_val.append(tY[i][k_middle: k_latter])
                i_trn.append(np.concatenate([
                    tY[i][k_latter:], tY[i][: k_former]], axis=0))
            else:
                i_val.append(tY[i][: k_latter])
                i_trn.append(np.concatenate([
                    tY[i][k_middle:], tY[i][k_latter: k_former]],
                    axis=0))

        i_tst = np.concatenate(i_tst, axis=0).tolist()
        i_val = np.concatenate(i_val, axis=0).tolist()
        i_trn = np.concatenate(i_trn, axis=0).tolist()

        if split_type.endswith("v2"):
            # "cross_valid_v2" or "cross_validation"
            temp_ = (deepcopy(i_trn + i_val), deepcopy(i_tst))
        else:
            temp_ = (deepcopy(
                i_trn), deepcopy(i_val), deepcopy(i_tst))

        split_idx.append(deepcopy(temp_))
        del k_former, k_middle, k_latter, i_tst, i_val, i_trn
    del k, y, vY, dY, iY, lY, tY, sY  # , nb_iter
    gc.collect()
    return deepcopy(split_idx)


# ----------------------------------
# Split situation 1
# i.e., one single iteration
# ----------------------------------

def situation_split1(y, pr_trn, pr_tst=None):
    y = np.array(y)
    vY = np.unique(y).tolist()
    dY = len(vY)
    iY = [np.where(y == j)[0] for j in vY]
    lY = [len(j) for j in iY]  # index & length
    # tmp_idx = [np.arange(j) for j in lY]

    tY = [np.copy(j) for j in iY]
    for j in tY:
        np.random.shuffle(j)
    if pr_tst is not None:
        pr_val = 1. - pr_trn - pr_tst
        sY = [[int(np.max([np.round(j * i), 1])) for i in (
            pr_trn, pr_trn + pr_tst)] for j in lY]
    else:
        sY = [int(np.max([np.round(j * pr_trn), 1])) for j in lY]

    nb_y = len(y)
    nb_trn = int(np.round(nb_y * pr_trn))
    nb_trn = min(max(nb_trn, 1), nb_y - 1)
    if pr_tst is None:
        nb_tst = nb_y - nb_trn
    else:
        nb_tst = int(np.round(nb_y * pr_tst))
        nb_tst = min(max(nb_tst, 1), nb_y - 1)
        nb_val = nb_y - nb_trn - nb_tst
        del pr_val

    i_tst, i_val, i_trn = [], [], []
    for i in range(dY):
        if pr_tst is not None:
            i_trn.append(tY[i][: sY[i][0]])
            i_val.append(tY[i][sY[i][0]: sY[i][1]])
            i_tst.append(tY[i][sY[i][1]:])
        else:
            i_trn.append(tY[i][: sY[i]])
            i_tst.append(tY[i][sY[i]:])

    i_trn = np.concatenate(i_trn, axis=0).tolist()
    i_tst = np.concatenate(i_tst, axis=0).tolist()
    if len(i_tst) == 0:
        i_tst = list(set(np.random.randint(nb_y, size=nb_tst)))
    if pr_tst is not None:
        i_val = np.concatenate(i_val, axis=0).tolist()
        if len(i_val) == 0:
            i_val = list(set(np.random.randint(nb_y, size=nb_val)))
        split_idx = [(i_trn, i_val, i_tst)]
    else:
        split_idx = [(i_trn, i_tst)]
    return split_idx


# =====================================
# carry_split.py
# different ways to split data
# =====================================


# Split situations 2 & 3
# Cross-Validation (CV)
# -------------------------------------

def sklearn_k_fold_cv(nb_iter, y):
    kf = KFold(n_splits=nb_iter)
    # rkf = RepeatedKFold(n_splits=nb_iter, n_repeats=2)
    split_idx = []
    for trn, tst in kf.split(y):
        # split_idx.append((trn, tst))
        split_idx.append((trn.tolist(), tst.tolist()))
    return split_idx  # element: np.ndarray


def sklearn_stratify(nb_iter, y, X):
    kf = StratifiedKFold(n_splits=nb_iter)  # i.e. skf
    split_idx = []
    for trn, tst in kf.split(X, y):
        split_idx.append((trn.tolist(), tst.tolist()))
    return split_idx  # element: np.ndarray


def manual_cross_valid(nb_iter, y):
    split_idx = situation_cross_validation(nb_iter, y)
    return [[x + y, z] for x, y, z in split_idx]


def manual_repetitive(nb_iter, y, gen=False):
    num = len(y)
    if not gen:
        split_idx = [list(range(num)) for i in range(nb_iter)]
        for i in range(nb_iter):
            np.random.shuffle(split_idx[i])
    else:
        split_idx = [np.random.randint(
            num, size=num).tolist() for i in range(nb_iter)]
    return split_idx


# ==================================
# INTERFACE
# ==================================
# Preliminaries


# ----------------------------------
# obtain data
# from core.fetch_utils import different_type_of_data
# ----------------------------------


# ----------------------------------
# split data
# ----------------------------------


def split_into_train_validation_test(split_type, *split_args):
    if split_type == "2split":
        pr_trn, nb_iter, y_trn = split_args
        split_idx = situation_split2(pr_trn, nb_iter, y_trn)
        del y_trn
    elif split_type == "3split":
        pr_trn, pr_tst, nb_iter, y = split_args
        split_idx = situation_split3(pr_trn, pr_tst, nb_iter, y)
        del y
        #   #
    elif split_type.startswith("cross_valid"):
        # elif split_type == "cross_validation":
        nb_iter, y = split_args
        split_idx = situation_cross_validation(nb_iter, y, split_type)
        del y
    else:
        raise UserWarning("Error occurred in "
                          "`split_into_train_validation_test`.")
    gc.collect()
    return deepcopy(split_idx)  # list


def according_index_split_train_valid_test(X, y, split_idx_item):
    # X, y: np.ndarray
    idx_trn, idx_val, idx_tst = split_idx_item
    X_trn, y_trn = X[idx_trn], y[idx_trn]
    X_val, y_val = X[idx_val], y[idx_val]
    X_tst, y_tst = X[idx_tst], y[idx_tst]
    del idx_trn, idx_val, idx_tst
    return X_trn.tolist(), X_val.tolist(), X_tst.tolist(), \
        y_trn.tolist(), y_val.tolist(), y_tst.tolist()


# ----------------------------------
# normalization
#
# scaler, min_max_scaler, normalizer
# preprocessing for feature normalization
# ----------------------------------


def scale_normalize_helper(scale_type):
    assert scale_type in ["standard", "min_max", "normalize"]
    if scale_type == "standard":
        scaler = preprocessing.StandardScaler()
    elif scale_type == "min_max":
        scaler = preprocessing.MinMaxScaler()  # min_max_scaler
    elif scale_type == "normalize":
        scaler = preprocessing.Normalizer()  # normalizer
    else:
        raise LookupError("Correct the `scale_type` please.")
    return scaler  # deepcopy(scaler)


def scale_normalize_dataset(scaler, X_trn, X_val, X_tst):
    # scaler = scale_normalize_helper(scale_type)
    scaler = scaler.fit(X_trn)
    # scaler.fit(X_trn)  # id(scaler) would not change
    X_trn = scaler.transform(X_trn)
    X_val = scaler.transform(X_val) if len(X_val) > 0 else []
    X_tst = scaler.transform(X_tst)
    # return scaler, X_trn.tolist(), X_val.tolist(), X_tst.tolist()
    X_trn, X_tst = X_trn.tolist(), X_tst.tolist()
    X_val = X_val.tolist() if len(X_val) > 0 else []
    return scaler, X_trn, X_val, X_tst  # deepcopy


# ----------------------------------
# model selection
# ----------------------------------
# ref:
#   https://scikit-learn.org/stable/modules/cross_validation.html


def cross_validate_training_test_set(y, X=None, cv=5):
    if X is None:
        kf = model_selection.KFold(n_splits=cv)
        kf.get_n_splits(y)
        split_idx = [(
            i.tolist(), j.tolist()) for i, j in kf.split(y)]
    else:
        kf = model_selection.StratifiedKFold(n_splits=cv)
        kf.get_n_splits(X)
        split_idx = [(
            i.tolist(), j.tolist()) for i, j in kf.split(X, y)]
    return split_idx
