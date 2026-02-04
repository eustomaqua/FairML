# coding: utf-8

import numpy as np
import pandas as pd


# binary classification

def synthetic_clf(y_trn, nb_cls, err=.2, prng=None):
    if not prng:
        prng = np.random
    nb_spl, nb_lbl = len(y_trn), len(set(y_trn))
    yt_insp = np.repeat(y_trn, repeats=nb_cls, axis=0)
    yt_insp = yt_insp.reshape(-1, nb_cls).T
    num = int(nb_spl * err)
    for k in range(nb_cls):
        for _ in range(num):
            i = prng.randint(nb_spl)
            yt_insp[k][i] = nb_lbl - 1 - yt_insp[k][i]
    return yt_insp.tolist()


def check_equal(tmp_a, tmp_b, diff=1e-7):
    return True if abs(tmp_a - tmp_b) < diff else False


nb_spl, nb_ftr, nb_lbl = 121, 7, 3
X_trn = np.random.rand(nb_spl, nb_ftr)
y_trn = np.random.randint(nb_lbl, size=nb_spl)
y_hat, hx_qtb = synthetic_clf(y_trn, 2, err=.2)
z_hat, hz_qtb = synthetic_clf(y_trn, 2, err=.3)


# discriminative risk

def test_my_DR():
    from fairml.discriminative_risk import (
        hat_L_fair, hat_L_loss, tandem_fair, tandem_loss,
        hat_L_objt, tandem_objt, cal_L_obj_v1, cal_L_obj_v2,
        perturb_numpy_ver, perturb_pandas_ver)

    ans = hat_L_fair(y_hat, hx_qtb)  # [0]
    res = hat_L_loss(y_hat, y_trn)   # [0]
    err = float(1. - np.mean(np.equal(y_trn, y_hat)))
    assert check_equal(err, res)

    ans = tandem_fair(y_hat, hx_qtb, z_hat, hz_qtb)
    res = tandem_loss(y_hat, z_hat, y_trn)
    lam = .5
    ans = hat_L_objt(y_hat, hx_qtb, y_trn, lam)
    assert 0. <= ans <= 1.
    ans = tandem_objt(y_hat, hx_qtb, z_hat, hz_qtb, y_trn, lam)
    assert 0. <= ans <= 1.

    nb_cls = 5
    yt_hat = synthetic_clf(y_trn, nb_cls, err=.15)
    yt_hat_qtb = synthetic_clf(y_trn, nb_cls, err=.2)
    coef = np.random.rand(nb_cls)
    coef /= np.sum(coef)
    coef = coef.tolist()
    res = cal_L_obj_v2(yt_hat, yt_hat_qtb, y_trn, coef)
    ans = cal_L_obj_v1(yt_hat, yt_hat_qtb, y_trn, coef)
    assert check_equal(res, ans)  # res == ans
    # import pandas as pd

    X = np.random.randint(5, size=(5, 4))
    X_qtb, _ = perturb_numpy_ver(X, [2, 1], [1, 1], ratio=.97)
    assert np.all(np.equal(X[:, [0, 3]], X_qtb[:, [0, 3]]))
    assert not np.equal(X, X_qtb).all()

    X = pd.DataFrame(X, columns=['A', 'B', 'C', 'D'])
    X_qtb, _ = perturb_pandas_ver(X, ['B', 'C'], [1, 0], ratio=.97)
    tmp = (X[['A', 'D']] == X_qtb[['A', 'D']]).all()
    assert tmp.to_numpy().all()
    assert not (X == X_qtb).to_numpy().all()
    return
