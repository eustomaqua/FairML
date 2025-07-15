# coding: utf-8

import numpy as np
import pdb
from fairml.widget.utils_const import check_equal, synthetic_dat, synthetic_clf

nb_inst, nb_cls, nb_lbl = 121, 4, 3
_, y_trn = synthetic_dat(nb_lbl, nb_inst, 4)
y_insp = synthetic_clf(y_trn, nb_cls, err=.2)
coef = np.random.rand(nb_cls)
coef /= np.sum(coef)
coef = coef.tolist()

y = [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
yt = [[1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0],
      [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
      [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
      [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0]]
wgh = [.2, .3, .1, .4]


def test_ensem_voting():
    from fairml.widget.ensem_voting import (
        plurality_voting, majority_voting, weighted_voting,
        tie_with_weight_plurality)

    f1 = plurality_voting(y_trn, y_insp)
    f2 = majority_voting(y_trn, y_insp)
    f3 = weighted_voting(y_trn, y_insp, coef)

    h1 = plurality_voting(y, yt)
    h2 = majority_voting(y, yt)
    h3 = weighted_voting(y, yt, wgh)
    h4 = weighted_voting(y, yt, [1. / 4 for _ in range(4)])
    assert np.equal(h1, h4).all()

    h1 = tie_with_weight_plurality(y, yt, None)
    h3 = tie_with_weight_plurality(y, yt, wgh)
    h4 = tie_with_weight_plurality(y, yt, [1. / 4] * 4)
    assert np.equal(h1, h4).all()
    assert np.equal(h1, h3).all()

    h1 = tie_with_weight_plurality(y, yt, None, nc=2)
    h3 = tie_with_weight_plurality(y, yt, wgh, nc=2)
    h4 = tie_with_weight_plurality(y, yt, [1. / 4] * 4, nc=2)
    assert np.equal(h1, h4).all()
    # pdb.set_trace()
    return
