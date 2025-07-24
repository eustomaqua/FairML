# coding: utf-8
# examples.py


from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
import numpy as np
import pandas as pd
import pdb

from fairml.datasets import Ricci, German, preprocess
from fairml.preprocessing import (
    adversarial, transform_X_and_y, transform_unpriv_tag)
from fairml.facils.ensem_voting import weighted_voting
from fairml.discriminative_risk import hat_L_fair, hat_L_loss
from fairml.dr_pareto_optimal import POAF_PEP as POAF
from fairml.dr_pareto_optimal import Ranking_based_fairness_Pruning


# --------------------
# hyperparameters

np.random.seed(16)

k = 2
nb_cls = 11
nb_pru = 5
lam = .5


# --------------------


dt = German()  # Ricci()
df = dt.load_raw_dataset()
processed_dat = preprocess(dt, df)
disturbed_dat = adversarial(dt, df, ratio=.97)
processed_dat = processed_dat['numerical-binsensitive']
disturbed_dat = disturbed_dat['numerical-binsensitive']
pos_label = dt.get_positive_class_val('')
non_sa, _ = transform_unpriv_tag(dt, processed_dat)
X, y = transform_X_and_y(dt, processed_dat)
Xp, _ = transform_X_and_y(dt, disturbed_dat)
y[y == 2] = 0  # only for German()


kf = model_selection.KFold(n_splits=5)
split_idx = []
for trn, tst in kf.split(y):
    split_idx.append((trn.tolist(), tst.tolist()))
del kf

i_trn, i_tst = split_idx[k]
X_trn, y_trn = X.iloc[i_trn], y.iloc[i_trn]
X_tst, y_tst = X.iloc[i_tst], y.iloc[i_tst]
X_qtb_trn = Xp.iloc[i_trn].to_numpy()
X_qtb_tst = Xp.iloc[i_tst].to_numpy()
X_trn, y_trn = X_trn.to_numpy(), y_trn.to_numpy()
X_tst, y_tst = X_tst.to_numpy(), y_tst.to_numpy()
nsa_idx_trn = [idx[i_trn] for idx in non_sa]
nsa_idx_tst = [idx[i_tst] for idx in non_sa]


clf = ensemble.BaggingClassifier(n_estimators=nb_cls)
clf.fit(X_trn, y_trn)
y_hat_trn = [clf.estimators_[i].predict(
    X_trn).tolist() for i in range(nb_cls)]
y_hat_tst = [clf.estimators_[i].predict(
    X_tst).tolist() for i in range(nb_cls)]
yt_qtb_trn = [clf.estimators_[i].predict(
    X_qtb_trn).tolist() for i in range(nb_cls)]
yt_qtb_tst = [clf.estimators_[i].predict(
    X_qtb_tst).tolist() for i in range(nb_cls)]


coef = [1. / nb_cls] * nb_cls
y_trn, y_tst = y_trn.tolist(), y_tst.tolist()
hens_trn = weighted_voting(y_hat_trn, coef)
hens_tst = weighted_voting(y_hat_tst, coef)
hx_qtb_trn = weighted_voting(yt_qtb_trn, coef)
hx_qtb_tst = weighted_voting(yt_qtb_tst, coef)


def get_accuracy(y, h_ens):
    # metrics.accuracy_score(y, hens)
    return np.mean(np.equal(y, h_ens))


print("\n\n")
print("(Original) ensemble accuracy:")
print("Training set:\t {:.5f} vs. {:.5f} aft perturbation".format(
    get_accuracy(y_trn, hens_trn), get_accuracy(y_trn, hx_qtb_trn)))
print(" Testing set:\t {:.5f} vs. {:.5f} aft perturbation".format(
    get_accuracy(y_tst, hens_tst), get_accuracy(y_tst, hx_qtb_tst)))

print("Discriminative risk:")
print(" on training set:\t {:.5f}  & loss {:.5f}".format(
    hat_L_fair(hens_trn, hx_qtb_trn), hat_L_loss(hens_trn, y_trn)))
print("  on testing set:\t {:.5f}  & loss {:.5f}".format(
    hat_L_fair(hens_tst, hx_qtb_tst), hat_L_loss(hens_tst, y_tst)))
del hens_trn, hens_tst, hx_qtb_trn, hx_qtb_tst


def get_subensemble(y_hat_trn, y_hat_tst, yt_qtb_trn, yt_qtb_tst,
                    coef, H, method=""):
    pruned_hx_trn = np.array(y_hat_trn)[H].tolist()
    pruned_hx_tst = np.array(y_hat_tst)[H].tolist()
    pru_hxqtb_trn = np.array(yt_qtb_trn)[H].tolist()
    pru_hxqtb_tst = np.array(yt_qtb_tst)[H].tolist()
    pruned_coef = np.array(coef)[H].tolist()

    hens_trn = weighted_voting(pruned_hx_trn, pruned_coef)
    hens_tst = weighted_voting(pruned_hx_tst, pruned_coef)
    hx_qtb_trn = weighted_voting(pru_hxqtb_trn, pruned_coef)
    hx_qtb_tst = weighted_voting(pru_hxqtb_tst, pruned_coef)

    print('\n')
    print("Pruned sub-ensemble accuracy:")
    print(f"\tpruning method:\t{method}")  # \n\tH= {H}
    print("Training set:\t {:.5f} vs. {:.5f} aft perturbation".format(
        get_accuracy(y_trn, hens_trn), get_accuracy(y_trn, hx_qtb_trn)))
    print(" Testing set:\t {:.5f} vs. {:.5f} aft perturbation".format(
        get_accuracy(y_tst, hens_tst), get_accuracy(y_tst, hx_qtb_tst)))
    print("DR on training set:\t {:.5f}  & loss {:.5f}".format(
        hat_L_fair(hens_trn, hx_qtb_trn), hat_L_loss(hens_trn, y_trn)))
    print("DR on testing set :\t {:.5f}  & loss {:.5f}".format(
        hat_L_fair(hens_tst, hx_qtb_tst), hat_L_loss(hens_tst, y_tst)))
    print("\tH= {H}")


seq = POAF(y_trn, y_hat_trn, yt_qtb_trn, coef, lam, nb_pru)
H = np.zeros(nb_cls, dtype='bool')
H[seq] = 1
get_subensemble(y_hat_trn, y_hat_tst, yt_qtb_trn, yt_qtb_tst,
                coef, H, "POAF")
print(f"\tseq= {seq}")

H, rank_idx = Ranking_based_fairness_Pruning(
    y_trn, y_hat_trn, yt_qtb_trn, nb_pru, lam, 'DR', pos_label)
get_subensemble(y_hat_trn, y_hat_tst, yt_qtb_trn, yt_qtb_tst,
                coef, H, "ranking by DR")
print(f"\trank_idx = {rank_idx}")

for criterion in ['DP', 'EO', 'PQP']:
    for i, _ in enumerate(non_sa):
        H, rank_idx = Ranking_based_fairness_Pruning(
            y_trn, y_hat_trn, yt_qtb_trn, nb_pru, lam,
            criterion, idx_priv=nsa_idx_trn[i])
        get_subensemble(
            y_hat_trn, y_hat_tst, yt_qtb_trn, yt_qtb_tst,
            coef, H, f"ranking by {criterion} using sa#{i+1}")
        print(f"\trank_idx = {rank_idx}")


pdb.set_trace()
# pdb.set_trace()
