# coding: utf-8
# examples.py


from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
# import pdb
# import pandas as pd
import numpy as np
import time

from fairml.datasets import German, preprocess  # Ricci,
from fairml.preprocessing import (
    adversarial, transform_X_and_y, transform_unpriv_tag)
from pyfair.facil.ensem_voting import weighted_voting
from pyfair.facil.utils_saver import elegant_print
from pyfair.facil.utils_timer import elegant_durat

from fairml.discriminative_risk import hat_L_fair, hat_L_loss
from pyfair.marble.metric_fair import (
    unpriv_group_one, unpriv_group_two, unpriv_group_thr,
    marginalised_np_mat, calc_fair_group,)
#     marginalised_pd_mat, prev_unpriv_grp_one, prev_unpriv_grp_two,
#     prev_unpriv_grp_thr)
from fairml.dr_pareto_optimal import POAF_PEP as POAF
from fairml.dr_pareto_optimal import Centralised_EPAF_Pruning as EPAF_C
from fairml.dr_pareto_optimal import Distributed_EPAF_Pruning as EPAF_D
from fairml.dr_pareto_optimal import Ranking_based_fairness_Pruning


# --------------------
# hyperparameters

# np.random.seed(16)

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
Xp_trn = Xp.iloc[i_trn]  # .to_numpy()
Xp_tst = Xp.iloc[i_tst]  # .to_numpy()
# X_trn, y_trn = X_trn.to_numpy(), y_trn.to_numpy()
# X_tst, y_tst = X_tst.to_numpy(), y_tst.to_numpy()
nsa_idx_trn = [idx[i_trn] for idx in non_sa]
nsa_idx_tst = [idx[i_tst] for idx in non_sa]


clf = ensemble.BaggingClassifier(n_estimators=nb_cls)
# pdb.set_trace()
clf.fit(X_trn, y_trn)
yhat_trn = [clf.estimators_[i].predict(
    X_trn).tolist() for i in range(nb_cls)]
yhat_tst = [clf.estimators_[i].predict(
    X_tst).tolist() for i in range(nb_cls)]
yp_qtb_trn = [clf.estimators_[i].predict(
    Xp_trn).tolist() for i in range(nb_cls)]
yp_qtb_tst = [clf.estimators_[i].predict(
    Xp_tst).tolist() for i in range(nb_cls)]


coef = [1. / nb_cls] * nb_cls
y_trn, y_tst = y_trn.tolist(), y_tst.tolist()
hens_trn = weighted_voting(yhat_trn, coef)
hens_tst = weighted_voting(yhat_tst, coef)
hp_qtb_trn = weighted_voting(yp_qtb_trn, coef)
hp_qtb_tst = weighted_voting(yp_qtb_tst, coef)


def get_accuracy(y, hens):
    # return np.mean(np.equal(y, h_ens))
    return metrics.accuracy_score(y, hens)


def get_grp_fairness(y, y_hat, pos_lbl, non_sa):
    g1_Cm, g0_Cm = marginalised_np_mat(y, y_hat, pos_lbl, non_sa)
    bias_grp1 = unpriv_group_one(g1_Cm, g0_Cm)
    bias_grp2 = unpriv_group_two(g1_Cm, g0_Cm)
    bias_grp3 = unpriv_group_thr(g1_Cm, g0_Cm)
    bias_grp1 = calc_fair_group(*bias_grp1)
    bias_grp2 = calc_fair_group(*bias_grp2)
    bias_grp3 = calc_fair_group(*bias_grp3)

    # '''
    # _, _, g1_Cm, g0_Cm = marginalised_pd_mat(
    #     y, y_hat, pos_lbl, non_sa)
    # comp_grp1 = prev_unpriv_grp_one(g1_Cm, g0_Cm)
    # comp_grp2 = prev_unpriv_grp_two(g1_Cm, g0_Cm)
    # comp_grp3 = prev_unpriv_grp_thr(g1_Cm, g0_Cm)
    # comp_grp1 = calc_fair_group(*comp_grp1)
    # comp_grp2 = calc_fair_group(*comp_grp2)
    # comp_grp3 = calc_fair_group(*comp_grp3)
    # pdb.set_trace()
    # '''
    return bias_grp1, bias_grp2, bias_grp3


bias_trn = [get_grp_fairness(
    y_trn, hens_trn, pos_label, i) for i in nsa_idx_trn]
bias_tst = [get_grp_fairness(
    y_tst, hens_tst, pos_label, i) for i in nsa_idx_tst]
# pdb.set_trace()


# '''
# print("\n\n")
# print("(Original) ensemble accuracy:")
# print("Training set:\t {:.5f} vs. {:.5f} aft perturbation".format(
#     get_accuracy(y_trn, hens_trn), get_accuracy(y_trn, hx_qtb_trn)))
# print(" Testing set:\t {:.5f} vs. {:.5f} aft perturbation".format(
#     get_accuracy(y_tst, hens_tst), get_accuracy(y_tst, hx_qtb_tst)))
#
# print("Discriminative risk:")
# print(" on training set:\t {:.5f}  & loss {:.5f}".format(
#     hat_L_fair(hens_trn, hx_qtb_trn), hat_L_loss(hens_trn, y_trn)))
# print("  on testing set:\t {:.5f}  & loss {:.5f}".format(
#     hat_L_fair(hens_tst, hx_qtb_tst), hat_L_loss(hens_tst, y_tst)))
# del hens_trn, hens_tst, hx_qtb_trn, hx_qtb_tst
#
# elegant_print([
#     "\n\n(Original) ensemble:",
#     "|                      | Training set | Test set |",
#     "|----------------------|--------------|----------|",
#     "| Accuracy             |   {:.8f}   | {:.8f} |",
#     "| acc aft perturbation |   {:.8f}   | {:.8f} |",
#     "| Discriminative risk  |   {:.8f}   | {:.8f} |",
#     "|   0/1 loss function  |   {:.8f}   | {:.8f} |"], None)
# '''

elegant_print([
    "\n\nEnsemble (originally):",
    "|                      | Training | Test set |",
    "|----------------------|----------|----------|",
    "| Accuracy             |  {:.5f} | {:.5f}  |".format(
        get_accuracy(y_trn, hens_trn),
        get_accuracy(y_trn, hp_qtb_trn)),
    "| acc aft perturbation |  {:.5f} | {:.5f}  |".format(
        get_accuracy(y_tst, hens_tst),
        get_accuracy(y_tst, hp_qtb_tst)),
    "| Discriminative risk  |  {:.5f} | {:.5f}  |".format(
        hat_L_fair(hens_trn, hp_qtb_trn)[0],
        hat_L_loss(hens_trn, y_trn)[0]),
    "|   0/1 loss function  |  {:.5f} | {:.5f}  |".format(
        hat_L_fair(hens_tst, hp_qtb_tst)[0],
        hat_L_loss(hens_tst, y_tst)[0]),
    "| Group fairness 1: DP | {:.6f} , {:.6f} | {:.6f} , {:.6f} |".format(
        bias_trn[0][0], bias_trn[1][0], bias_tst[0][0], bias_tst[1][0]),
    "| Group fairness 2: EO | {:.6f} , {:.6f} | {:.6f} , {:.6f} |".format(
        bias_trn[0][1], bias_trn[1][1], bias_tst[0][1], bias_tst[1][1]),
    "| Group fairness 3: PP | {:.6f} , {:.6f} | {:.6f} , {:.6f} |".format(
        bias_trn[0][2], bias_trn[1][2], bias_tst[0][2], bias_tst[1][2])], None)
del hens_trn, hens_tst, hp_qtb_trn, hp_qtb_tst


def get_subensemble(yhat_trn, yhat_tst, yp_qtb_trn, yp_qtb_tst,
                    coef, H, method=""):
    pru_hx_trn = np.array(yhat_trn)[H].tolist()
    pru_hx_tst = np.array(yhat_tst)[H].tolist()
    pru_hqtb_trn = np.array(yp_qtb_trn)[H].tolist()
    pru_hqtb_tst = np.array(yp_qtb_tst)[H].tolist()
    pruned_coef = np.array(coef)[H].tolist()

    hens_trn = weighted_voting(pru_hx_trn, pruned_coef)
    hens_tst = weighted_voting(pru_hx_tst, pruned_coef)
    hp_qtb_trn = weighted_voting(pru_hqtb_trn, pruned_coef)
    hp_qtb_tst = weighted_voting(pru_hqtb_tst, pruned_coef)

    '''
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
    '''

    bias_trn = [get_grp_fairness(
        y_trn, hens_trn, pos_label, i) for i in nsa_idx_trn]
    bias_tst = [get_grp_fairness(
        y_tst, hens_tst, pos_label, i) for i in nsa_idx_tst]

    elegant_print([
        f"\nPruned sub-ensemble:\t{method}",
        "|                      | Training | Test set |",
        "|----------------------|----------|----------|",
        "| Accuracy             |  {:.5f} | {:.5f}  |".format(
            get_accuracy(y_trn, hens_trn),
            get_accuracy(y_trn, hp_qtb_trn)),
        "| acc aft perturbation |  {:.5f} | {:.5f}  |".format(
            get_accuracy(y_tst, hens_tst),
            get_accuracy(y_tst, hp_qtb_tst)),
        "| Discriminative risk  |  {:.5f} | {:.5f}  |".format(
            hat_L_fair(hens_trn, hp_qtb_trn)[0],
            hat_L_loss(hens_trn, y_trn)[0]),
        "|   0/1 loss function  |  {:.5f} | {:.5f}  |".format(
            hat_L_fair(hens_tst, hp_qtb_tst)[0],
            hat_L_loss(hens_tst, y_tst)[0]),
        "| Group fairness 1: DP | {:.6f} , {:.6f} | {:.6f} , {:.6f} |".format(
            bias_trn[0][0], bias_trn[1][0], bias_tst[0][0], bias_tst[1][0]),
        "| Group fairness 2: EO | {:.6f} , {:.6f} | {:.6f} , {:.6f} |".format(
            bias_trn[0][1], bias_trn[1][1], bias_tst[0][1], bias_tst[1][1]),
        "| Group fairness 3: PP | {:.6f} , {:.6f} | {:.6f} , {:.6f} |".format(
            bias_trn[0][2], bias_trn[1][2], bias_tst[0][2], bias_tst[1][2]),
        f"\tH = {H}", ])


since = time.time()
seq = POAF(y_trn, yhat_trn, yp_qtb_trn, coef, lam, nb_pru)
H = np.zeros(nb_cls, dtype='bool')
H[seq] = 1
tim_elapsed = time.time() - since
get_subensemble(yhat_trn, yhat_tst, yp_qtb_trn, yp_qtb_tst,
                coef, H, "POAF")
elegant_print(f"\tseq= {seq}")
elegant_print(f"tim_elapsed: {elegant_durat(tim_elapsed, False)}")

since = time.time()
seq = EPAF_C(y_trn, yhat_trn, yp_qtb_trn, coef, nb_pru, lam)
tim_elapsed = time.time() - since
get_subensemble(yhat_trn, yhat_tst, yp_qtb_trn, yp_qtb_tst,
                coef, seq, "EPAF-C")
elegant_print(f"tim_elapsed: {elegant_durat(tim_elapsed, False)}")

since = time.time()
seq = EPAF_D(y_trn, yhat_trn, yp_qtb_trn, coef, nb_pru, lam, 2)
tim_elapsed = time.time() - since
get_subensemble(yhat_trn, yhat_tst, yp_qtb_trn, yp_qtb_tst,
                coef, seq, "EPAF-D (n_m=2)")
elegant_print(f"tim_elapsed: {elegant_durat(tim_elapsed, False)}")

# since = time.time()
# seq = EPAF_D(y_trn, yhat_trn, yp_qtb_trn, coef, nb_pru, lam, 3)
# tim_elapsed = time.time() - since
# get_subensemble(yhat_trn, yhat_tst, yp_qtb_trn, yp_qtb_tst,
#                 coef, seq, "EPAF-D (n_m=3)")
# elegant_print(f"tim_elapsed: {elegant_durat(tim_elapsed, False)}")


H, rank_idx = Ranking_based_fairness_Pruning(
    y_trn, yhat_trn, yp_qtb_trn, nb_pru, lam, 'DR', pos_label)
get_subensemble(yhat_trn, yhat_tst, yp_qtb_trn, yp_qtb_tst,
                coef, H, "ranking by DR")
elegant_print(f"\trank_idx = {rank_idx}")

for criterion in ['DP', 'EO', 'PQP']:
    for i, _ in enumerate(non_sa):
        H, rank_idx = Ranking_based_fairness_Pruning(
            y_trn, yhat_trn, yp_qtb_trn, nb_pru, lam,
            criterion, idx_priv=nsa_idx_trn[i])
        get_subensemble(
            yhat_trn, yhat_tst, yp_qtb_trn, yp_qtb_tst,
            coef, H, f"ranking by {criterion} using sa#{i+1}")
        elegant_print(f"\tseq= {np.where(H)[0].tolist()}")
        elegant_print(f"\trank_idx = {rank_idx}")


# pdb.set_trace()
