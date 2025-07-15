# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority voting


import os
# import pickle
# import pip
# import sys

from torch import nn
import pyro
# import pyro.distributions as dist
from pyro.nn import PyroModule

# from fairml.lawschool import (
from experiment.lawschool import (
    CURR_PATH, LawSchoolData, model_Unaware_or_Full,
    main_model_infer_K, main_model_Fair_Add)
from fairml.widget.utils_saver import elegant_print
from fairml.facils.fairness_group import (
    unpriv_group_one, unpriv_group_two, unpriv_group_thr,
    marginalised_pd_mat)  # .metrics.group_fair
from fairml.discriminative_risk import hat_L_fair, hat_L_loss


# -----------------------
# Case study


DTY_FLT = 'float'
SAV_PLT = False
CURR_FYA_LOC = -2
logger = None
ratio = .7


dataset = LawSchoolData()
data_frame = dataset.load_raw_dataset()
data_df = dataset.data_specific_processing(data_frame)

train_set, test_set = \
    dataset.data_partitioning(data_df, 42, "dat_dist")
train_adv = dataset.adversarial(train_set, ratio)
test_adv = dataset.adversarial(test_set, ratio)


# for CI testing
# smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)
# setup
assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)


def _generic_no4_adversarial(y_trn, y_insp, hx_qtb, picked, pick_trntst,
                             logger, threshold, positive_label, non_sa):
    clas_y_trn = (y_trn >= threshold).numpy().astype(DTY_FLT)
    lab_y_insp = (y_insp >= threshold).numpy().astype(DTY_FLT)
    lab_hx_qtb = (hx_qtb >= threshold).numpy().astype(DTY_FLT)
    # elegant_print([
    #     f"{picked} Model: on {pick_trntst} samples",
    #     f"\t\t threshold = {threshold}"], logger)

    _, _, g1_Cm, g0_Cm = marginalised_pd_mat(
        clas_y_trn, lab_y_insp, positive_label, non_sa)
    tmp_1 = unpriv_group_one(g1_Cm, g0_Cm)
    tmp_2 = unpriv_group_two(g1_Cm, g0_Cm)
    tmp_3 = unpriv_group_thr(g1_Cm, g0_Cm)
    grp_1 = abs(tmp_1[0] - tmp_1[1])
    grp_2 = abs(tmp_2[0] - tmp_2[1])
    grp_3 = abs(tmp_3[0] - tmp_3[1])
    elegant_print([
        "\t\t Normally (fairness)",
        "\t\t\t Group #1: {:.6f} {:.6f}  abs: {:.6f}".format(
            tmp_1[0], tmp_1[1], grp_1),
        "\t\t\t Group #2: {:.6f} {:.6f}  abs: {:.6f}".format(
            tmp_2[0], tmp_2[1], grp_2),
        "\t\t\t Group #3: {:.6f} {:.6f}  abs: {:.6f}".format(
            tmp_3[0], tmp_3[1], grp_3),
    ], logger)

    _, _, g1_Cm, g0_Cm = marginalised_pd_mat(
        clas_y_trn, lab_hx_qtb, positive_label, non_sa)
    tmp_1 = unpriv_group_one(g1_Cm, g0_Cm)
    tmp_2 = unpriv_group_two(g1_Cm, g0_Cm)
    tmp_3 = unpriv_group_thr(g1_Cm, g0_Cm)
    adv_1 = abs(tmp_1[0] - tmp_1[1])
    adv_2 = abs(tmp_2[0] - tmp_2[1])
    adv_3 = abs(tmp_3[0] - tmp_3[1])
    elegant_print([
        "\t\t Adversarial (fairness)",
        "\t\t\t Group #1: {:.6f} {:.6f}  abs: {:.6f}".format(
            tmp_1[0], tmp_1[1], adv_1),
        "\t\t\t Group #2: {:.6f} {:.6f}  abs: {:.6f}".format(
            tmp_2[0], tmp_2[1], adv_2),
        "\t\t\t Group #3: {:.6f} {:.6f}  abs: {:.6f}".format(
            tmp_3[0], tmp_3[1], adv_3),
    ], logger)

    elegant_print([
        "\t\t Discriminative risk (DR)",
        "\t\t\t  hat_loss : {:.12f}".format(hat_L_loss(
            lab_y_insp, clas_y_trn)),
        "\t\t\t  hat_fair : {:.12f}".format(hat_L_fair(
            lab_y_insp, lab_hx_qtb)),
        "\t\t\t GF drop #1: {:.8f}".format(grp_1 - adv_1),
        "\t\t\t GF drop #2: {:.8f}".format(grp_2 - adv_2),
        "\t\t\t GF drop #3: {:.8f}".format(grp_3 - adv_3),
    ], logger)  # diff

    return


def pseudo_classification(non_sa_trn, non_sa_tst, picked, logger,
                          curr_model, loss_fn, curr_dat_s,
                          threshold=.5, positive_label=1):
    non_sa_1, non_sa_2 = non_sa_trn
    non_sa_1_tst, non_sa_2_tst = non_sa_tst

    (train_y, y_insp, hx_qtb_trn,
     test_y, y_pred, hx_qtb_tst) = curr_dat_s

    # pdb.set_trace()
    elegant_print(f"\n{picked} Model:", logger)

    elegant_print(
        f"\t training, thres= {threshold}, non_sa#1 sex", logger)
    _generic_no4_adversarial(train_y, y_insp, hx_qtb_trn, picked,
                             'training', logger, threshold,
                             positive_label, non_sa_1)
    elegant_print(
        f"\t training, thres= {threshold}, non_sa#2 race", logger)
    _generic_no4_adversarial(train_y, y_insp, hx_qtb_trn, picked,
                             'training', logger, threshold,
                             positive_label, non_sa_2)

    elegant_print(
        f"\t test, thres= {threshold}, non_sa#1 sex", logger)
    _generic_no4_adversarial(test_y, y_pred, hx_qtb_tst, picked,
                             'test', logger, threshold,
                             positive_label, non_sa_1_tst)
    elegant_print(
        f"\t test, thres= {threshold}, non_sa#2 race", logger)
    _generic_no4_adversarial(test_y, y_pred, hx_qtb_tst, picked,
                             'test', logger, threshold,
                             positive_label, non_sa_2_tst)

    elegant_print('\n', logger)
    return


picked_unaware = ["ugpa", "lsat", "zfygpa"]
picked_full = ["sex", "race"] + picked_unaware
pick_figname = "visualize_model"

threshold = 0.  # .7
non_sa_trn = [(train_set['sex'] == 2).values,
              (train_set['race'] == 7).values]
non_sa_tst = [(test_set['sex'] == 2).values,
              (test_set['race'] == 7).values]


curr_model, loss_fn, curr_dat_s = model_Unaware_or_Full(
    train_set, test_set, picked_full, "Full", logger,
    pick_figname, train_adv, test_adv)
pseudo_classification(non_sa_trn, non_sa_tst, "Full", logger,
                      curr_model, loss_fn, curr_dat_s, threshold)

curr_model, loss_fn, curr_dat_s = model_Unaware_or_Full(
    train_set, test_set, picked_unaware, "Unaware", logger,
    pick_figname, train_adv, test_adv)
pseudo_classification(non_sa_trn, non_sa_tst, "Unaware", logger,
                      curr_model, loss_fn, curr_dat_s, threshold)

# Fair K model
curr_model, loss_fn, curr_dat_s = main_model_infer_K(
    train_set, test_set, logger, figname=pick_figname,
    train_adv=train_adv, test_adv=test_adv,
    CURR_FYA_LOC=CURR_FYA_LOC, CURR_PATH=CURR_PATH)
pseudo_classification(non_sa_trn, non_sa_tst, "Fair K", logger,
                      curr_model, loss_fn, curr_dat_s, threshold)

# Fair Add model
curr_model, loss_fn, curr_dat_s = main_model_Fair_Add(
    train_set, test_set, logger, figname=pick_figname,
    train_adv=train_adv, test_adv=test_adv)
pseudo_classification(non_sa_trn, non_sa_tst, "Fair add", logger,
                      curr_model, loss_fn, curr_dat_s, threshold)


if not SAV_PLT:
    # os.remove("dat_dist.pdf")
    os.remove(CURR_PATH + "dat_dist.pdf")

    os.remove(CURR_PATH + "visualize_model_Full.pdf")
    os.remove(CURR_PATH + "visualize_model_adv_Full.pdf")
    os.remove(CURR_PATH + "visualize_model_Unaware.pdf")
    os.remove(CURR_PATH + "visualize_model_adv_Unaware.pdf")

    os.remove(CURR_PATH + "visualize_model_FairK_InferK.pdf")
    os.remove(CURR_PATH + "visualize_model_fairK.pdf")
    os.remove(CURR_PATH + "visualize_model_fairAdd.pdf")
    os.remove(CURR_PATH + "test.pdf")
