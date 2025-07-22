# coding: utf-8


import numpy as np
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier, RandomForestClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier)
# import lightgbm
from lightgbm import LGBMClassifier
from fairgbm import FairGBMClassifier
from fairml.widget.pkgs_AdaFair import AdaFair

from fairml.widget.metric_cont import contg_tab_mu_type2 \
    as contingency_table
from fairml.facils.metric_perf import (
    calc_accuracy, calc_precision, calc_recall, calc_f1_score,
    calc_fpr, calc_fnr, calc_sensitivity, calc_specificity,
    imba_geometric_mean, imba_discriminant_power,
    imba_Matthew_s_cc, imba_Cohen_s_kappa)  # calc_tpr,
# from prgm.nucleus.oracle_metric import (
#     marginalised_pd_mat, unpriv_unaware, unpriv_manual,
#     unpriv_group_one, unpriv_group_two, unpriv_group_thr)

from fairml.facils.metric_fair import marginalised_pd_mat
from fairml.dr_pareto_optimal import (
    unpriv_group_one, unpriv_group_two, unpriv_group_thr,
    unpriv_unaware, unpriv_manual)
# from fairml.facils.metric_fair import (
#     marginalised_pd_mat, prev_unpriv_unaware, prev_unpriv_manual,
#     prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr)
# unpriv_group_one = prev_unpriv_grp_one
# unpriv_group_two = prev_unpriv_grp_two
# unpriv_group_thr = prev_unpriv_grp_thr
# unpriv_unaware = prev_unpriv_unaware
# unpriv_manual = prev_unpriv_manual
# del prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr
# del prev_unpriv_unaware, prev_unpriv_manual
from fairml.discriminative_risk import (
    hat_L_fair, hat_L_loss)  # E_rho_L_fair_f,E_rho_L_loss_f
# from archv.fair.oracle_fetch import EnsembleSetup

from fairml.facils.utils_wpclf import (
    FAIR_INDIVIDUALS)  # ,TREE_ENSEMBLES,HOMO_ENSEMBLES)#,ALG_NAMES)
from fairml.dr_pareto_optimal import (
    Pareto_Optimal_EPAF_Pruning, POAF_PEP,  # POAF_PEP(alternative)
    Centralised_EPAF_Pruning, Distributed_EPAF_Pruning,
    Ranking_based_fairness_Pruning)
from fairml.widget.utils_const import unique_column
from fairml.facils.data_classify import EnsembleAlgorithm


# rev_baseline.py
import csv
import json
# import logging
import os
import sys
import time
from fairml.widget.utils_saver import (
    get_elogger, rm_ehandler, elegant_print)
from fairml.widget.utils_timer import elegant_dated, elegant_durat
from fairml.widget.utils_const import _get_tmp_name_ens
from fairml.widget.data_split import (  # situation_split1,
    sklearn_k_fold_cv, sklearn_stratify, manual_cross_valid)
from fairml.datasets import preprocess
from fairml.preprocessing import (
    adversarial, transform_X_and_y, transform_unpriv_tag,
    transform_perturbed)
from experiment.wp2_oracle.fetch_data import DataSetup


# =====================================
# Benchmarks
# =====================================


INDIVIDUALS = FAIR_INDIVIDUALS
del FAIR_INDIVIDUALS

ENSEM_NAMES = [
    "bagging", "adaboost", "rforest", "extrats",
    "gradbst", ]
ALG_NAMES = [
    "DT", "NB", "SVM", "linSVM", "MLP",  # "LR",
    "LR1", "LR2", "LM1", "LM2", "kNNu", "kNNd", ]


# -------------------------------------
# Classifier


class ClassifierSetup:
    def __init__(self):
        # super().__init__(None, None, None)
        pass

    def majority_vote(self, y, yt, wgt=None):
        # that is, majority_vote_subscript_rho()
        #     aka. weighted_voting
        # y = y.to_numpy()
        # y : np.ndarray or list
        # yt: list of np.ndarray

        if isinstance(yt[0], np.ndarray):
            yt = [i.tolist() for i in yt]
        vY = np.unique(np.concatenate([[y], yt]))
        # wgt = self._weight if not wgt else wgt
        if wgt is None:
            wgt = np.array([1 for i in yt])
            wgt = (wgt / np.sum(wgt)).tolist()
        coef = np.array([wgt]).transpose()
        weig = [np.sum(coef * np.equal(
            yt, i), axis=0).tolist() for i in vY]
        loca = np.array(weig).argmax(axis=0).tolist()
        fens = [vY[i] for i in loca]
        return fens

    def each_individual(self, abbr_cls):
        return INDIVIDUALS[abbr_cls]

    def each_ensemble(self, name_ens, abbr_cls="DT", nb_cls=21):
        if (name_ens == "bagging") and (abbr_cls != "DT"):
            clf = INDIVIDUALS[abbr_cls]  # inv
            return BaggingClassifier(clf, n_estimators=nb_cls)
        elif name_ens == "bagging":
            return BaggingClassifier(n_estimators=nb_cls)
        elif name_ens == "adaboost":
            return AdaBoostClassifier(n_estimators=nb_cls)
        elif name_ens == "rforest":
            return RandomForestClassifier(n_estimators=nb_cls)
        elif name_ens == "extrats":
            return ExtraTreesClassifier(n_estimators=nb_cls)
        elif name_ens == "gradbst":
            return GradientBoostingClassifier(n_estimators=nb_cls)
        raise ValueError("Wrong `name_ens`= {}".format(name_ens))

    def each_fair_ens(self, name_ens, nb_cls=2, constraint='',
                      saIndex=None, saValue=None):
        # above: clf.estimators_
        if name_ens == "lightgbm":
            # return lightgbm.LGBMClassifier(n_estimators=nb_cls)
            return LGBMClassifier(n_estimators=nb_cls)
        elif name_ens == "fairgbm":  # choices = 'FPR,FNR'
            return FairGBMClassifier(n_estimators=nb_cls,
                                     constraint_type=constraint)
        elif name_ens == "adafair":
            return AdaFair(n_estimators=nb_cls,
                           saIndex=saIndex, saValue=saValue)
        raise ValueError("Wrong `name_ens`= {}".format(name_ens))

    def count_single_member(self, y, y_hat, y_qtb,
                            non_sa, positive_label=1):
        # NB. must be np.ndarray
        tp, fp, fn, tn = contingency_table(y, y_hat, positive_label)

        res_indi = []
        res_indi.append(calc_accuracy(tp, fp, fn, tn))
        res_indi.append(calc_precision(tp, fp, fn, tn))
        res_indi.append(calc_recall(tp, fp, fn, tn))
        res_indi.append(calc_f1_score(tp, fp, fn, tn))

        # res_indi.append(calc_tpr(tp, fp, fn, tn))
        res_indi.append(calc_fpr(tp, fp, fn, tn))
        res_indi.append(calc_fnr(tp, fp, fn, tn))

        sen = calc_sensitivity(tp, fp, fn, tn)
        spe = calc_specificity(tp, fp, fn, tn)
        res_indi.extend([sen, spe, imba_geometric_mean(sen, spe)])
        res_indi.append(imba_discriminant_power(sen, spe))
        res_indi.append(imba_Matthew_s_cc(tp, fp, fn, tn))
        res_indi.extend(imba_Cohen_s_kappa(tp, fp, fn, tn))

        _, _, gones_Cm, gzero_Cm = marginalised_pd_mat(
            y, y_hat, positive_label, non_sa)  # g1_Cij, g0_Cij,
        cmp_fair = []
        # cmp_fair.extend(unpriv_unaware(gones_Cm, gzero_Cm))
        # cmp_fair.extend(unpriv_group_one(gones_Cm, gzero_Cm))
        # cmp_fair.extend(unpriv_group_two(gones_Cm, gzero_Cm))
        # cmp_fair.extend(unpriv_group_thr(gones_Cm, gzero_Cm))
        # cmp_fair.extend(unpriv_manual(gones_Cm, gzero_Cm))

        tmp_0 = unpriv_unaware(gones_Cm, gzero_Cm)
        tmp_1 = unpriv_group_one(gones_Cm, gzero_Cm)
        tmp_2 = unpriv_group_two(gones_Cm, gzero_Cm)
        tmp_3 = unpriv_group_thr(gones_Cm, gzero_Cm)
        tmp_4 = unpriv_manual(gones_Cm, gzero_Cm)
        cmp_fair.extend(tmp_0 + tmp_1 + tmp_2 + tmp_3 + tmp_4)
        cmp_fair.append(abs(tmp_0[0] - tmp_0[1]))
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))
        cmp_fair.append(abs(tmp_4[0] - tmp_4[1]))

        cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        cmp_fair.append(hat_L_loss(y_hat, y))

        # res_indi.shape: (13,)= (6+3+3+1,)
        # cmp_fair.shape: (17,)= (2*5+5+2,)
        return res_indi + cmp_fair  # shape: (30,)= (24+5+1,)

    def count_scores(self, clf,
                     X_trn, y_trn, Xd_trn, priv_trn,
                     X_tst, y_tst, Xd_tst, priv_tst,
                     positive_label):
        y_insp = clf.predict(X_trn)
        y_pred = clf.predict(X_tst)
        yq_insp = clf.predict(Xd_trn)
        yq_pred = clf.predict(Xd_tst)

        ans_trn = self.count_single_member(
            y_trn, y_insp, yq_insp, priv_trn, positive_label)
        ans_tst = self.count_single_member(
            y_tst, y_pred, yq_pred, priv_tst, positive_label)
        return ans_trn + ans_tst  # .shape: (60,)= (30*2,)

    '''
    def count_pruning(self, clf, name_ens, name_pru, nb_pru,
                      X_trn, y_trn, Xd_trn, sa_trn,
                      X_tst, y_tst, Xd_tst, sa_tst,
                      positive_label, dist=1, n_m=2, lam=.5):
        y_insp = [ind.predict(X_trn).tolist() for ind in clf.estimators_]
        y_pred = [ind.predict(X_tst).tolist() for ind in clf.estimators_]
        yq_insp = [ind.predict(Xd_trn).tolist() for ind in clf.estimators_]
        yq_pred = [ind.predict(Xd_tst).tolist() for ind in clf.estimators_]
        if name_ens == "bagging":
            coef = np.ones(len(clf.estimators_))
        elif name_ens == "adaboost":
            # coef = clf.estimator_weights_
            coef = 1 - clf.estimator_errors_
        coef /= np.sum(coef)
        coef = coef.tolist()

        if name_pru == "POEPAF":
            H = Pareto_Optimal_EPAF_Pruning(
                y_trn, y_insp, yq_insp, coef, nb_pru, lam)
        elif name_pru == "EPAF-C":
            H = Centralised_EPAF_Pruning(
                y_trn, y_insp, yq_insp, coef, nb_pru, lam)
        elif name_pru == "EPAF-D":
            H = Distributed_EPAF_Pruning(
                y_trn, y_insp, yq_insp, coef, nb_pru, lam, n_m)
        elif name_pru == "POPEP":
            H = POAF_PEP(y_trn, y_insp, yq_insp, coef, lam, nb_pru)

        ys_insp = np.array(y_insp)[H].tolist()
        ys_pred = np.array(y_pred)[H].tolist()
        yr_insp = np.array(yq_insp)[H].tolist()
        yr_pred = np.array(yq_pred)[H].tolist()
        coef = np.array(coef)[H].tolist()

        fens_trn = self.majority_vote(y_trn, ys_insp, coef)
        fens_tst = self.majority_vote(y_tst, ys_pred, coef)
        fqtb_trn = self.majority_vote(y_trn, yr_insp, coef)
        fqtb_tst = self.majority_vote(y_tst, yr_pred, coef)

        ans_trn = self.count_single_member(
          y_trn, fens_trn, fqtb_trn, sa_trn, positive_label)
        ans_tst = self.count_single_member(
          y_tst, fens_tst, fqtb_tst, sa_tst, positive_label)
        # TODO: revise, no need to do so many times of pruning
        return ans_trn + ans_tst  # .shape: (48,)= (24*2,)
    '''


# =====================================
# Experiments
# =====================================


# -------------------------------------
# To verify


class PartA_FairMeasure(ClassifierSetup):
    def __init__(self):
        super().__init__()

    def prepare_trial(self):
        csv_row_1 = unique_column(12 + 61)
        csv_row_2c = ['Train'] + [''] * 29 + ['Test'] + [''] * 29 + [
            'Time Cost (sec)']
        csv_row_3c_a = ['Accuracy', 'Precision', 'Recall',
                        'f1_score', 'fpr', 'fnr', 'sen', 'spe',
                        'g_mean', 'dp', 'Matthew', 'Cohen', '']  # 12+1
        csv_row_3c_b = ['unaware', '', 'group_one', '', 'group_two',
                        '', 'group_thr', '', 'manual', '']
        csv_row_3c_c = [
            'unaware', 'group_one', 'group_two', 'group_thr',
            'manual'] + ['hat_L(fair)', 'hat_L(loss)']  # 15+2
        csv_row_3c = (csv_row_3c_a + csv_row_3c_b + csv_row_3c_c +
                      csv_row_3c_a + csv_row_3c_b + csv_row_3c_c +
                      [''])  # 30*2+1
        csv_row_4c_a = [''] * 12 + ['(random_acc)'] + [
            'g1', 'g0'] * 5 + ['abs'] * 5 + [''] * 2
        csv_row_4c = csv_row_4c_a + csv_row_4c_a + ['/ut']  # 30*2+1
        del csv_row_3c_a, csv_row_3c_b, csv_row_3c_c, csv_row_4c_a
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    def schedule_content(self, nb_cls, logger,
                         X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                         X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                         saIndex, saValue, positive_label):
        #                    logger=None, nb_cls=21):
        # sa = gones_trn[0] if not jt_trn else jt_trn
        '''
        clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        # y_insp = clf.predict(X_trn)
        # y_pred = clf.predict(X_tst)

        for constraint_type in ["FPR", "FNR", "FPR,FNR"]:
          clf = FairGBMClassifier(n_estimators=nb_cls,
                                  constraint_type=constraint_type)
          clf.fit(X_trn, y_trn, constraint_group=sa)

        clf = AdaFair(n_estimators=nb_cls)
        '''
        res_iter = []

        if len(jt_trn) == 0:
            tmp = self.routine_one_sens_attr(
                nb_cls, logger,
                X_trn, y_trn, Xd_trn, gones_trn[0],
                X_tst, y_tst, Xd_tst, gones_tst[0],
                saIndex[0], saValue[0], positive_label)
            res_iter.append(tmp)
            return res_iter  # .shape: (#attr= 1, 5, 61)

        sa_len = len(gones_trn)
        for i in range(sa_len):
            tmp = self.routine_one_sens_attr(
                nb_cls, logger,
                X_trn, y_trn, Xd_trn, gones_trn[i],
                X_tst, y_tst, Xd_tst, gones_tst[i],
                saIndex[i], saValue[i], positive_label)
            res_iter.append(tmp)
        '''
        tmp = self.routine_one_sens_attr(
            nb_cls, logger,
            X_trn, y_trn, Xd_trn, jt_trn,
            X_tst, y_tst, Xd_tst, jt_tst,
            saIndex[-1], saValue[-1], positive_label)
        res_iter.append(tmp)
        '''
        return res_iter  # .shape: (#attr= 2, 5, 61)

    def routine_one_sens_attr(self, nb_cls, logger,
                              X_trn, y_trn, Xd_trn, non_sa_trn,
                              X_tst, y_tst, Xd_tst, non_sa_tst,
                              sa_idx, sa_val, positive_label):
        res_attr = []

        ut = time.time()
        clf = BaggingClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        res_attr.append(self.count_scores(
            clf,
            X_trn, y_trn, Xd_trn, non_sa_trn,
            X_tst, y_tst, Xd_tst, non_sa_tst,
            positive_label) + [ut, ])

        ut = time.time()
        clf = AdaBoostClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        res_attr.append(self.count_scores(
            clf,
            X_trn, y_trn, Xd_trn, non_sa_trn,
            X_tst, y_tst, Xd_tst, non_sa_tst,
            positive_label) + [ut, ])

        ut = time.time()
        # clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
        clf = LGBMClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        res_attr.append(self.count_scores(
            clf,
            X_trn, y_trn, Xd_trn, non_sa_trn,
            X_tst, y_tst, Xd_tst, non_sa_tst,
            positive_label) + [ut, ])

        for constraint_type in ["FPR", "FNR", "FPR,FNR"]:
            ut = time.time()
            clf = FairGBMClassifier(n_estimators=nb_cls,
                                    constraint_type=constraint_type)
            clf.fit(X_trn, y_trn, constraint_group=~non_sa_trn)
            ut = time.time() - ut
            res_attr.append(self.count_scores(
                clf,
                X_trn, y_trn, Xd_trn, non_sa_trn,
                X_tst, y_tst, Xd_tst, non_sa_tst,
                positive_label) + [ut, ])

        ut = time.time()
        clf = AdaFair(n_estimators=nb_cls,
                      saIndex=sa_idx, saValue=sa_val)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        res_attr.append(self.count_scores(
            clf,
            X_trn, y_trn, Xd_trn, non_sa_trn,
            X_tst, y_tst, Xd_tst, non_sa_tst,
            positive_label) + [ut, ])

        return res_attr  # .shape: (5,61)= (5,48+5*2+1+2)


class PartD_FairMeasure(PartA_FairMeasure):
    def __init__(self):
        super().__init__()

    def prepare_trial(self):
        csv_row_1 = unique_column(12 + 87 + 26)

        csv_row_2c_a = ['Train'] + [''] * 38 + ['Train'] + [''] * 16
        csv_row_2c_b = ['Test'] + [''] * 38 + ['Test'] + [''] * 16
        csv_row_2c = csv_row_2c_a + csv_row_2c_b + [
            'Time Cost (sec)']  # (39+17)*2+1
        csv_row_3c_a = ['Normal'] + [''] * 12 + ['Adversarial'] + [
            ''] * 12 + ['abs'] + [''] * 12  # 13*3
        csv_row_3c_b = ['Fairness'] + [''] * 9 + ['Group'] + [''] * 4
        csv_row_3c_c = ['Proposed', '']
        csv_row_3c_d = csv_row_3c_a + csv_row_3c_b + csv_row_3c_c  # 13*3+15+2=56
        csv_row_3c = csv_row_3c_d + csv_row_3c_d + ['']  # 56+56+1 =113
        del csv_row_3c_a, csv_row_3c_b, csv_row_3c_c, csv_row_3c_d
        csv_row_4c_a = ['Accuracy', 'Precision', 'Recall', 'f1_score',
                        'fpr', 'fnr', 'sen', 'spe', 'g_mean', 'dp',
                        'Matthew', 'Cohen', '(random_acc)']  # 12+1
        csv_row_4c_b = ['g1', 'g0'] * 5 + ['abs'] * 5
        csv_row_4c_c = ['hat_L(fair)', 'hat_L(loss)']
        csv_row_4c_d = (csv_row_4c_a + csv_row_4c_a + [
            'abs(Acc)', 'abs(P)', 'abs(R)', 'abs(f1)',
            'abs(fpr)', 'abs(fnr)', 'abs(sen)', 'abs(spe)',
            'abs(g_)', 'abs(dp)', 'abs(_M)', 'abs(_C)', ''
        ] + csv_row_4c_b + csv_row_4c_c)  # 13*2+13+(15+2) =26+17+13 =43+13
        csv_row_4c = csv_row_4c_d + csv_row_4c_d + ['/ut']  # 86+1+26
        del csv_row_4c_a, csv_row_4c_b, csv_row_4c_c, csv_row_4c_d
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    # def schedule_content(self):
    #   return res_iter  # .shape (#attr=1/2,7,87)
    # def routine_one_sens_attr(self):
    #   return res_attr  # .shape (7,86+1) -> (7,112+1)
    # def count_scores(self):
    #   return ans_trn+ans_tst  # .shape (43*2,) -> (56*2,)

    def count_single_member(self, y, y_hat, y_qtb,
                            non_sa, positive_label=1):
        res_norm = self.sub_count_normally(y, y_hat, positive_label)
        res_irre = self.sub_count_normally(y, y_qtb, positive_label)
        # irregular, adversarial
        tmp = [abs(i - j) for i, j in zip(res_norm, res_irre)]
        res_fair = self.sub_count_together(y, y_hat, y_qtb,
                                           non_sa, positive_label)
        # return res_norm + res_irre + res_fair  # .shape (26+17,)
        return res_norm + res_irre + tmp + res_fair  # .shape (39+17)= (13*3+,)

    def sub_count_normally(self, y, y_hat, positive_label):
        tp, fp, fn, tn = contingency_table(y, y_hat, positive_label)

        res_indi = []
        res_indi.append(calc_accuracy(tp, fp, fn, tn))
        res_indi.append(calc_precision(tp, fp, fn, tn))
        res_indi.append(calc_recall(tp, fp, fn, tn))
        res_indi.append(calc_f1_score(tp, fp, fn, tn))

        res_indi.append(calc_fpr(tp, fp, fn, tn))
        res_indi.append(calc_fnr(tp, fp, fn, tn))

        sen = calc_sensitivity(tp, fp, fn, tn)
        spe = calc_specificity(tp, fp, fn, tn)
        res_indi.extend([sen, spe, imba_geometric_mean(sen, spe)])
        res_indi.append(imba_discriminant_power(sen, spe))
        res_indi.append(imba_Matthew_s_cc(tp, fp, fn, tn))
        res_indi.extend(imba_Cohen_s_kappa(tp, fp, fn, tn))

        return res_indi  # .shape (13,)= (6+3+2+2,)

    def sub_count_together(self, y, y_hat, y_qtb,
                           non_sa, positive_label):
        g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
            marginalised_pd_mat(y, y_hat, positive_label, non_sa)

        tmp_0 = unpriv_unaware(gones_Cm, gzero_Cm)
        tmp_1 = unpriv_group_one(gones_Cm, gzero_Cm)
        tmp_2 = unpriv_group_two(gones_Cm, gzero_Cm)
        tmp_3 = unpriv_group_thr(gones_Cm, gzero_Cm)
        tmp_4 = unpriv_manual(gones_Cm, gzero_Cm)

        cmp_fair = []
        cmp_fair.extend(tmp_0 + tmp_1 + tmp_2 + tmp_3 + tmp_4)

        cmp_fair.append(abs(tmp_0[0] - tmp_0[1]))
        cmp_fair.append(abs(tmp_1[0] - tmp_1[1]))
        cmp_fair.append(abs(tmp_2[0] - tmp_2[1]))
        cmp_fair.append(abs(tmp_3[0] - tmp_3[1]))
        cmp_fair.append(abs(tmp_4[0] - tmp_4[1]))

        cmp_fair.append(hat_L_fair(y_hat, y_qtb))
        cmp_fair.append(hat_L_loss(y_hat, y))
        return cmp_fair  # .shape (17)= (2*5+5+2,)


# -------------------------------------
# To verify


CRITERIA = ["unaware", "DP", "EO", "PQP", "manual", "DR"]


class PartB_FairMeasure(ClassifierSetup):
    def __init__(self):
        super().__init__()

    def prepare_trial(self):
        csv_row_1 = unique_column(12 + 63)
        csv_row_2c = ['Train'] + [''] * 29 + ['Test'] + [''] * 29 + [
            'Time Cost (sec)', 'H', 'rank']  # 30*2+3 =63
        csv_row_3c_a = ['Accuracy', 'Precision', 'Recall', 'f1_score',
                        'fpr', 'fnr', 'sen', 'spe', 'g_mean',
                        'dp', 'Matthew', 'Cohen', '']  # 6+(6+1)=13
        csv_row_3c_b = ['unaware', '', 'group_one', '', 'group_two',
                        '', 'group_thr', '', 'manual', '']  # 5*2=10
        csv_row_3c_c = [
            'unaware', 'group_one', 'group_two', 'group_thr',
            'manual'] + ['hat_L(fair)', 'hat_L(loss)']  # 5+2=7
        csv_row_3c = (csv_row_3c_a + csv_row_3c_b + csv_row_3c_c +
                      csv_row_3c_a + csv_row_3c_b + csv_row_3c_c + [''] * 3)
        csv_row_4c_a = [''] * 12 + ['(random_acc)'] + [
            'g1', 'g0'] * 5 + ['abs'] * 5 + [''] * 2  # 13+10+5+2 =30
        csv_row_4c = csv_row_4c_a + csv_row_4c_a + ['/ut', 'H', '/idx']  # 30*2+3
        del csv_row_3c_a, csv_row_3c_b, csv_row_3c_c, csv_row_4c_a
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    def schedule_content(self, nb_cls, nb_pru, logger,
                         X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                         X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                         # saIndex, saValue,
                         positive_label, lam):
        res_iter = []
        tmp = self.routine_each_ensemble('bagging', '', nb_cls, nb_pru, logger,
                                         X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                                         X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                                         positive_label, lam)
        res_iter.append(tmp)

        for abbr_cls in ALG_NAMES[1:]:
            tmp = self.routine_each_ensemble('bagging', abbr_cls, nb_cls, nb_pru, logger,
                                             X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                                             X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                                             positive_label, lam)
            res_iter.append(tmp)

        '''
        tmp = self.routine_each_ensemble(
            'AdaBoost', '', nb_cls, nb_pru, logger,
            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
            X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
            positive_label, lam)
        res_iter.append(tmp)
        '''
        return res_iter  # .shape (1+(11-1)+1, #attr=1/3, 8, 63)

    def routine_each_ensemble(
            self, name_ens, abbr_cls, nb_cls, nb_pru, logger,
            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
            X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
            positive_label, lam):
        ut = time.time()
        if name_ens == "AdaBoost":
            clf = AdaBoostClassifier(n_estimators=nb_cls)
            clf.fit(X_trn, y_trn)
            coef = 1 - clf.estimator_errors_
            coef /= np.sum(coef)
            coef = coef.tolist()
        elif name_ens == "bagging" and abbr_cls == "":
            clf = BaggingClassifier(n_estimators=nb_cls)
            clf.fit(X_trn, y_trn)
            coef = [1. / nb_cls for _ in range(nb_cls)]
        else:
            clf = BaggingClassifier(
                base_estimator=INDIVIDUALS[abbr_cls],
                n_estimators=nb_cls)
            clf.fit(X_trn, y_trn)
            coef = [1. / nb_cls for _ in range(nb_cls)]
        ut = time.time() - ut

        fens_trn = clf.predict(X_trn)
        fens_tst = clf.predict(X_tst)
        fqtb_trn = clf.predict(Xd_trn)
        fqtb_tst = clf.predict(Xd_tst)

        y_insp = [ind.predict(
            X_trn).tolist() for ind in clf.estimators_]
        y_pred = [ind.predict(
            X_tst).tolist() for ind in clf.estimators_]
        yq_insp = [ind.predict(
            Xd_trn).tolist() for ind in clf.estimators_]
        yq_pred = [ind.predict(
            Xd_tst).tolist() for ind in clf.estimators_]

        res_iter = []
        if len(jt_trn) == 0:
            ans_trn = self.count_single_member(
                y_trn, fens_trn, fqtb_trn, gones_trn[0],
                positive_label)
            ans_tst = self.count_single_member(
                y_tst, fens_tst, fqtb_tst, gones_tst[0],
                positive_label)
            res_ens = [ans_trn + ans_tst + [ut, '', '']]
            tmp = self.routine_rank_pruning(
                nb_pru, coef, logger,
                y_trn, y_insp, yq_insp, gones_trn[0],
                y_tst, y_pred, yq_pred, gones_tst[0],
                positive_label, lam)
            res_ens.extend(tmp)  # .shape (8,63)= (1+7, 30*2+3)
            res_iter.append(res_ens)
            return res_iter  # .shape: (#attr=1, 8, 63)

        sa_len = len(gones_trn)
        for i in range(sa_len):
            ans_trn = self.count_single_member(
                y_trn, fens_trn, fqtb_trn, gones_trn[i],
                positive_label)
            ans_tst = self.count_single_member(
                y_tst, fens_tst, fqtb_tst, gones_tst[i],
                positive_label)
            res_ens = [ans_trn + ans_tst + [''] * 3]
            tmp = self.routine_rank_pruning(
                nb_pru, coef, logger,
                y_trn, y_insp, yq_insp, gones_trn[i],
                y_tst, y_pred, yq_pred, gones_tst[i],
                positive_label, lam)
            res_ens.extend(tmp)
            res_iter.append(res_ens)
        ans_trn = self.count_single_member(
            y_trn, fens_trn, fqtb_trn, jt_trn, positive_label)
        ans_tst = self.count_single_member(
            y_tst, fens_tst, fqtb_tst, jt_tst, positive_label)
        res_ens = [ans_trn + ans_tst + [ut] + [''] * 2]
        tmp = self.routine_rank_pruning(
            nb_pru, coef, logger,
            y_trn, y_insp, yq_insp, jt_trn,
            y_tst, y_pred, yq_pred, jt_tst,
            positive_label, lam)
        res_ens.extend(tmp)
        res_iter.append(res_ens)
        return res_iter  # .shape (#attr=3, 8, 63)

    def routine_rank_pruning(self, nb_pru, wgt, logger,
                             y_trn, y_insp, yq_insp, non_sa_trn,
                             y_tst, y_pred, yq_pred, non_sa_tst,
                             positive_label, lam):
        res_pru = []
        fens_trn = self.majority_vote(y_trn, y_insp, wgt)
        fens_tst = self.majority_vote(y_tst, y_pred, wgt)
        fqtb_trn = self.majority_vote(y_trn, yq_insp, wgt)
        fqtb_tst = self.majority_vote(y_tst, yq_pred, wgt)
        fens_trn, fens_tst = np.array(fens_trn), np.array(fens_tst)
        fqtb_trn, fqtb_tst = np.array(fqtb_trn), np.array(fqtb_tst)

        ans_trn = self.count_single_member(
            y_trn, fens_trn, fqtb_trn, non_sa_trn, positive_label)
        ans_tst = self.count_single_member(
            y_tst, fens_tst, fqtb_tst, non_sa_tst, positive_label)
        res_pru.append(ans_trn + ans_tst + ['', '', ''])

        for rule in CRITERIA:
            ut = time.time()
            H, idx = Ranking_based_fairness_Pruning(
                y_trn, y_insp, yq_insp, nb_pru, lam,
                rule, positive_label, non_sa_trn)
            ut = time.time() - ut

            ys_insp = np.array(y_insp)[H].tolist()
            ys_pred = np.array(y_pred)[H].tolist()
            yr_insp = np.array(yq_insp)[H].tolist()
            yr_pred = np.array(yq_pred)[H].tolist()
            coef = np.array(wgt)[H].tolist()

            fens_trn = self.majority_vote(y_trn, ys_insp, coef)
            fens_tst = self.majority_vote(y_tst, ys_pred, coef)
            fqtb_trn = self.majority_vote(y_trn, yr_insp, coef)
            fqtb_tst = self.majority_vote(y_tst, yr_pred, coef)
            fens_trn, fens_tst = np.array(fens_trn), np.array(fens_tst)
            fqtb_trn, fqtb_tst = np.array(fqtb_trn), np.array(fqtb_tst)

            ans_trn = self.count_single_member(
                y_trn, fens_trn, fqtb_trn, non_sa_trn, positive_label)
            ans_tst = self.count_single_member(
                y_tst, fens_tst, fqtb_tst, non_sa_tst, positive_label)
            res_pru.append(ans_trn + ans_tst + [ut, H, idx])
        # res_pru.shape: (1+6 ens+#criteria, 30+30+3)
        return res_pru  # .shape (1+6, 63)

    '''
    def routine_proposed_item(self, nb_cls, nb_pru, logger,
                            X_trn, y_trn, Xd_trn, sa_trn,
                            X_tst, y_tst, Xd_tst, sa_tst,
                            sa_idx, sa_val, positive_label):
        res_prop = []
        
        >>> func('POEPAF')
        (98.91268467903137, [0, 2, 3, 5, 9, 11, 20, 21, 22, 28, 30])
        >>> func('EPAF-C')
        (0.17669177055358887, [0, 2, 5, 7, 9, 11, 16, 18, 22, 23, 24])
        >>> func('EPAF-D')
        (1.0444560050964355, [0, 2, 5, 7, 9, 11, 16, 22, 23, 24, 28])
        >>> func('POPEP')
        (41.36838722229004, [9, 11])
        
        return res_prop
    '''


class PartC_FairMeasure(PartB_FairMeasure):
    def __init__(self, name_ens, abbr_cls):
        super().__init__()
        self._name_ens = name_ens
        self._abbr_cls = abbr_cls
    # def prepare_trial(self):
    #   pass

    def schedule_content(self, nb_cls, nb_pru, logger,
                         X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                         X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                         positive_label, lam):
        res_iter = []
        tmp = self.routine_each_ensemble(
            nb_cls, nb_pru, logger,
            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
            X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
            positive_label, lam)
        res_iter.append(tmp)
        return res_iter  # .shape (1, #attr=1/3, 7, 63)

    def routine_each_ensemble(self, nb_cls, nb_pru, logger,
                              X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                              X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                              positive_label, lam):
        ut = time.time()
        # def achieve_ensemble_from_train_set():
        name_cls = INDIVIDUALS[self._abbr_cls]
        coef, clfs, indices = EnsembleAlgorithm(
            self._name_ens, name_cls, nb_cls, X_trn, y_trn)
        # def achieve_ensemble_from_train_set():
        ut = time.time() - ut

        y_insp = [j.predict(X_trn).tolist() for j in clfs]
        y_pred = [j.predict(X_tst).tolist() for j in clfs]
        yq_insp = [j.predict(Xd_trn).tolist() for j in clfs]
        yq_pred = [j.predict(Xd_tst).tolist() for j in clfs]
        fens_trn = self.majority_vote(y_trn, y_insp, coef)
        fens_tst = self.majority_vote(y_tst, y_pred, coef)
        fqtb_trn = self.majority_vote(y_trn, yq_insp, coef)
        fqtb_tst = self.majority_vote(y_tst, yq_pred, coef)
        fens_trn, fens_tst = np.array(fens_trn), np.array(fens_tst)
        fqtb_trn, fqtb_tst = np.array(fqtb_trn), np.array(fqtb_tst)

        res_iter = []
        if len(jt_trn) == 0:
            ans_trn = self.count_single_member(
                y_trn, fens_trn, fqtb_trn, gones_trn[0],
                positive_label)
            ans_tst = self.count_single_member(
                y_tst, fens_tst, fqtb_tst, gones_tst[0],
                positive_label)
            res_ens = [ans_trn + ans_tst + [ut, '', '']]
            tmp = self.routine_rank_pruning(
                nb_pru, coef, logger,
                y_trn, y_insp, yq_insp, gones_trn[0],
                y_tst, y_pred, yq_pred, gones_tst[0],
                positive_label, lam)
            res_ens.extend(tmp)  # .shape (7,63)= (1+6, 30*2+3)
            res_iter.append(res_ens)
            return res_iter  # .shape: (#attr=1, 7, 63)

        sa_len = len(gones_trn)
        for i in range(sa_len):
            ans_trn = self.count_single_member(
                y_trn, fens_trn, fqtb_trn, gones_trn[i],
                positive_label)
            ans_tst = self.count_single_member(
                y_tst, fens_tst, fqtb_tst, gones_tst[i],
                positive_label)
            res_ens = [ans_trn + ans_tst + [''] * 3]
            tmp = self.routine_rank_pruning(
                nb_pru, coef, logger,
                y_trn, y_insp, yq_insp, gones_trn[i],
                y_tst, y_pred, yq_pred, gones_tst[i],
                positive_label, lam)
            res_ens.extend(tmp)
            res_iter.append(res_ens)
        ans_trn = self.count_single_member(
            y_trn, fens_trn, fqtb_trn, jt_trn, positive_label)
        ans_tst = self.count_single_member(
            y_tst, fens_tst, fqtb_tst, jt_tst, positive_label)
        res_ens = [ans_trn + ans_tst + [ut] + [''] * 2]
        tmp = self.routine_rank_pruning(
            nb_pru, coef, logger,
            y_trn, y_insp, yq_insp, jt_trn,
            y_tst, y_pred, yq_pred, jt_tst,
            positive_label, lam)
        res_ens.extend(tmp)
        res_iter.append(res_ens)
        return res_iter  # .shape: (#attr=3, 7, 63)

    def routine_rank_pruning(self, nb_pru, wgt, logger,
                             y_trn, y_insp, yq_insp, non_sa_trn,
                             y_tst, y_pred, yq_pred, non_sa_tst,
                             positive_label, lam):
        res_pru = []
        for rule in CRITERIA:
            ut = time.time()
            H, idx = Ranking_based_fairness_Pruning(
                y_trn, y_insp, yq_insp, nb_pru, lam,
                rule, positive_label, non_sa_trn)
            ut = time.time() - ut

            ys_insp = np.array(y_insp)[H].tolist()
            ys_pred = np.array(y_pred)[H].tolist()
            yr_insp = np.array(yq_insp)[H].tolist()
            yr_pred = np.array(yq_pred)[H].tolist()
            coef = np.array(wgt)[H].tolist()

            fens_trn = self.majority_vote(y_trn, ys_insp, coef)
            fens_tst = self.majority_vote(y_tst, ys_pred, coef)
            fqtb_trn = self.majority_vote(y_trn, yr_insp, coef)
            fqtb_tst = self.majority_vote(y_tst, yr_pred, coef)
            fens_trn, fens_tst = np.array(fens_trn), np.array(fens_tst)
            fqtb_trn, fqtb_tst = np.array(fqtb_trn), np.array(fqtb_tst)

            ans_trn = self.count_single_member(
                y_trn, fens_trn, fqtb_trn, non_sa_trn, positive_label)
            ans_tst = self.count_single_member(
                y_tst, fens_tst, fqtb_tst, non_sa_tst, positive_label)
            res_pru.append(ans_trn + ans_tst + [ut, H, idx])
        return res_pru  # .shape (6=#criteria, 30+30+3)


# -------------------------------------
# To verify theorems


class PartE_FairPruning(PartD_FairMeasure):
    # def __init__(self):
    #   super().__init__()

    def prepare_trial(self):
        csv_row_1 = unique_column(12 + 337)  # 56*6+1

        csv_row_2c_a = ['A Trn'] + [''] * 38 + ['A Trn'] + [''] * 16
        csv_row_2c_b = ['A Tst'] + [''] * 38 + ['A Tst'] + [''] * 16
        csv_row_2c_c = ['B Trn'] + [''] * 38 + ['B Trn'] + [''] * 16
        csv_row_2c_d = ['B Tst'] + [''] * 38 + ['B Tst'] + [''] * 16
        csv_row_2c_e = ['J Trn'] + [''] * 38 + ['J Trn'] + [''] * 16
        csv_row_2c_f = ['J Tst'] + [''] * 38 + ['J Tst'] + [''] * 16
        csv_row_2c = (csv_row_2c_a + csv_row_2c_b + csv_row_2c_c +
                      csv_row_2c_d + csv_row_2c_e + csv_row_2c_f +
                      ['Time Cost (sec)'])  # 56*6+1 =337
        del csv_row_2c_a, csv_row_2c_b, csv_row_2c_c
        del csv_row_2c_d, csv_row_2c_e, csv_row_2c_f

        csv_row_3c_a = ['Normal'] + [''] * 12 + [
            'Adversarial'] + [''] * 12 + ['abs'] + [''] * 12
        csv_row_3c_b = ['Fairness'] + [''] * 9 + [
            'Group'] + [''] * 4 + ['Proposed', '']
        csv_row_3c_c = csv_row_3c_a + csv_row_3c_b  # 39+17=56
        csv_row_3c = csv_row_3c_c * 6 + ['']  # 56*6+1 =337
        # csv_row_3c = (csv_row_3c_a + csv_row_3c_b +
        #               csv_row_3c_a + csv_row_3c_b + [''])
        del csv_row_3c_a, csv_row_3c_b, csv_row_3c_c

        csv_row_4c_a = [
            'Accuracy', 'Precision', 'Recall', 'f1_score',
            'fpr', 'fnr', 'sen', 'spe', 'g_mean', 'dp',
            'Matthew', 'Cohen', '(random_acc)']  # 13
        csv_row_4c_b = ['g1', 'g0'] * 5 + [
            'abs'] * 5 + ['hat_L(fair)', 'hat_L(loss)']  # 17
        csv_row_4c_c = csv_row_4c_a + csv_row_4c_a + [
            'abs(Acc)', 'abs(P)', 'abs(R)', 'abs(f1)',
            'abs(fpr)', 'abs(fnr)', 'abs(sen)', 'abs(spe)',
            'abs(g_)', 'abs(dp)', 'abs(_M)', 'abs(_C)', ''
        ] + csv_row_4c_b  # 13*2+13+17 =26+30=56
        csv_row_4c = csv_row_4c_c * 6 + ['/ut']  # 56*6+1 =337
        del csv_row_4c_a, csv_row_4c_b, csv_row_4c_c
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    def count_scores(self,
                     y_trn, hx_trn, hq_trn, gones_trn, jt_trn,
                     y_tst, hx_tst, hq_tst, gones_tst, jt_tst,
                     positive_label):
        # NB. self.count_single_member .shape (56,)
        attr_A1 = self.count_single_member(
            y_trn, hx_trn, hq_trn, gones_trn[0], positive_label)
        attr_A2 = self.count_single_member(
            y_tst, hx_tst, hq_tst, gones_tst[0], positive_label)
        if len(jt_trn) == 0:
            # attr_B1 = attr_B2 = [''] * 56
            # attr_J1 = attr_J2 = [''] * 56
            return attr_A1 + attr_A2 + [''] * (56 * 4)

        attr_B1 = self.count_single_member(
            y_trn, hx_trn, hq_trn, gones_trn[1], positive_label)
        attr_B2 = self.count_single_member(
            y_tst, hx_tst, hq_tst, gones_tst[1], positive_label)
        attr_J1 = self.count_single_member(
            y_trn, hx_trn, hq_trn, jt_trn, positive_label)
        attr_J2 = self.count_single_member(
            y_tst, hx_tst, hq_tst, jt_tst, positive_label)
        return (attr_A1 + attr_A2 + attr_B1 + attr_B2 + 
                attr_J1 + attr_J2)  # .shape (336,)= (56*6,)

    def routine_one_sens_attr(self, nb_cls, logger,
                              X_trn, y_trn, Xd_trn, non_sa_trn,
                              X_tst, y_tst, Xd_tst, non_sa_tst,
                              sa_idx, sa_val, positive_label,
                              gones_trn=None, jt_trn=None,
                              gones_tst=None, jt_tst=None):
        res_attr = []

        ut = time.time()
        clf = BaggingClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        fens_trn = clf.predict(X_trn)
        fens_tst = clf.predict(X_tst)
        fqtb_trn = clf.predict(Xd_trn)
        fqtb_tst = clf.predict(Xd_tst)
        res_attr.append(self.count_scores(
            y_trn, fens_trn, fqtb_trn, gones_trn, jt_trn,
            y_tst, fens_tst, fqtb_tst, gones_tst, jt_tst,
            positive_label) + [ut, ])
        del fens_trn, fens_tst, fqtb_trn, fqtb_tst, ut, clf

        ut = time.time()
        clf = AdaBoostClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        fens_trn = clf.predict(X_trn)
        fens_tst = clf.predict(X_tst)
        fqtb_trn = clf.predict(Xd_trn)
        fqtb_tst = clf.predict(Xd_tst)
        res_attr.append(self.count_scores(
            y_trn, fens_trn, fqtb_trn, gones_trn, jt_trn,
            y_tst, fens_tst, fqtb_tst, gones_tst, jt_tst,
            positive_label) + [ut, ])
        del fens_trn, fens_tst, fqtb_trn, fqtb_tst, ut, clf

        ut = time.time()
        # clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
        clf = LGBMClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        fens_trn = clf.predict(X_trn)
        fens_tst = clf.predict(X_tst)
        fqtb_trn = clf.predict(Xd_trn)
        fqtb_tst = clf.predict(Xd_tst)
        res_attr.append(self.count_scores(
            y_trn, fens_trn, fqtb_trn, gones_trn, jt_trn,
            y_tst, fens_tst, fqtb_tst, gones_tst, jt_tst,
            positive_label) + [ut, ])
        del fens_trn, fens_tst, fqtb_trn, fqtb_tst, ut, clf

        for constraint_type in ["FPR", "FNR", "FPR,FNR"]:
            ut = time.time()
            clf = FairGBMClassifier(n_estimators=nb_cls,
                                    constraint_type=constraint_type)
            clf.fit(X_trn, y_trn, constraint_group=~non_sa_trn)
            ut = time.time() - ut
            fens_trn = clf.predict(X_trn)
            fens_tst = clf.predict(X_tst)
            fqtb_trn = clf.predict(Xd_trn)
            fqtb_tst = clf.predict(Xd_tst)
            res_attr.append(self.count_scores(
                y_trn, fens_trn, fqtb_trn, gones_trn, jt_trn,
                y_tst, fens_tst, fqtb_tst, gones_tst, jt_tst,
                positive_label) + [ut, ])
            del fens_trn, fens_tst, fqtb_trn, fqtb_tst, ut, clf

        ut = time.time()
        clf = AdaFair(n_estimators=nb_cls,
                      saIndex=sa_idx, saValue=sa_val)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        fens_trn = clf.predict(X_trn)
        fens_tst = clf.predict(X_tst)
        fqtb_trn = clf.predict(Xd_trn)
        fqtb_tst = clf.predict(Xd_tst)
        res_attr.append(self.count_scores(
            y_trn, fens_trn, fqtb_trn, gones_trn, jt_trn,
            y_tst, fens_tst, fqtb_tst, gones_tst, jt_tst,
            positive_label) + [ut, ])
        del fens_trn, fens_tst, fqtb_trn, fqtb_tst, ut, clf

        return res_attr  # .shape (7,337)= (7,336+1)

    def schedule_content(self, nb_cls, logger,
                         X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                         X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                         saIndex, saValue, positive_label):
        res_iter = []  # y* np.ndarray
        if len(jt_trn) == 0:
            tmp = self.routine_one_sens_attr(
                nb_cls, logger,
                X_trn, y_trn, Xd_trn, gones_trn[0],
                X_tst, y_tst, Xd_tst, gones_tst[0],
                saIndex[0], saValue[0], positive_label,
                gones_trn, jt_trn, gones_tst, jt_tst)
            res_iter.append(tmp)
            return res_iter  # .shape (#attr= 1,7,337)

        sa_len = len(gones_trn)
        for i in range(sa_len):
            tmp = self.routine_one_sens_attr(
                nb_cls, logger,
                X_trn, y_trn, Xd_trn, gones_trn[i],
                X_tst, y_tst, Xd_tst, gones_tst[i],
                saIndex[i], saValue[i], positive_label,
                gones_trn, jt_trn, gones_tst, jt_tst)
            res_iter.append(tmp)
        return res_iter  # .shape (#attr= 2,7,337)

    # def routine_one(self, nb_cls, logger):
    #   clf = BaggingClassifier(n_estimators=nb_cls)


class PartF_FairPruning(PartE_FairPruning):
    def prepare_trial(self):
        csv_row_1 = unique_column(12 + 339)

        csv_row_2c_a = ['A Trn'] + [''] * 38 + ['A Trn'] + [''] * 16
        csv_row_2c_b = ['A Tst'] + [''] * 38 + ['A Tst'] + [''] * 16
        csv_row_2c_c = ['B Trn'] + [''] * 38 + ['B Trn'] + [''] * 16
        csv_row_2c_d = ['B Tst'] + [''] * 38 + ['B Tst'] + [''] * 16
        csv_row_2c_e = ['J Trn'] + [''] * 38 + ['J Trn'] + [''] * 16
        csv_row_2c_f = ['J Tst'] + [''] * 38 + ['J Tst'] + [''] * 16
        csv_row_2c = (csv_row_2c_a + csv_row_2c_b + csv_row_2c_c +
                      csv_row_2c_d + csv_row_2c_e + csv_row_2c_f +
                      ['Time Cost (sec)', '', ''])  # 56*6+3 =339
        del csv_row_2c_a, csv_row_2c_b, csv_row_2c_c
        del csv_row_2c_d, csv_row_2c_e, csv_row_2c_f

        csv_row_3c_a = ['Normal'] + [''] * 12 + [
            'Adversarial'] + [''] * 12 + ['abs'] + [''] * 12
        csv_row_3c_b = ['Fairness'] + [''] * 9 + [
            'Group'] + [''] * 4 + ['Proposed', '']
        csv_row_3c_c = csv_row_3c_a + csv_row_3c_b  # 39+17=56
        # csv_row_3c = csv_row_3c_c * 6 + [''] * 3  # 56*6+3 =339
        csv_row_3c = csv_row_3c_c * 6 + ['ens', 'pru', 'ens+pru']
        del csv_row_3c_a, csv_row_3c_b, csv_row_3c_c

        csv_row_4c_a = ['Accuracy', 'Precision', 'Recall', 'f1_score',
                        'fpr', 'fnr', 'sen', 'spe', 'g_mean', 'dp',
                        'Matthew', 'Cohen', '(random_acc)']  # 13
        csv_row_4c_b = ['g1', 'g0'] * 5 + [
            'abs'] * 5 + ['hat_L(fair)', 'hat_L(loss)']  # 17
        csv_row_4c_c = csv_row_4c_a + csv_row_4c_a + [
            'abs(Acc)', 'abs(P)', 'abs(R)', 'abs(f1)',
            'abs(fpr)', 'abs(fnr)', 'abs(sen)', 'abs(spe)',
            'abs(g_)', 'abs(dp)', 'abs(_M)', 'abs(_C)', ''
        ] + csv_row_4c_b  # 13*2+13+17 =26+30=56
        csv_row_4c = csv_row_4c_c * 6 + ['ut', 'us', 'u+']  # 56*6+3=339
        del csv_row_4c_a, csv_row_4c_b, csv_row_4c_c
        return csv_row_1, csv_row_2c, csv_row_3c, csv_row_4c

    def schedule_content(self, nb_cls, nb_pru, logger,
                         X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                         X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                         saIndex, saValue, positive_label, lam=.25):
        res_iter = []  # lam is the weight of fairness (vs. accuracy)

        tmp = self.routine_each_ensemble(
            "Bagging", "DT", nb_cls, nb_pru, logger,
            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
            X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
            positive_label, lam)
        # res_iter.append(tmp)
        res_iter.extend(tmp)
        del tmp
        tmp = self.routine_each_ensemble(
            "AdaBoostM1", "DT", nb_cls, nb_pru, logger,
            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
            X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
            positive_label, lam)
        res_iter.extend(tmp)
        del tmp
        tmp = self.routine_each_ensemble(
            "SAMME", "DT", nb_cls, nb_pru, logger,
            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
            X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
            positive_label, lam)
        res_iter.extend(tmp)
        del tmp
        # current res_iter.shape (3,1+5,339)

        ut = time.time()
        clf = BaggingClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        tmp = self.subroute_ensemble(
            clf,
            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
            X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
            positive_label)
        res_iter.append(tmp + [ut, 0, ut])
        # res_iter.append([tmp + [ut, 0, ut]])
        del clf, ut, tmp

        ut = time.time()
        clf = AdaBoostClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        tmp = self.subroute_ensemble(
            clf,
            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
            X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
            positive_label)
        res_iter.append(tmp + [ut, 0, ut])
        del clf, ut, tmp

        ut = time.time()
        # clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
        clf = LGBMClassifier(n_estimators=nb_cls)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        tmp = self.subroute_ensemble(
            clf,
            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
            X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
            positive_label)
        res_iter.append(tmp + [ut, 0, ut])
        del clf, ut, tmp
        # current res_iter.shape (21,339)= (6*3+2+1,339)

        if len(jt_trn) == 0:
            tmp = self.routine_one_sens_attr(
                nb_cls, logger,
                X_trn, y_trn, Xd_trn, gones_trn[0],
                X_tst, y_tst, Xd_tst, gones_tst[0],
                saIndex[0], saValue[0], positive_label,
                gones_trn, jt_trn, gones_tst, jt_tst)
            res_iter.extend(tmp)
            return res_iter  # .shape (21+4*1,339)

        sa_len = len(gones_trn)
        for i in range(sa_len):
            tmp = self.routine_one_sens_attr(
                nb_cls, logger,
                X_trn, y_trn, Xd_trn, gones_trn[i],
                X_tst, y_tst, Xd_tst, gones_tst[i],
                saIndex[i], saValue[i], positive_label,
                gones_trn, jt_trn, gones_tst, jt_tst)
            res_iter.extend(tmp)
        return res_iter  # .shape (21+4*2,339)

    def routine_one_sens_attr(self, nb_cls, logger,
                              X_trn, y_trn, Xd_trn, non_sa_trn,
                              X_tst, y_tst, Xd_tst, non_sa_tst,
                              sa_idx, sa_val, positive_label,
                              tag_trn=None, jt_trn=None,
                              tag_tst=None, jt_tst=None):
        res_attr = []

        for constraint_type in ["FPR", "FNR", "FPR,FNR"]:
            ut = time.time()
            clf = FairGBMClassifier(n_estimators=nb_cls,
                                    constraint_type=constraint_type)
            clf.fit(X_trn, y_trn, constraint_group=~non_sa_trn)
            ut = time.time() - ut
            tmp = self.subroute_ensemble(
                clf,
                X_trn, y_trn, Xd_trn, tag_trn, jt_trn,
                X_tst, y_tst, Xd_tst, tag_tst, jt_tst,
                positive_label)
            res_attr.append(tmp + [ut, 0, ut])
            del clf, ut, tmp

        ut = time.time()
        clf = AdaFair(n_estimators=nb_cls,
                      saIndex=sa_idx, saValue=sa_val)
        clf.fit(X_trn, y_trn)
        ut = time.time() - ut
        tmp = self.subroute_ensemble(
            clf,
            X_trn, y_trn, Xd_trn, tag_trn, jt_trn,
            X_tst, y_tst, Xd_tst, tag_tst, jt_tst,
            positive_label)
        res_attr.append(tmp + [ut, 0, ut])
        del clf, ut, tmp

        return res_attr  # .shape (3+1,336+3)

    def subroute_ensemble(self, clf,
                          X_trn, y_trn, Xd_trn, tag_trn, jt_trn,
                          X_tst, y_tst, Xd_tst, tag_tst, jt_tst,
                          positive_label):
        fens_trn = clf.predict(X_trn)
        fens_tst = clf.predict(X_tst)
        fqtb_trn = clf.predict(Xd_trn)
        fqtb_tst = clf.predict(Xd_tst)
        return self.count_scores(
            y_trn, fens_trn, fqtb_trn, tag_trn, jt_trn,
            y_tst, fens_tst, fqtb_tst, tag_tst, jt_tst,
            positive_label)

    def routine_each_ensemble(self, name_ens, abbr_cls,
                              nb_cls, nb_pru, logger,
                              X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                              X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                              positive_label, lam=.5):
        ut = time.time()
        name_cls = INDIVIDUALS[abbr_cls]
        coef, clfs, indices = EnsembleAlgorithm(
            name_ens, name_cls, nb_cls, X_trn, y_trn)
        ut = time.time() - ut

        y_insp = [j.predict(X_trn).tolist() for j in clfs]
        y_pred = [j.predict(X_tst).tolist() for j in clfs]
        yq_insp = [j.predict(Xd_trn).tolist() for j in clfs]
        yq_pred = [j.predict(Xd_tst).tolist() for j in clfs]

        fens_trn = self.majority_vote(y_trn, y_insp, coef)
        fens_tst = self.majority_vote(y_tst, y_pred, coef)
        fqtb_trn = self.majority_vote(y_trn, yq_insp, coef)
        fqtb_tst = self.majority_vote(y_tst, yq_pred, coef)
        fens_trn, fens_tst = np.array(fens_trn), np.array(fens_tst)
        fqtb_trn, fqtb_tst = np.array(fqtb_trn), np.array(fqtb_tst)

        res_ens = []
        tmp = self.count_scores(
            y_trn, fens_trn, fqtb_trn, gones_trn, jt_trn,
            y_tst, fens_tst, fqtb_tst, gones_tst, jt_tst,
            positive_label)
        res_ens.append(tmp + [ut, 0, ut])
        del tmp, fens_trn, fens_tst, fqtb_trn, fqtb_tst

        tmp = self.routine_each_pruning(
            coef, nb_cls, nb_pru, logger,
            y_trn, y_insp, yq_insp, gones_trn, jt_trn,
            y_tst, y_pred, yq_pred, gones_tst, jt_tst,
            positive_label, lam, ut)
        res_ens.extend(tmp)

        return res_ens  # .shape (1+5,339)

    # def subroute_ensemble(self, coef,
    #                       y_trn, y_insp, yq_insp, tag_trn, jt_trn,
    #                       y_tst, y_pred, yq_pred, tag_tst, jt_tst,
    #                       positive_label):
    #   return 

    def routine_each_pruning(self, wgt, nb_cls, nb_pru, logger,
                             y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                             y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                             # positive_label, lam=.5, n_m=2, dist=1,
                             # ut=0):
                             positive_label, lam=.5, ut=0):
        res_pru = []

        us = time.time()
        H = Centralised_EPAF_Pruning(
            y_trn, y_insp, yq_insp, wgt, nb_pru, lam)
        us = time.time() - us
        tmp = self.subroute_pruning(
            H, wgt,
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            positive_label)
        res_pru.append(tmp + [ut, us, ut + us])
        del H, us, tmp

        us = time.time()
        H = Distributed_EPAF_Pruning(
            y_trn, y_insp, yq_insp, wgt, nb_pru, lam, n_m=2)
        us = time.time() - us
        tmp = self.subroute_pruning(
            H, wgt,
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            positive_label)
        res_pru.append(tmp + [ut, us, ut + us])
        del H, us, tmp

        us = time.time()
        H = Distributed_EPAF_Pruning(
            y_trn, y_insp, yq_insp, wgt, nb_pru, lam, n_m=3)
        us = time.time() - us
        tmp = self.subroute_pruning(
            H, wgt,
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            positive_label)
        res_pru.append(tmp + [ut, us, ut + us])
        del H, us, tmp

        us = time.time()
        H = Pareto_Optimal_EPAF_Pruning(
            y_trn, y_insp, yq_insp, wgt, nb_pru, lam, dist=1)
        us = time.time() - us
        tmp = self.subroute_pruning(
            H, wgt,
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            positive_label)
        res_pru.append(tmp + [ut, us, ut + us])
        del H, us, tmp

        us = time.time()
        H = POAF_PEP(
            y_trn, y_insp, yq_insp, wgt, 1. - lam, nb_pru)
        us = time.time() - us
        tmp = self.subroute_pruning(
            H, wgt,
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            positive_label)
        res_pru.append(tmp + [ut, us, ut + us])
        del H, us, tmp

        return res_pru  # .shape (5,339)= (1+2+2,56*6+3)

    def subroute_pruning(self, H, wgt,
                         y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                         y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                         positive_label):
        ys_insp = np.array(y_insp)[H].tolist()
        ys_pred = np.array(y_pred)[H].tolist()
        yr_insp = np.array(yq_insp)[H].tolist()
        yr_pred = np.array(yq_pred)[H].tolist()
        coef = np.array(wgt)[H].tolist()

        fens_trn = self.majority_vote(y_trn, ys_insp, coef)
        fens_tst = self.majority_vote(y_tst, ys_pred, coef)
        fqtb_trn = self.majority_vote(y_trn, yr_insp, coef)
        fqtb_tst = self.majority_vote(y_tst, yr_pred, coef)
        fens_trn, fens_tst = np.array(fens_trn), np.array(fens_tst)
        fqtb_trn, fqtb_tst = np.array(fqtb_trn), np.array(fqtb_tst)

        return self.count_scores(
            y_trn, fens_trn, fqtb_trn, tag_trn, jt_trn,
            y_tst, fens_tst, fqtb_tst, tag_tst, jt_tst,
            positive_label)  # .shape (56*6,)


# -------------------------------------
# Experimental design
#


# class ExperimentSetup(DataSetup):  # that is, ExptSetting()
class FairVoteEmpirical(DataSetup):
    def __init__(self, trial_type, data_type,
                 # name_ens, abbr_cls, nb_cls, nb_pru,
                 # nb_iter=5, ratio=.5, screen=True, logged=False):
                 nb_cls, nb_pru, nb_iter=5, ratio=.5, lam=.5,
                 name_ens='bagging', abbr_cls='DT',
                 screen=True, logged=False):
        super().__init__(data_type)
        self._trial_type = trial_type

        # nmens_temp = _get_tmp_name_ens(name_ens)
        # self._log_document = "_".join([
        #     trial_type, nmens_temp, abbr_cls + str(nb_cls),
        #     self._log_document])  # aka. data_type
        self._log_document = "_".join([
            trial_type, "{}vs{}".format(nb_cls, nb_pru),
            "iter{}".format(nb_iter), self._log_document,
            "ratio{}".format(int(ratio * 100)), "pms"])

        self._nb_cls = nb_cls
        self._nb_pru = nb_pru
        self._nb_iter = nb_iter
        self._ratio = ratio
        self._lam = lam

        self._screen = screen
        self._logged = logged
        # self._log_document += '_iter{}'.format(nb_iter)
        # self._log_document += '_pms'
        # self._iterator = EnsembleSetup(
        #     name_ens, abbr_cls, nb_cls, nb_pru)

        self._name_ens = name_ens
        self._abbr_cls = abbr_cls
        if trial_type.endswith('expt3'):
            nmens_tmp = _get_tmp_name_ens(name_ens)
            self._log_document = self._log_document.replace(
                data_type, '')
            self._log_document += "_{}_{}_{}".format(
                nmens_tmp, abbr_cls, data_type)
        if trial_type.endswith('expt6'):
            self._log_document += "_lam{}".format(int(lam * 100))

        if trial_type.endswith('expt1'):
            self._iterator = PartA_FairMeasure()
        elif trial_type.endswith('expt4'):
            self._iterator = PartD_FairMeasure()
        elif trial_type.endswith('expt2'):
            self._iterator = PartB_FairMeasure()
        elif trial_type.endswith('expt3'):
            self._iterator = PartC_FairMeasure(name_ens, abbr_cls)
            # self._iterator = PartB_FairTheorems()
        elif trial_type.endswith('expt5'):
            self._iterator = PartE_FairPruning()
        elif trial_type.endswith('expt6'):
            self._iterator = PartF_FairPruning()

    # def __del__(self):
    #     pass
    # @property
    # def trial_type(self):
    #     return self._trial_type

    @property
    def nb_iter(self):
        return self._nb_iter

    # @property
    # def ratio(self):
    #     return self._ratio
    #
    # @property
    # def iterator(self):
    #     return self._iterator
    #
    # @iterator.setter
    # def iterator(self, value):
    #     self._iterator = value

    def trial_one_process(self):
        since = time.time()
        csv_t = open(self._log_document + '.csv', "w")
        csv_w = csv.writer(csv_t)

        if (not self._screen) and (not self._logged):
            saveout = sys.stdout
            fsock = open(self._log_document + '.log', "w")
            sys.stdout = fsock
        if self._logged:
            '''
            logger = logging.getLogger("oracle_fair")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s:%(levelname)s | %(message)s')
            if os.path.exists(self._log_document + ".txt"):
                os.remove(self._log_document + ".txt")
            log_file = logging.FileHandler(self._log_document + '.txt')
            log_file.setLevel(logging.DEBUG)
            log_file.setFormatter(formatter)
            logger.addHandler(log_file)
            '''
            if os.path.exists(self._log_document + ".txt"):
                os.remove(self._log_document + ".txt")
            logger, formatter, fileHandler = get_elogger(
                "fairvote", self._log_document + ".txt")
        else:
            logger = None

        '''
        elegant_print("[BEGAN AT {:s}]".format(time.strftime(
            "%d-%b-%Y %H:%M:%S", time.localtime(since)
        )), logger)
        elegant_print(" EXPERIMENT  :", logger)
        elegant_print([
            "\t trial    = {}".format(self._trial_type),
            "\t dataset  = {}".format(self._data_type),
            "\t binary?  = {}".format(
                str(not self._trial_type.startswith('mu')))], logger)
        elegant_print(" PARAMETERS  :", logger)
        elegant_print([
            "\t name_ens = {}".format(self._iterator.name_ens),
            "\t abbr_cls = {}".format(self._iterator.abbr_cls),
            "\t   nb_cls = {}".format(self._iterator.nb_cls),
            "\t   nb_pru = {}".format(self._iterator.nb_pru),
            "\t  nb_iter = {}".format(self._nb_iter)], logger)
        elegant_print(" HYPER-PARAMS:", logger)
        elegant_print("", logger)
        '''

        elegant_print([
            "[BEGAN AT {}]".format(elegant_dated(since)),
            " EXPERIMENT",
            "\t   trial = {}".format(self._trial_type),
            "\t dataset = {}".format(self._data_type),
            "\t binary? = {}".format(
                not self._trial_type.startswith('mu')),
            " PARAMETERS",
            "\t  nb_cls = {}".format(self._nb_cls),
            "\t  nb_pru = {}".format(self._nb_pru),
            "\t nb_iter = {}".format(self._nb_iter),
            "\t   ratio = {}".format(self._ratio),
            " HYPER-PARAMS", ""], logger)
        # self.trial_one_dataset(logger)

        # START
        '''
        csv_row_2a = ['data_name', 'binary',
                      'name_ens', 'abbr_cls', 'nb_cls', 'nb_pru',
                      'nb_iter', 'iteration']
        csv_row_2b = ['Ensemble'] + [''] * 12
        csv_row_3b = ['', 'accuracy',
                      'Acc', 'P', 'R', 'F1*', 'F1', 'F2', 'F3',
                      'TPR', 'FPR', 'FNR', 'TNR']
        csv_row_1, csv_row_2c, csv_row_3c = self._iterator.prepare_trial()
        csv_row_2 = csv_row_2a + csv_row_2b + csv_row_2c
        csv_row_3 = [''] * 8 + csv_row_3b + csv_row_3c
        csv_w.writerows([csv_row_1, csv_row_2, csv_row_3])
        del csv_row_1, csv_row_2, csv_row_3
        del csv_row_2a, csv_row_2b, csv_row_3b
        del csv_row_2c, csv_row_3c
        res_all = self.trial_one_dataset(logger=logger)
        # csv_w.writerows(res_all)
        csv_w.writerow(res_all[0])
        for res_iter in res_all[1:]:
            for i_ens in res_iter:
                csv_w.writerow([''] * 7 + i_ens)
        del res_all
        '''

        csv_row_2a = ['data_name', 'binary',
                      'nb_cls', 'nb_pru', 'nb_iter', 'iteration']
        csv_row_2b = ["#sens_attr", "name_ens", "abbr_cls",
                      "name_pru", "#iter", "#eval"]
        csv_row_1, csv_r2c, csv_r3c, csv_r4c = self._iterator.prepare_trial()
        csv_row_2 = csv_row_2a + csv_row_2b + csv_r2c
        csv_w.writerows([csv_row_1, csv_row_2])
        csv_w.writerow([''] * 12 + csv_r3c)
        csv_w.writerow([''] * 12 + csv_r4c)
        del csv_row_2a, csv_row_2b, csv_row_2
        del csv_row_1, csv_r2c, csv_r3c, csv_r4c

        res_data, res_all = self.trial_one_dataset(logger)
        json_saver = json.dumps({
            "res_all": res_all, "res_data": res_data})
        json_w = open(self._log_document + ".json", "w")
        json_w.write(json_saver)
        json_w.close()
        del json_saver, json_w

        if self._trial_type[-5:] in ['expt1', 'expt4', 'expt5']:
            # res_data.shape: (#iter=5, #att=1/2, #alg/baseline=5, #criteria)
            res_data = np.array(res_data).transpose(1, 2, 0, 3)
            for sa, sens_attr in enumerate(res_all[1]):
                csv_w.writerow(res_all[0] + [sens_attr])
                for en, name_ens in enumerate(res_all[-1]):
                    tmp_1 = np.array([[name_ens] + [''] * (self._nb_iter - 1),
                                      [''] * self._nb_iter,
                                      [''] * self._nb_iter,
                                      res_all[2],
                                      [''] * self._nb_iter]).T
                    tmp_2 = np.array([[''] * self._nb_iter] * 7).T
                    curr = np.c_[tmp_2, tmp_1, res_data[sa, en]]
                    csv_w.writerows(curr)
        elif self._trial_type.endswith('expt6'):
            # res_data.shape (#iter=5, 6*3+2+1+4*?, 339)
            #            --> (6*3+2+1+4*?, #iter=5, 339)
            res_data = np.array(res_data).transpose(1, 0, 2)
            csv_w.writerow(res_all[0])
            tmp_2 = np.array([[''] * self._nb_iter] * 7).T
            for e_i, e_n in enumerate(res_all[-4]):
                s_a, s_b = 6 * e_i, 6 * (e_i + 1)
                tmp_1 = np.array([
                    [e_n] + [''] * (self._nb_iter - 1),
                    ['DT'] + [''] * (self._nb_iter - 1),
                    ['Entire'] + [''] * (self._nb_iter - 1),
                    res_all[2],
                    [''] * self._nb_iter]).T
                curr = np.c_[tmp_2, tmp_1, res_data[s_a]]
                csv_w.writerows(curr)
                del tmp_1, curr
                s_a += 1
                for p_i, p_n in enumerate(res_all[-3][1:]):
                    tmp_1 = np.array([
                        [''] * self._nb_iter,
                        [''] * self._nb_iter,
                        [p_n] + [''] * (self._nb_iter - 1),
                        res_all[2],
                        [''] * self._nb_iter]).T
                    curr = np.c_[tmp_2, tmp_1, res_data[s_a + p_i]]
                    csv_w.writerows(curr)
                    del tmp_1, curr
                del s_a, s_b
            for e_i, e_n in enumerate(res_all[-2]):
                tmp_1 = np.array([
                    [e_n] + [''] * (self._nb_iter - 1),
                    [''] * self._nb_iter,
                    [''] * self._nb_iter,
                    res_all[2],
                    [''] * self._nb_iter]).T
                curr = np.c_[tmp_2, tmp_1, res_data[18 + e_i]]
                csv_w.writerows(curr)
                del tmp_1, curr
            del tmp_2
            tmp_2 = np.array([[''] * self._nb_iter] * 6).T
            for sa, sens_attr in enumerate(res_all[1]):
                s_b = 21 + 4 * sa
                for e_i, e_n in enumerate(res_all[-1]):
                    tmp_1 = np.array([
                        [sens_attr] + [''] * (self._nb_iter - 1),
                        [e_n] + [''] * (self._nb_iter - 1),
                        [''] * self._nb_iter,
                        [''] * self._nb_iter,
                        res_all[2],
                        [''] * self._nb_iter]).T
                    curr = np.c_[tmp_2, tmp_1, res_data[s_b + e_i]]
                    csv_w.writerows(curr)
                    del tmp_1, curr
                del s_b
            del tmp_2
        elif self._trial_type[-5:] in ['expt2', 'expt3']:
            # res_data.shape: (#iter=5, #ens=13-1, #attr=1/3, 1+1+#pru=8, 63)
            res_data = np.array(
                res_data, dtype=object).transpose(2, 1, 3, 0, 4)
            for sa, sens_attr in enumerate(res_all[1]):
                for j, (name_ens, abbr_cls) in enumerate(zip(res_all[-3], res_all[-2])):
                    if name_ens == "AdaBoost":
                        continue
                    csv_w.writerow(
                        # res_all[0] + [sens_attr, "{} ({})".format(name_ens, abbr_cls)]
                        res_all[0] + [sens_attr, name_ens, abbr_cls])
                    for pr, name_pru in enumerate(res_all[-1]):
                        tmp_1 = np.array([
                            # ['{} .{}'.format(name_ens, abbr_cls)] + [''] * (self._nb_iter - 1),
                            [name_ens] + [''] * (self._nb_iter - 1),
                            [abbr_cls] + [''] * (self._nb_iter - 1),
                            # ['rank .{}'.format(i) for i in CRITERIA],
                            # ['rank .{}'.format(name_pru)] + [''] * (self._nb_iter - 1),
                            [name_pru] + [''] * (self._nb_iter - 1),
                            res_all[2], [''] * self._nb_iter]).T
                        tmp_2 = np.array([[''] * self._nb_iter] * 7).T
                        curr = np.c_[tmp_2, tmp_1, res_data[sa, j, pr]]
                        csv_w.writerows(curr)

        # END
        '''
        tim_elapsed = time.time() - since
        elegant_print(" Duration: {:.0f} min {:.2f} sec".format(
            tim_elapsed // 60, tim_elapsed % 60), logger)
        tim_elapsed /= 60
        elegant_print(" Duration: {:.0f} hrs {:.2f} min".format(
            tim_elapsed // 60, tim_elapsed % 60), logger)
        elegant_print(" Time Cost in total: {:.10f} hour(s)."
                      "".format(tim_elapsed / 60), logger)
        since = time.time()
        elegant_print("[ENDED AT {:s}]".format(time.strftime(
            "%d-%b-%Y %H:%M:%S", time.localtime(since)
        )), logger)  # min, sec, minute(s), hrs, min, hour(s)

        if self._logged:
            logger.removeHandler(log_file)
            del log_file, formatter
        '''
        tim_elapsed = time.time() - since
        elegant_print([
            "",
            "Duration /Time Cost: {}".format(elegant_durat(
                tim_elapsed)),
            "[ENDED AT {:s}]".format(elegant_dated(time.time()))
        ], logger)
        if self._logged:
            rm_ehandler(logger, formatter, fileHandler)
        del logger
        if not self._screen and not self._logged:
            fsock.close()
            sys.stdout = saveout
        csv_t.close()
        del csv_t, csv_w, since, tim_elapsed
        return

    def trial_one_dataset(self, logger):
        processed_data = preprocess(
            self._dataset, self._data_frame, logger)
        disturbed_data = adversarial(
            self._dataset, self._data_frame, self._ratio, logger)

        processed_Xy = processed_data['numerical-binsensitive']
        disturbed_Xy = disturbed_data['numerical-binsensitive']
        X, y = transform_X_and_y(self._dataset, processed_Xy)
        Xp, _ = transform_X_and_y(self._dataset, disturbed_Xy)
        belongs_priv, ptb_with_joint = transform_unpriv_tag(
            self._dataset, processed_data['original'])
        # Note that PTB: place to belong
        # X, Xp, y: pd.DataFrame

        tmp = processed_data['original'][self._dataset.label_name]
        elegant_print("\t BINARY? Y= {}".format(set(y)), logger)
        elegant_print("\t  NB. i.e., {}".format(set(tmp)), logger)
        del tmp

        # {tr/bi/mu}_{KF,KFS,mCV}_{trial_part_*}
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
            elegant_print("\t CrossValid  {}".format('mCV'), logger)
        elif 'KFS' in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y, X)
            elegant_print("\t CrossValid  {}".format('KFS'), logger)
        elif 'KF' in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
            elegant_print("\t CrossValid  {}".format('KF'), logger)
        else:
            raise ValueError("No proper CV (cross-validation).")

        # START
        # res_all = []
        # res_all.append([
        #     self._dataset.dataset_name, len(set(y)),
        #     self._iterator.name_ens, self._iterator.abbr_cls,
        #     self._iterator.nb_cls, self._iterator.nb_pru,
        #     self._nb_iter])
        res_all = [[self._dataset.dataset_name, len(set(y)),
                    self._nb_cls, self._nb_pru, self._nb_iter, ''],
                   self._dataset.sensitive_attrs,
                   list(range(self._nb_iter))]
        if self._trial_type[-5:] in ['expt1', 'expt4', 'expt5']:
            res_all.append(['bagging', 'AdaBoost',  # SAMME
                            'LightGBM', 'FairGBM (FPR)', 'FairGBM (FNR)',
                            'FairGBM (FPR,FNR)', 'AdaFair'])
        elif self._trial_type.endswith('expt6'):
            res_all.append(["Bagging", "AdaBoostM1", "SAMME"])
            res_all.append(["Entire",
                            "EPAF-C", "EPAF-D:2", "EPAF-D:3",
                            "POEPAF", "POPEP"])
            res_all.append(["bagging", "adaboost", "LightGBM"])
            res_all.append(["FairGBM (fpr)", "FairGBM (fnr)",
                            "FairGBM (fpr,fnr)", "AdaFair"])
        elif self._trial_type.endswith('expt2'):
            # # res_all.append(['bagging'] + [''] * (12 - 1) + ['AdaBoost'])
            res_all.append(['bagging'] * 11 + ['AdaBoost'])
            res_all.append(ALG_NAMES + ['DT'])
            res_all.append(['Ensem'] * 2 + [
                'rank.' + i for i in CRITERIA])
        elif self._trial_type.endswith('expt3'):
            res_all.append([self._name_ens])
            res_all.append([self._abbr_cls])
            # res_all.append(CRITERIA)
            res_all.append(['Ensem'] + ['rank.' + i for i in CRITERIA])
        res_data = []

        for k, (i_trn, i_tst) in enumerate(split_idx):
            # X_trn = X.iloc[i_trn]
            # y_trn = y.iloc[i_trn]
            # Xp_trn = Xp.iloc[i_trn]
            # X_tst = X.iloc[i_tst]
            # y_tst = y.iloc[i_tst]
            # Xp_tst = Xp.iloc[i_tst]

            X_trn, Xd_trn, y_trn, gones_trn, jt_trn = transform_perturbed(
                X, Xp, y, i_trn, belongs_priv, ptb_with_joint)
            X_tst, Xd_tst, y_tst, gones_tst, jt_tst = transform_perturbed(
                X, Xp, y, i_tst, belongs_priv, ptb_with_joint)
            # X_*, Xd_*, y_*: pd.DataFrame
            # gones_*, jt_* : list of np.ndarray (element)

            X_trn = X_trn.to_numpy()   # .tolist()
            X_tst = X_tst.to_numpy()   # .tolist()
            y_trn = y_trn.to_numpy().reshape(-1)  # .tolist()
            y_tst = y_tst.to_numpy().reshape(-1)  # .tolist()
            Xd_trn = Xd_trn.to_numpy()  # .tolist()
            Xd_tst = Xd_tst.to_numpy()  # .tolist()

            # i-th K-Fold
            elegant_print("Iteration {}-th".format(k + 1), logger)
            res_iter = self.trial_one_iteration(
                logger, k,
                X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                X_tst, y_tst, Xd_tst, gones_tst, jt_tst)
            res_data.append(res_iter)
            del X_trn, Xd_trn, y_trn, gones_trn, jt_trn
            del X_tst, Xd_tst, y_tst, gones_tst, jt_tst
        # # res_data.shape: (#iter=5, #attr=1/2, #alg/baseline, #criteria)
        # res_data = np.array(res_data, dtype=object).transpose(1, 2, 0, 3)
        del split_idx  # return res_all
        return res_data, res_all

    def trial_one_iteration(self, logger, k,
                            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                            X_tst, y_tst, Xd_tst, gones_tst, jt_tst):
        since = time.time()
        positive_label = \
            self._dataset.get_positive_class_val('numerical-binsensitive')

        '''
        y_insp, _, y_pred, _ = \
            self._iterator.achieve_ensemble_from_train_set(
                X_trn, y_trn, [], X_tst)  # logger,
        yd_insp = [j.predict(Xd_trn) for j in self._iterator.member]
        yd_pred = [j.predict(Xd_tst) for j in self._iterator.member]
        # y_*, yd_* : list of np.ndarray (element)
        fens_trn = self._iterator.majority_vote(y_trn, y_insp)
        fens_tst = self._iterator.majority_vote(y_tst, y_pred)
        fd_E_trn = self._iterator.majority_vote(y_trn, yd_insp)
        fd_E_tst = self._iterator.majority_vote(y_tst, yd_pred)
        # fens_*, fd_E_*: list

        res_ensem = []
        Acc, temp = self._iterator.calculate_sub_ensemble_metrics(
            y_trn, fens_trn, self._dataset.positive_label)
        res_ensem.append([k, 'X_trn', Acc] + temp)
        Acc, temp = self._iterator.calculate_sub_ensemble_metrics(
            y_tst, fens_tst, self._dataset.positive_label)
        res_ensem.append([k, 'X_tst', Acc] + temp)
        Acc, temp = self._iterator.calculate_sub_ensemble_metrics(
            y_trn, fd_E_trn, self._dataset.positive_label)
        res_ensem.append([k, 'Xd_trn', Acc] + temp)
        Acc, temp = self._iterator.calculate_sub_ensemble_metrics(
            y_tst, fd_E_tst, self._dataset.positive_label)
        res_ensem.append([k, 'Xd_tst', Acc] + temp)

        tim_elapsed = time.time() - since
        elegant_print("\tEnsem: time cost {:.2f} seconds"
                      "".format(tim_elapsed), logger)
        # self._iterator.schedule_content()
        return res_ensem
        '''

        if self._trial_type.endswith('expt1'):
            res_iter = self._iterator.schedule_content(
                self._nb_cls, logger,
                X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                self.saIndex, self.saValue, positive_label)
            # res_iter.shape (#attr= 1/2, 5, 61)
        elif self._trial_type.endswith('expt4'):
            res_iter = self._iterator.schedule_content(
                self._nb_cls, logger,
                X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                self.saIndex, self.saValue, positive_label)
            # res_iter.shape (#attr= 1/2, 7, 87)
            # res_iter.shape (#attr= 1/2, 7, 113)
        elif self._trial_type.endswith('expt5'):
            res_iter = self._iterator.schedule_content(
                self._nb_cls, logger,
                X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                self.saIndex, self.saValue, positive_label)
            # res_iter.shape (#attr= 1/2, 7, 337)
        elif self._trial_type.endswith('expt6'):
            res_iter = self._iterator.schedule_content(
                self._nb_cls, self._nb_pru, logger,
                X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                self.saIndex, self.saValue, positive_label, self._lam)
            # res_iter.shape (21+4*?, 339) where ?=1|2
            # ie. (6*3+2+1+(3+1)*#attr, 56, 339)
        elif self._trial_type.endswith('expt2'):
            res_iter = self._iterator.schedule_content(
                self._nb_cls, self._nb_pru, logger,
                X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                positive_label, self._lam)
            # res_iter.shape (11, #attr=1/3, 8, 63)
            # # res_iter.shape (1+11+1, #attr=1/3, 8, 63)
        elif self._trial_type.endswith('expt3'):
            res_iter = self._iterator.schedule_content(
                self._nb_cls, self._nb_pru, logger,
                X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                X_tst, y_tst, Xd_tst, gones_tst, jt_tst,
                positive_label, self._lam)
            # res_iter.shape (1, #attr=1/3, 7, 63)

        tim_elapsed = time.time() - since
        elegant_print(["Iteration {}, Consumed {}".format(
            k + 1, elegant_durat(tim_elapsed)), ], logger)
        return res_iter

    # def routine_fair_ensem(self, logger, k,
    #                        X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
    #                        X_tst, y_tst, Xd_tst, gones_tst, jt_tst):
    #   pass
    #   # self._iterator = ClassifierSetup()
