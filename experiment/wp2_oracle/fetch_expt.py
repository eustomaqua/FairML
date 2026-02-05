# coding: utf-8
# fetch_data.py


from pyfair.facil.utils_saver import elegant_print
from pyfair.facil.utils_remark import (
    AVAILABLE_NAME_PRUNE, LATEST_NAME_PRUNE)
# Experiments
from pyfair.facil.utils_const import _get_tmp_name_ens

from fairml.datasets import preprocess
from fairml.preprocessing import (
    adversarial, transform_X_and_y, transform_unpriv_tag,
    transform_perturbed)

from pyfair.marble.data_classify import EnsembleAlgorithm
# from experiment.utils.utils_learner import FAIR_INDIVIDUALS as INDIVIDUALS
from experiment.widget.utils_learner import FAIR_INDIVIDUALS as INDIVIDUALS

from experiment.wp2_oracle.fetch_data import (
    calc_accuracy, calc_Acc, calc_PR, calc_F1, calc_4Rate,
    calc_confusion, DataSetup)

from pyfair.granite.ensem_pruning import \
    contrastive_pruning_methods as exist_pruning_basics
from pyfair.granite.ensem_prulatest import \
    contrastive_pruning_lately as exist_pruning_latest
from fairml.discriminative_risk import (
    hat_L_fair, E_rho_L_fair_f, Erho_sup_L_fair,
    hat_L_loss, E_rho_L_loss_f)
from fairml.dr_pareto_optimal import (
    Pareto_Optimal_EPAF_Pruning, _bi_objectives, POAF_PEP,
    Centralised_EPAF_Pruning, Distributed_EPAF_Pruning)

# from pyfair.facil.utils_const import unique_column  # check_zero,
# from pyfair.facil.utils_saver import elegant_print
from pyfair.marble.metric_fair import (
    marginalised_pd_mat, prev_unpriv_grp_one, prev_unpriv_grp_two,
    prev_unpriv_grp_thr, prev_unpriv_unaware, prev_unpriv_manual)

from pyfair.facil.data_split import (
    sklearn_k_fold_cv, sklearn_stratify, manual_cross_valid)
# Experiments

import numpy as np
import csv
import logging
import os
import sys
import time

unpriv_group_one = prev_unpriv_grp_one
unpriv_group_two = prev_unpriv_grp_two
unpriv_group_thr = prev_unpriv_grp_thr
unpriv_unaware = prev_unpriv_unaware
unpriv_manual = prev_unpriv_manual

del prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr
del prev_unpriv_unaware, prev_unpriv_manual


# ==================================
# Experimental design
# ==================================


# Classifiers
# ----------------------------------


class IndividualSetup:
    def __init__(self, abbr_cls):
        self._abbr_cls = abbr_cls
        self._member = INDIVIDUALS[abbr_cls]

    @property
    def abbr_cls(self):
        return self._abbr_cls

    @property
    def member(self):
        return self._member

    @member.setter
    def member(self, value):
        self._member = value

    def calculate_fair_measure_in_groups(self, y, hx, pos=1,
                                         # ind_priv=list()):
                                         ind_priv=tuple()):
        if not isinstance(ind_priv, list):
            ind_priv = list(ind_priv)
        # pos : dataset.positive_label
        # priv: dataset.get_positive_class_val('numerical-binsensitive')
        # sens: an element in `belongs_priv` or `belongs_priv_with_joint`

        # # y : pd.DataFrame  --> `.to_numpy()`
        # # hx: np.ndarray    --> `.tolist()`
        # y,hx,ind_priv: np.ndarray  # indicator_of_priv

        _, _, gones_Cm, gzero_Cm = marginalised_pd_mat(
            y, hx, pos, ind_priv)  # g1_Cij, g0_Cij,
        fair_measure = {
            'unaware': unpriv_unaware(gones_Cm, gzero_Cm),
            'group_1': unpriv_group_one(gones_Cm, gzero_Cm),
            'group_2': unpriv_group_two(gones_Cm, gzero_Cm),
            'group_3': unpriv_group_thr(gones_Cm, gzero_Cm),
            'accuracy': unpriv_manual(gones_Cm, gzero_Cm),
        }

        # gones: privileged group
        tmp_y = y[ind_priv].tolist()
        tmp_hx = hx[ind_priv].tolist()
        tp, fp, fn, tn = calc_confusion(tmp_y, tmp_hx, pos=pos)
        gones_ans = [calc_accuracy(tmp_y, tmp_hx),
                     calc_Acc(tp, fp, fn, tn)[0]]   # acc x2
        p, r, f1 = calc_PR(tp, fp, fn)
        gones_ans.extend([p, r, f1])  # P,R,F1
        gones_ans.append(calc_F1(p, r, beta=1))
        gones_ans.append(calc_F1(p, r, beta=2))     # F2
        gones_ans.extend(calc_4Rate(tp, fp, fn, tn))  # ratesx4

        # gzero: unprivileged groups
        ind_priv = np.logical_not(ind_priv)
        tmp_y = y[ind_priv].tolist()
        tmp_hx = hx[ind_priv].tolist()
        tp, fp, fn, tn = calc_confusion(tmp_y, tmp_hx, pos=pos)
        gzero_ans = [calc_accuracy(tmp_y, tmp_hx),
                     calc_Acc(tp, fp, fn, tn)[0]]   # acc x2
        p, r, f1 = calc_PR(tp, fp, fn)
        gzero_ans.extend([p, r, f1])  # P,R,F1
        gzero_ans.append(calc_F1(p, r, beta=1))
        gzero_ans.append(calc_F1(p, r, beta=2))     # F2
        gzero_ans.extend(calc_4Rate(tp, fp, fn, tn))  # rates x4

        return fair_measure, gones_ans, gzero_ans


class EnsembleSetup(IndividualSetup):
    def __init__(self, name_ens, abbr_cls, nb_cls, nb_pru=None):
        super().__init__(abbr_cls)
        self._name_ens = name_ens
        self._nb_cls = nb_cls
        self._nb_pru = nb_pru
        if nb_pru is None:
            self._nb_pru = nb_cls
        self._weight = list()

    @property
    def name_ens(self):
        return self._name_ens

    @property
    def nb_cls(self):
        return self._nb_cls

    @property
    def nb_pru(self):
        return self._nb_pru

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value=None):
        self._weight = value

    def achieve_ensemble_from_train_set(self,
                                        X_trn, y_trn,
                                        X_val, X_tst):
        # since = time.time()
        name_cls = INDIVIDUALS[self._abbr_cls]
        coef, clfs, indices = EnsembleAlgorithm(
            self._name_ens, name_cls, self._nb_cls, X_trn, y_trn)

        y_insp = [j.predict(X_trn) for j in clfs]
        y_pred = [j.predict(X_tst) for j in clfs]
        y_cast = [j.predict(X_val) for j in clfs] if X_val else []
        self._weight = coef
        self._member = clfs

        return y_insp, y_cast, y_pred, indices

    def majority_vote(self, y, yt, wgt=None):
        # that is, majority_vote_subscript_rho()
        #     aka. weighted_voting

        # y = y.to_numpy()
        # y : np.ndarray or list
        # yt: list of np.ndarray

        if isinstance(yt[0], np.ndarray):
            yt = [i.tolist() for i in yt]
        vY = np.unique(np.concatenate([[y], yt]))
        wgt = self._weight if not wgt else wgt
        coef = np.array([wgt]).transpose()
        weig = [np.sum(coef * np.equal(
            yt, i), axis=0).tolist() for i in vY]
        loca = np.array(weig).argmax(axis=0).tolist()
        fens = [vY[i] for i in loca]
        return fens

    def pruning_baseline(
            self, y, yt, name_pru, epsilon=1e-6, rho=.4,
            alpha=0.5, L=3, R=2, X=None, indices=None):
        # rho = float(self._nb_pru) / self._nb_cls  # default: 0.4

        if name_pru in AVAILABLE_NAME_PRUNE + ['KP']:
            ys, P, seq, _ = exist_pruning_basics(
                name_pru, self._nb_cls, self._nb_pru, y, yt,
                epsilon, rho)

        elif name_pru in LATEST_NAME_PRUNE:
            kwargs = {}
            if name_pru.startswith("TSP"):
                kwargs["indices"] = indices
            elif name_pru.startswith("TSPrev"):
                kwargs["X_trn"] = X
                kwargs["X_val"] = X

            ys, _, P, seq = exist_pruning_latest(
                name_pru, self._nb_cls, self._nb_pru,
                y, [], yt, [], alpha, L, R, **kwargs)

        else:
            raise ValueError("No such pruning named `{}`".format(
                name_pru))
        return ys, P, seq

    def pruning_proposed(self, y, yt, yq, name_pru, lam,
                         dist=1, n_m=2):
        if name_pru == 'POEPAF':
            H = Pareto_Optimal_EPAF_Pruning(
                y, yt, yq, self._weight, self._nb_pru, lam, dist)
        elif name_pru == 'EPAF-C':
            H = Centralised_EPAF_Pruning(
                y, yt, yq, self._weight, self._nb_pru, lam)
        elif name_pru == 'EPAF-D':
            H = Distributed_EPAF_Pruning(
                y, yt, yq, self._weight, self._nb_pru, lam, n_m)
        elif name_pru == 'POPEP':
            H = POAF_PEP(y, yt, yq, self._weight, lam,
                         self._nb_pru)
        else:
            raise ValueError(
                "No such pruning proposed `{}`".format(
                    name_pru))
        return H

    def calculate_sub_ensemble_metrics(self, y, hx, pos=1):
        # calculate ensemble or sub's metrics
        Acc = calc_accuracy(y, hx)

        tp, fp, fn, tn = calc_confusion(y, hx, pos=pos)
        A, _ = calc_Acc(tp, fp, fn, tn)
        P, R, F = calc_PR(tp, fp, fn)

        F1 = calc_F1(P, R)
        F2 = calc_F1(P, R, beta=2)
        F3 = calc_F1(P, R, beta=3)

        TPR, FPR, FNR, TNR = calc_4Rate(tp, fp, fn, tn)
        return Acc, [A, P, R, F, F1, F2, F3, TPR, FPR, FNR, TNR]

    def calculate_fair_quality_pruned(self, y, ys, yr, wgt,
                                      pos=1, lam=.5):
        fens = self.majority_vote(y, ys, wgt)
        fqtb = self.majority_vote(y, yr, wgt)

        tmp_pru = []
        # Acc, (a, p, r, f, _, _, _, _, _, _, _) = \
        Acc, (_, p, r, f1, _, _, _, _, _, _, _) = \
            self.calculate_sub_ensemble_metrics(y, fens, pos)
        tmp_pru.extend([Acc, p, r, f1])

        # L_fair(MV_rho), L_acc(MV_rho)
        sub_whole = (hat_L_fair(fens, fqtb), hat_L_loss(fens, y))
        tmp_pru.extend(sub_whole)
        tmp_pru.append(_bi_objectives(sub_whole, lam))

        # objective
        sub_split = (Erho_sup_L_fair(ys, yr, wgt),
                     E_rho_L_loss_f(ys, y, wgt))
        tmp_pru.extend(sub_split)
        tmp_pru.append(_bi_objectives(sub_split, lam))
        tmp_pru.append(E_rho_L_fair_f(ys, yr, wgt))

        return tmp_pru  # 4+(2+1)+(2+1)+1 =4+3*2+1 =11

    def prepare_trial(self):
        return [], [], []

    # def schedule_content(self):
    #     raise NotImplementedError


# Experimental design
# -------------------------------------


class ExperimentSetup(DataSetup):
    # that is, ExperimentSetting()
    def __init__(self, trial_type, data_type,
                 name_ens, abbr_cls, nb_cls, nb_pru,
                 nb_iter=5, ratio=.5, screen=True, logged=False):
        # super().__init__(trial_type, data_type)
        super().__init__(data_type)
        self._trial_type = trial_type

        nmens_temp = _get_tmp_name_ens(name_ens)
        self._log_document = "_".join([
            trial_type, nmens_temp, abbr_cls + str(nb_cls),
            self._log_document])  # aka. data_type

        self._nb_pru = nb_pru
        self._nb_iter = nb_iter
        self._ratio = ratio
        self._screen = screen
        self._logged = logged
        self._log_document += '_iter{}'.format(nb_iter)
        self._log_document += '_pms'
        # self._iterator = TrialPartSub(abbr_cls, nb_cls, name_ens)
        self._iterator = EnsembleSetup(name_ens, abbr_cls, nb_cls, nb_pru)

    def __del__(self):
        pass

    @property
    def trial_type(self):
        return self._trial_type

    @property
    def nb_iter(self):
        return self._nb_iter

    @property
    def ratio(self):
        return self._ratio

    @property
    def iterator(self):
        return self._iterator

    @iterator.setter
    def iterator(self, value):
        self._iterator = value

    def trial_one_process(self):
        since = time.time()
        csv_t = open(self._log_document + '.csv', "w")
        csv_w = csv.writer(csv_t)

        if (not self._screen) and (not self._logged):
            saveout = sys.stdout
            fsock = open(self._log_document + '.log', "w")
            sys.stdout = fsock
        if self._logged:
            logger = logging.getLogger("oracle_fair")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s:%(levelname)s | %(message)s')
            if os.path.exists(self._log_document + ".txt"):
                os.remove(self._log_document + ".txt")
            log_file = logging.FileHandler(self._log_document + '.txt')
            log_file.setLevel(logging.DEBUG)
            log_file.setFormatter(formatter)
            logger.addHandler(log_file)
        else:
            logger = None

        elegant_print("[BEGAN AT {:s}]".format(time.strftime(
            "%d-%b-%Y %H:%M:%S", time.localtime(since)
        )), logger)
        elegant_print(" EXPERIMENT  :", logger)
        elegant_print(["\t trial    = {}".format(self._trial_type),
                       "\t dataset  = {}".format(self._data_type),
                       "\t binary?  = {}".format(
            str(not self._trial_type.startswith('mu')))], logger)
        elegant_print(" PARAMETERS  :", logger)
        elegant_print(["\t name_ens = {}".format(self._iterator.name_ens),
                       "\t abbr_cls = {}".format(self._iterator.abbr_cls),
                       "\t   nb_cls = {}".format(self._iterator.nb_cls),
                       "\t   nb_pru = {}".format(self._iterator.nb_pru),
                       "\t  nb_iter = {}".format(self._nb_iter)], logger)
        elegant_print(" HYPER-PARAMS:", logger)
        elegant_print("", logger)

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

        # '''
        # tim_elapsed = (time.time() - since) / 60
        # elegant_print(
        #     "Total Time Cost: {:.0f}h {:.2f}m, i.e., {:.10f} hour(s)"
        #     ".".format(tim_elapsed // 60,
        #                tim_elapsed % 60, tim_elapsed / 60 ), logger)
        # '''
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

        res_all = []
        res_all.append([
            self._dataset.dataset_name, len(set(y)),
            self._iterator.name_ens, self._iterator.abbr_cls,
            self._iterator.nb_cls, self._iterator.nb_pru,
            self._nb_iter])
        for k, (i_trn, i_tst) in enumerate(split_idx):
            # '''
            # X_trn = X.iloc[i_trn]
            # y_trn = y.iloc[i_trn]
            # Xp_trn = Xp.iloc[i_trn]
            # X_tst = X.iloc[i_tst]
            # y_tst = y.iloc[i_tst]
            # Xp_tst = Xp.iloc[i_tst]
            # '''

            X_trn, Xd_trn, y_trn, gones_trn, jt_trn = transform_perturbed(
                X, Xp, y, i_trn, belongs_priv, ptb_with_joint)
            X_tst, Xd_tst, y_tst, gones_tst, jt_tst = transform_perturbed(
                X, Xp, y, i_tst, belongs_priv, ptb_with_joint)
            # X_*, Xd_*, y_*: pd.DataFrame
            # gones_*, jt_* : list of np.ndarray (element)
            X_trn = X_trn.to_numpy().tolist()
            X_tst = X_tst.to_numpy().tolist()
            y_trn = y_trn.to_numpy().reshape(-1).tolist()
            y_tst = y_tst.to_numpy().reshape(-1).tolist()
            Xd_trn = Xd_trn.to_numpy().tolist()
            Xd_tst = Xd_tst.to_numpy().tolist()

            # i-th K-Fold
            elegant_print("Iteration {}-th".format(k + 1), logger)
            res_iter = self.trial_one_iteration(
                logger, k,
                X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                X_tst, y_tst, Xd_tst, gones_tst, jt_tst)
            res_all.append(res_iter)
            del X_trn, Xd_trn, y_trn, gones_trn, jt_trn
            del X_tst, Xd_tst, y_tst, gones_tst, jt_tst
        del split_idx
        return res_all

    def trial_one_iteration(self, logger, k,
                            X_trn, y_trn, Xd_trn, gones_trn, jt_trn,
                            X_tst, y_tst, Xd_tst, gones_tst, jt_tst):
        since = time.time()
        # y_insp, _, y_pred, ut, us = \
        y_insp, _, y_pred, _ = \
            self._iterator.achieve_ensemble_from_train_set(
                # logger,
                X_trn, y_trn, [], X_tst)
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


# ----------------------------------
