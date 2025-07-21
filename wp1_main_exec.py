# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority vote
#


import argparse
import csv
import logging
import sys
import time

from fairml.widget.utils_saver import (
    get_elogger, rm_ehandler, elegant_print)
from fairml.widget.utils_timer import elegant_durat
from fairml.widget.data_split import (
    sklearn_k_fold_cv, sklearn_stratify, manual_cross_valid,
    situation_split1)

from fairml.datasets import preprocess
from fairml.preprocessing import (
    adversarial, transform_X_and_y, transform_unpriv_tag,
    transform_perturbed)
from experiment.wp2_oracle.fetch_data import ExperimentSetup
# from experiment.wp2_oracle.empirical import (
#     PartC_TheoremsLemma, PartK_PACGeneralisation,
#     PartH_ImprovedPruning, PartJ_LambdaEffect)
# from experiment.wp2_oracle.empirical import (
#     PartD_ImprovedPruning, PartF_ImprovedFairness)  # legacy
# from experiment.wp2_oracle.empirical import (
#     PartA_TheoremsLemma, PartB_TheoremsLemma, PartI_LambdaEffect,
#     PartG_ImprovedPruning, PartE_ImprovedFairness)  # legacy

from experiment.wp2_oracle.empirical import (
    PartC_TheoremsLemma, PartK_PACGeneralisation,
    PartJ_LambdaEffect, PartF_ImprovedFairness)
from experiment.wp2_oracle.empirical_ep import (
    PartH_ImprovedPruning, PartD_ImprovedPruning)
from experiment.wp2_oracle.empirical_ep import (
    PartG_ImprovedPruning)  # legacy
from experiment.wp2_oracle.empirical import (
    PartA_TheoremsLemma, PartB_TheoremsLemma,
    PartI_LambdaEffect, PartE_ImprovedFairness)  # legacy

AVAILABLE_FAIR_DATASET = [
    'ricci', 'german', 'adult', 'ppr', 'ppvr']


# =====================================
# Experiments
# =====================================


class OracleEmpirical(ExperimentSetup):
    def __init__(self, trial_type, data_type,
                 name_ens, abbr_cls, nb_cls, nb_pru=1,
                 nb_iter=5, lam=.45, ratio=.5,
                 epsilon=1e-3, rho=.4, alpha=.5, L=3, R=2,
                 nb_lam = 11, delta=1e-6,
                 screen=True, logged=False):
        super().__init__(trial_type, data_type,
                         name_ens, abbr_cls, nb_cls, nb_pru,
                         nb_iter, ratio, screen, logged)
        # trial_type: {mCV, KFS, KF}_part{*n}
        #  data_type: ['ricci', 'german', 'adult', 'ppr', 'ppvr']
        self._delta = 1. - delta

        if trial_type.endswith('expt1'):
            self._iterator = PartA_TheoremsLemma(
                name_ens, abbr_cls, nb_cls)
        elif trial_type.endswith('expt2'):
            self._iterator = PartB_TheoremsLemma(
                name_ens, abbr_cls, nb_cls)
        elif trial_type.endswith('expt3'):
            self._iterator = PartC_TheoremsLemma(
                name_ens, abbr_cls, nb_cls)

        elif trial_type.endswith('expt11'):
            self._iterator = PartK_PACGeneralisation(
                name_ens, abbr_cls, nb_cls)

        # rho: maximum size of pruned ensemble, aka. ratio'
        elif trial_type.endswith('expt4'):
            self._iterator = PartD_ImprovedPruning(
                name_ens, abbr_cls, nb_cls, nb_pru, lam,
                epsilon, rho, alpha, L, R)
        elif trial_type.endswith('expt7'):
            self._iterator = PartG_ImprovedPruning(
                name_ens, abbr_cls, nb_cls, nb_pru, lam,
                epsilon, rho, alpha, L, R)
        elif trial_type.endswith('expt8'):
            self._iterator = PartH_ImprovedPruning(
                name_ens, abbr_cls, nb_cls, nb_pru, lam,
                epsilon, rho, alpha, L, R)

        elif trial_type.endswith('expt9'):
            self._nb_lam = nb_lam
            self._log_document += '_lam{}'.format(nb_lam)
            self._iterator = PartI_LambdaEffect(
                name_ens, abbr_cls, nb_cls, nb_pru)
        elif trial_type.endswith('expt10'):
            self._nb_lam = nb_lam
            self._log_document += '_lam{}'.format(nb_lam)
            self._iterator = PartJ_LambdaEffect(
                name_ens, abbr_cls, nb_cls, nb_pru)

        elif trial_type.endswith('expt5'):
            self._iterator = PartE_ImprovedFairness(
                name_ens, abbr_cls, nb_cls, nb_pru, lam)
        elif trial_type.endswith('expt6'):
            self._iterator = PartF_ImprovedFairness(
                name_ens, abbr_cls, nb_cls, nb_pru, lam)

        elif trial_type.endswith('expt12'):
            self._iterator = None

    def trial_one_process(self):
        since = time.time()
        csv_t = open(self._log_document + '.csv', "w", newline="")
        csv_w = csv.writer(csv_t)

        if not (self._screen or self._logged):
            saveout = sys.stdout
            fsock = open(self._log_document + '.log', "w")
            sys.stdout = fsock

        if self._logged:
            logger, formatter, log_f = get_elogger(
                "oracle_fair", self._log_document + ".txt", mode='w')
        else:
            logger = None

        elegant_print([
            "[BEGAN AT {:s}]".format(
                time.strftime("%d-%b-%Y %H:%M:%S", time.localtime(since))),
            " EXPERIMENT",
            "\t   trial = {}".format(self._trial_type),
            "\t dataset = {}".format(self._data_type),
            "\t binary? = {}".format(not self._trial_type.startswith('mu')),
            " PARAMETERS",
            "\tname_ens = {}".format(self._iterator.name_ens),
            "\tabbr_cls = {}".format(self._iterator.abbr_cls),
            "\t  nb_cls = {}".format(self._iterator.nb_cls),
            "\t  nb_pru = {}".format(self._iterator.nb_pru),
            "\t nb_iter = {}".format(self._nb_iter),
            " HYPER-PARAMS", ""], logger)
        if self._trial_type.endswith('expt11'):
            elegant_print("\t  delta = {}".format(self._delta), logger)

        # START

        csv_row_2a = ['data_name', 'binary',
                      'name_ens', 'abbr_cls', 'nb_cls', 'nb_pru',
                      'nb_iter', 'iteration']
        csv_row_2b = ['Ensem_trn'] + [''] * 5 + ['Ensem_tst'] + [
            ''] * 5 + ['Ensem qtb_trn'] + [''] * 5 + ['Ensem qtb_tst'] + [
            ''] * 5 + ['Time Cost', '']  # 6*4+1 +1= 25+1=26
        csv_row_3b = ['Accuracy', 'Precision', 'Recall', 'F1',
                      'F2', 'F3'] * 4 + ['Ensem', 'Oracle']

        csv_row_1, csv_row_2c, csv_row_3c = self._iterator.prepare_trial()
        csv_row_2 = csv_row_2a + csv_row_2b + csv_row_2c
        if self._trial_type[-5:] in ['expt7', 'expt8']:
            csv_row_3 = [[''] * 8 + csv_row_3b + i for i in csv_row_3c]
            csv_w.writerows([csv_row_1, csv_row_2])
            csv_w.writerows(csv_row_3)
        elif self._trial_type[-6:] in ['expt10', 'expt11']:
            csv_row_2b = ['Time Cost (min)', '']  # earlier `(sec)`
            csv_row_3b = csv_row_3b[-2:]
            csv_row_2 = csv_row_2a + csv_row_2b + csv_row_2c
            csv_row_3 = [''] * 8 + csv_row_3b + csv_row_3c
            csv_w.writerows([csv_row_1, csv_row_2, csv_row_3])
        else:
            csv_row_3 = [''] * 8 + csv_row_3b + csv_row_3c
            csv_w.writerows([csv_row_1, csv_row_2, csv_row_3])
        del csv_row_1, csv_row_2, csv_row_3
        del csv_row_2a, csv_row_2b, csv_row_2c, csv_row_3b, csv_row_3c

        res_all = self.trial_one_dataset(logger=logger)
        csv_w.writerows(res_all)
        del res_all

        # -END-

        tim_elapsed = time.time() - since
        tim_goes_by = tim_elapsed / 60
        since = time.time()
        elegant_print([
            "",
            " Duration in total: {}".format(elegant_durat(tim_elapsed)),
            " Time cost: {:.0f} hrs {:.2f} min"
            "".format(tim_goes_by // 60, tim_goes_by % 60),
            "            {:.0f} min {:.2f} sec"
            "".format(tim_elapsed // 60, tim_elapsed % 60),
            "            {:.10f} hour(s)".format(tim_goes_by / 60),
            "[ENDED AT {:s}]".format(
                time.strftime("%d-%b-%Y %H:%M:%S", time.localtime(since)))
        ], logger)
        del since, tim_elapsed, tim_goes_by  # tim_passby

        if self._logged:
            rm_ehandler(logger, formatter, log_f)
            del log_f, formatter
        del logger

        if not (self._screen or self._logged):
            fsock.close()
            sys.stdout = saveout

        csv_t.close()
        del csv_t, csv_w
        logging.shutdown()
        return

    def trial_one_dataset(self, logger=None):
        processed_data = preprocess(
            self._dataset, self._data_frame, logger)
        disturbed_data = adversarial(
            self._dataset, self._data_frame, self._ratio, logger)

        processed_Xy = processed_data['numerical-binsensitive']
        disturbed_Xy = disturbed_data['numerical-binsensitive']
        X, y = transform_X_and_y(self._dataset, processed_Xy)
        Xp, _ = transform_X_and_y(self._dataset, disturbed_Xy)
        # X, Xp, y: pd.DataFrame

        ptb_priv, ptb_with_joint = transform_unpriv_tag(
            self._dataset, processed_data['original'])
        # Note that PTB: place to belong
        # ptb_priv: list of np.ndarray (element in np.ndarray: boolean)
        #           number= how many sensitive attributes do they have
        # ptb_with_joint: [] or list of boolean (as elements)

        tmp = processed_data['original'][self._dataset.label_name]
        elegant_print([
            "\tBINARY? Y = {}".format(set(y)),
            "\tNB. formerly",
            "\t     ds.Y = {}".format(set(tmp)),
            "\t\tlabel_name     : {}".format(self._dataset.label_name),
            "\t\tpositive_label : {}".format(self._dataset.positive_label),
            "\t\tsensitive_attrs: {}".format(self._dataset.sensitive_attrs),
            "\t\tprivileged_vals: {}".format(self._dataset.privileged_vals),
            "\t\t- dataset_name : {}".format(self._dataset.dataset_name)
        ], logger)
        del tmp

        # {tr/bi/mu}_{KF,KFS,mCV}_{trial_part_*}
        if self._nb_iter == 1:
            split_idx = situation_split1(y, pr_trn=.8)
        elif "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
            elegant_print("\tCrossValid= {}".format('mCV'), logger)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y, X)
            elegant_print("\tCrossValid= {}".format('KFS'), logger)
        elif "KF" in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
            elegant_print("\tCrossValid= {}".format('KF '), logger)
        else:
            raise ValueError("No proper CV (cross-validation).")

        # START
        res_all = []
        res_all.append([
            self._dataset.dataset_name, len(set(y)),
            self._iterator.name_ens, self._iterator.abbr_cls,
            self._iterator.nb_cls, self._iterator.nb_pru,
            self._nb_iter])

        for k, (i_trn, i_tst) in enumerate(split_idx):
            X_trn, Xd_trn, y_trn, tag_trn, jt_trn = transform_perturbed(
                X, Xp, y, i_trn, ptb_priv, ptb_with_joint)
            X_tst, Xd_tst, y_tst, tag_tst, jt_tst = transform_perturbed(
                X, Xp, y, i_tst, ptb_priv, ptb_with_joint)
            # X,Xd,y: pd.DataFrame
            # tag,jt: list of np.ndarray, & []/[boolean]

            X_trn = X_trn.to_numpy().tolist()
            X_tst = X_tst.to_numpy().tolist()
            y_trn = y_trn.to_numpy().reshape(-1).tolist()
            y_tst = y_tst.to_numpy().reshape(-1).tolist()
            Xd_trn = Xd_trn.to_numpy().tolist()
            Xd_tst = Xd_tst.to_numpy().tolist()

            # i-th K-Fold
            elegant_print("Iteration {}-th".format(k + 1), logger)
            res_iter = self.trial_one_iteration(
                logger,  # k,
                X_trn, y_trn, Xd_trn, tag_trn, jt_trn,
                X_tst, y_tst, Xd_tst, tag_tst, jt_tst)

            if self._trial_type[-5:] in ['expt6', 'expt7', 'expt8']:
                res_all.extend([([''] * 7 + [k] + i) for i in res_iter])
            elif self._trial_type.endswith(
                    'expt9') or self._trial_type.endswith('expt10'):
                res_all.extend([([''] * 7 + [k] + i) for i in res_iter])
            else:
                res_all.append([''] * 7 + [k] + res_iter)
            del X_trn, Xd_trn, y_trn, tag_trn, jt_trn
            del X_tst, Xd_tst, y_tst, tag_tst, jt_tst

        # -END-
        del split_idx
        return res_all

    def trial_one_iteration(self, logger,  # k,
                            X_trn, y_trn, Xd_trn, tag_trn, jt_trn,
                            X_tst, y_tst, Xd_tst, tag_tst, jt_tst):
        since = time.time()  # Xd: disturb/data; Xp: prime
        y_insp, _, y_pred, indices = \
            self._iterator.achieve_ensemble_from_train_set(
                X_trn, y_trn, [], X_tst)
        yd_insp = [j.predict(Xd_trn) for j in self._iterator.member]
        yd_pred = [j.predict(Xd_tst) for j in self._iterator.member]
        # {y,yd}_{insp,pred}: list of np.ndarray, each size: [#inst,]

        fens_trn = self._iterator.majority_vote(y_trn, y_insp)
        fens_tst = self._iterator.majority_vote(y_tst, y_pred)
        fqtb_trn = self._iterator.majority_vote(y_trn, yd_insp)
        fqtb_tst = self._iterator.majority_vote(y_tst, yd_pred)
        # all list, size=(#inst,)

        positive_label = self._dataset.get_positive_class_val(
            'numerical-binsensitive')
        res_ens = []

        Acc, (a, p, r, f, f1, f2, f3, tpr, fpr, fnr, tnr) = \
            self._iterator.calculate_sub_ensemble_metrics(
            y_trn, fens_trn, positive_label)
        res_ens.extend([a, p, r, f1, f2, f3])
        Acc, (a, p, r, f, f1, f2, f3, tpr, fpr, fnr, tnr) = \
            self._iterator.calculate_sub_ensemble_metrics(
            y_tst, fens_tst, positive_label)
        res_ens.extend([a, p, r, f1, f2, f3])

        Acc, (a, p, r, f, f1, f2, f3, tpr, fpr, fnr, tnr) = \
            self._iterator.calculate_sub_ensemble_metrics(
            y_trn, fqtb_trn, positive_label)
        res_ens.extend([a, p, r, f1, f2, f3])
        Acc, (a, p, r, f, f1, f2, f3, tpr, fpr, fnr, tnr) = \
            self._iterator.calculate_sub_ensemble_metrics(
            y_tst, fqtb_tst, positive_label)
        res_ens.extend([a, p, r, f1, f2, f3])

        tim_elapsed = time.time() - since
        elegant_print(
            "\tEnsem : time cost {:.4f} seconds".format(tim_elapsed), logger)
        res_ens.append(tim_elapsed)  # 6*4+1 =25
        del Acc, a, p, r, f, f1, f2, f3, tpr, fpr, fnr, tnr

        # START
        since = time.time()
        res_bnd = []  # BND: abbr. bound

        if self._trial_type.endswith('expt1'):
            tmp = self._iterator.schedule_content(
                y_trn, y_insp, fens_trn, yd_insp, fqtb_trn)
            res_bnd.extend(tmp)
            tmp = self._iterator.schedule_content(
                y_tst, y_pred, fens_tst, yd_pred, fqtb_tst)
            res_bnd.extend(tmp)
            del tmp
        elif self._trial_type.endswith('expt2'):
            tmp = self._iterator.schedule_content(
                y_trn, y_insp, fens_trn, yd_insp, fqtb_trn)
            res_bnd.extend(tmp)
            tmp = self._iterator.schedule_content(
                y_tst, y_pred, fens_tst, yd_pred, fqtb_tst)
            res_bnd.extend(tmp)
            del tmp
        elif self._trial_type.endswith('expt3'):
            tmp = self._iterator.schedule_content(
                y_trn, y_insp, fens_trn, yd_insp, fqtb_trn)
            res_bnd.extend(tmp)
            tmp = self._iterator.schedule_content(
                y_tst, y_pred, fens_tst, yd_pred, fqtb_tst)
            res_bnd.extend(tmp)
            del tmp

        elif self._trial_type.endswith('expt11'):
            tmp = self._iterator.schedule_content(
                y_trn, y_insp, yd_insp, fens_trn, fqtb_trn,
                y_tst, y_pred, yd_pred, fens_tst, fqtb_tst,
                delta=self._delta)
            tim_elapsed = time.time() - since
            elegant_print("\tPACbnd: time cost {:.4f} seconds"
                          "".format(tim_elapsed), logger)
            ut = [res_ens[-1] / 60, tim_elapsed / 60]
            return ut + tmp

        elif self._trial_type.endswith('expt4'):
            res_bnd = self._iterator.schedule_content(
                y_trn, y_insp, yd_insp, y_tst, y_pred, yd_pred,
                positive_label, X_trn, indices)
        elif self._trial_type[-5:] in ['expt7', 'expt8']:
            ut, attr_A1, attr_A2, attr_Jt = self._iterator.schedule_content(
                y_trn, y_insp, yd_insp, tag_trn, jt_trn,
                y_tst, y_pred, yd_pred, tag_tst, jt_tst,
                fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                positive_label, X_trn, indices)
            attr_A1 = [''] * 26 + attr_A1
            attr_A2 = [''] * 26 + attr_A2
            attr_Jt = [''] * 26 + attr_Jt
            tim_elapsed = time.time() - since
            elegant_print("\tOPrune: time cost {:.4f} seconds"
                          "".format(tim_elapsed), logger)
            res_ens.append(tim_elapsed)
            res_ens.extend(ut)  # 26+23*?
            return [res_ens, attr_A1, attr_A2, attr_Jt]

        elif self._trial_type.endswith('expt9'):
            ut, attr_A1, attr_A2, attr_Jt = self._iterator.schedule_content(
                y_trn, y_insp, yd_insp, tag_trn, jt_trn,
                y_tst, y_pred, yd_pred, tag_tst, jt_tst,
                fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                positive_label, nb_lam=self._nb_lam, logger=logger)
            ut = [[''] * 26 + i for i in ut]
            attr_A1 = [[''] * 26 + i for i in attr_A1]
            attr_A2 = [[''] * 26 + i for i in attr_A2]
            attr_Jt = [[''] * 26 + i for i in attr_Jt]
            tim_elapsed = time.time() - since
            elegant_print("\tOPrune: time cost {:.4f} seconds"
                          "".format(tim_elapsed), logger)
            res_ens.append(tim_elapsed)
            return [res_ens] + ut + attr_A1 + attr_A2 + attr_Jt
        elif self._trial_type.endswith('expt10'):
            ut, attr_A1, attr_A2, attr_Jt = self._iterator.schedule_content(
                y_trn, y_insp, yd_insp, tag_trn, jt_trn,
                y_tst, y_pred, yd_pred, tag_tst, jt_tst,
                fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                positive_label, nb_lam=self._nb_lam, logger=logger)
            attr_A1 = [[''] * 2 + i for i in attr_A1]
            attr_A2 = [[''] * 2 + i for i in attr_A2]
            attr_Jt = [[''] * 2 + i for i in attr_Jt]
            ut = [[''] * 2 + i for i in ut]
            tim_elapsed = time.time() - since
            elegant_print("\tOPrune: time cost {:.4f} seconds"
                          "".format(tim_elapsed), logger)
            res_ens = [res_ens[-1] / 60, tim_elapsed / 60]
            return [res_ens] + ut + attr_A1 + attr_A2 + attr_Jt

        elif self._trial_type.endswith('expt5'):
            res_bnd = self._iterator.schedule_content(
                y_trn, y_insp, yd_insp, tag_trn, jt_trn,
                y_tst, y_pred, yd_pred, tag_tst, jt_tst,
                fens_trn, fqtb_trn, fens_tst, fqtb_tst, positive_label)
        elif self._trial_type.endswith('expt6'):
            ut, attr_A1, attr_A2, attr_Jt = self._iterator.schedule_content(
                y_trn, y_insp, yd_insp, tag_trn, jt_trn,
                y_tst, y_pred, yd_pred, tag_tst, jt_tst,
                fens_trn, fqtb_trn, fens_tst, fqtb_tst, positive_label)
            attr_A1 = [''] * (26 + 6) + attr_A1  # +1+33*2*5
            attr_A2 = [''] * (26 + 6) + attr_A2  # +1+33*2*5
            attr_Jt = [''] * (26 + 6) + attr_Jt  # +1+33*2*5
            tim_elapsed = time.time() - since
            elegant_print("\tOracle: time cost {:.4f} seconds"
                          "".format(tim_elapsed), logger)
            res_ens.append(tim_elapsed)
            ut = res_ens + ut  # 26+6 =32
            return [ut, attr_A1, attr_A2, attr_Jt]

        elif self._trial_type.endswith('expt12'):
            self._iterator.schedule_content()

        else:
            raise ValueError("No such experiment designed/designated.")

        tim_elapsed = time.time() - since
        elegant_print("\tOracle: time cost {:.4f} seconds".format(
            tim_elapsed), logger)
        res_ens.append(tim_elapsed)
        # -END-
        return res_ens + res_bnd


# =====================================
# Trials
# =====================================


def default_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-exp', '--expt-id', type=str, default='mCV_expt3',
        help='Type of trial: experiment id')
    parser.add_argument(  # , help='Data set'
        '-dat', '--dataset', type=str, default='ricci',
        choices=['ricci', 'german', 'adult', 'ppr', 'ppvr'])
    parser.add_argument('-add', '--add-expt', action='store_true')

    parser.add_argument(
        '--name-ens', type=str, default='Bagging',
        choices=['Bagging', 'AdaBoostM1', 'SAMME'],
        help='Construct ensemble classifiers')
    parser.add_argument(
        '--abbr-cls', type=str, default='DT', choices=[
            'DT', 'NB', 'SVM', 'linSVM', 'LR', 'kNNu', 'kNNd', 'MLP',
            'lmSGD', 'NN', 'LM', 'LR1', 'LR2', 'LM1', 'LM2',
        ], help='Individual classifiers')
    parser.add_argument(
        '--nb-cls', type=int, default=21, help='Size of ensemble')
    parser.add_argument(
        '--nb-pru', type=int, default=11, help='Size of pruned sub-')
    parser.add_argument('-nk', '--nb-iter', type=int, default=5,
                        help='Cross validation')  # '-it'

    parser.add_argument(
        '--ratio', type=float, default=.4,
        help='Percentage/proportion/ratio of pruned sub-ensemble')
    parser.add_argument(
        '--lam', type=float, default=.5, help='Regularization factor')
    parser.add_argument('--epsilon', type=float, default=1e-4)
    parser.add_argument('--rho', type=float, default=.4)
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--L', type=int, default=3)
    parser.add_argument('--R', type=int, default=2)

    parser.add_argument(
        '--nb-lam', type=int, default=5, help='Number of lam values')
    parser.add_argument(
        '--delta', type=float, default=1e-6, help='$1-\delta$')
    parser.add_argument(
        '--screen', action='store_true', help='Where to output')
    parser.add_argument(
        '--logged', action='store_true', help='Where to output')
    return parser


screen = logged = None
parser = default_parameters()
args = parser.parse_args()

# Parse args
trial_type = args.expt_id
data_type = args.dataset
name_ens = args.name_ens
abbr_cls = args.abbr_cls
nb_cls = args.nb_cls
nb_pru = args.nb_pru

nb_iter = args.nb_iter
screen = args.screen
logged = args.logged
# ratio = float(nb_pru) / nb_cls
# Note that there is a bit misunderstanding.
#   ratio is the percentage of how many instances are disturbed.
rho = float(nb_pru) / nb_cls


if args.add_expt:
    kwargs = {}
    if trial_type.endswith('expt3'):
        kwargs['name_ens'] = args.name_ens
        kwargs['abbr_cls'] = args.abbr_cls
    # elif trial_type[-5:] in ('expt4', 'expt5', 'expt6'):
    #     kwargs['gather'] = args.gather
    case = ExperimentSetup(
        trial_type, data_type, nb_cls, nb_pru,
        nb_iter, args.ratio, args.lam,
        screen=screen, logged=logged, **kwargs)
    case.trial_one_process()
    del kwargs, case, rho, screen, logged
    del nb_cls, nb_pru, abbr_cls, name_ens
    del nb_iter, data_type, trial_type
    sys.exit()


kwargs = {}
if trial_type.endswith('3') or trial_type.endswith('11'):
    if trial_type.endswith('11'):
        kwargs['delta'] = args.delta
    case = OracleEmpirical(
        trial_type, data_type, name_ens, abbr_cls, nb_cls,
        nb_iter=nb_iter, screen=screen, logged=logged, **kwargs)
else:
    if trial_type[-1:] in ('8', '4', '6'):
        kwargs['lam'] = args.lam
    elif trial_type.endswith('9') or trial_type.endswith('10'):
        kwargs['nb_lam'] = args.nb_lam

    if trial_type[-1:] in ('8', '4'):
        kwargs['ratio'] = args.ratio
        kwargs['epsilon'] = args.epsilon
        kwargs['rho'] = rho
        kwargs['alpha'] = args.alpha
        kwargs['L'] = args.L
        kwargs['R'] = args.R
    case = OracleEmpirical(
        trial_type, data_type, name_ens, abbr_cls, nb_cls,
        nb_pru, nb_iter, screen=screen, logged=logged, **kwargs)


case.trial_one_process()

del screen, logged, nb_iter, case, kwargs
del name_ens, abbr_cls, nb_cls, nb_pru, rho
del trial_type, data_type, args, parser


# Experiments
"""
python wp1_main_exec.py --logged -exp mCV_expt3 --name-ens Bagging --abbr-cls DT --nb-cls 11 -dat ricci
python wp1_main_exec.py --logged -exp mCV_expt11 --name-ens Bagging --abbr-cls DT --nb-cls 11 --nb-pru 5 --delta 1e-6 -dat ricci
python wp1_main_exec.py --logged -exp mCV_expt8 --name-ens Bagging --abbr-cls DT --nb-cls 11 --nb-pru 5 -dat ricci
python wp1_main_exec.py --logged -exp mCV_expt10 --name-ens Bagging --abbr-cls DT --nb-cls 11 --nb-pru 5 --nb-lam 9 -nk 2 -dat ricci

# legacy
python wp1_main_exec.py --logged -exp mCV_expt4 --name-ens Bagging --abbr-cls DT --nb-cls 21 --nb-pru 11 -dat *
python wp1_main_exec.py --logged -exp mCV_expt6 --name-ens Bagging --abbr-cls DT --nb-cls 21 --nb-pru 7 -dat *
python wp1_main_exec.py --logged -exp mCV_expt4 --name-ens Bagging --abbr-cls DT --nb-cls 7 --nb-pru 3 -nk 2 -dat ricci
python wp1_main_exec.py --logged -exp mCV_expt6 --name-ens Bagging --abbr-cls DT --nb-cls 7 --nb-pru 3 -nk 2 -dat ricci

# add_expt
python wp1_main_exec.py -add -exp mCV_expt1 -dat german
"""
