# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority vote
#


import numpy as np
# import pandas as pd
import time

from fairml.widget.utils_const import check_zero, unique_column
from fairml.widget.utils_saver import elegant_print

from fairml.discriminative_risk import (
    ell_fair_x, hat_L_fair, tandem_fair, E_rho_L_fair_f,
    ell_loss_x, hat_L_loss, tandem_loss, E_rho_L_loss_f,
    Erho_sup_L_fair, Erho_sup_L_loss, ED_Erho_I_fair,
    hat_L_objt, cal_L_obj_v1, cal_L_obj_v2)
from fairml.dr_pareto_optimal import (
    Pareto_Optimal_EPAF_Pruning, Centralised_EPAF_Pruning,
    Distributed_EPAF_Pruning, POAF_PEP, _POAF_calc_eval)
from fairml.dr_pareto_optimal import _bi_objectives  # as _POAF_calc_eval

from experiment.wp2_oracle.fetch_data import EnsembleSetup
# from fairml.facils.fairness_group import (
#     unpriv_group_one, unpriv_group_two, unpriv_group_thr,
#     marginalised_pd_mat, unpriv_unaware, unpriv_manual)
from fairml.facilc.metric_fair import marginalised_pd_mat
from fairml.facilc.metric_fair import prev_unpriv_grp_one \
    as unpriv_group_one
from fairml.facilc.metric_fair import prev_unpriv_grp_two \
    as unpriv_group_two
from fairml.facilc.metric_fair import prev_unpriv_grp_thr \
    as unpriv_group_thr
from fairml.facilc.metric_fair import prev_unpriv_unaware \
    as unpriv_unaware
from fairml.facilc.metric_fair import prev_unpriv_manual \
    as unpriv_manual


# =====================================
# Properties
# =====================================


# Trail Part A.  To demonstrate that 
#   for one classifier (individual or ensemble), calculate its
#   values of optimisation objectives
# -------------------------------------


class PartA_TheoremsLemma(EnsembleSetup):
    def __init__(self, name_ens, abbr_cls, nb_cls):
        super().__init__(name_ens, abbr_cls, nb_cls)

    def schedule_content(self, y, yt, fens, yqtb, fqtb):
        #   y : list, (nb_inst,)
        #   yt: list of np.ndarray, (nb_cls, nb_inst)
        # fens: list, (nb_inst,)
        yt = [j.tolist() for j in yt]
        # res_thm = []

        # Theorem 3.1 (First order oracle bound)
        #   L_fair(MV_rho) <= 2E_rho[ L_fair(f)]
        tmp_fair = [hat_L_fair(i, j) for i, j in zip(yt, yqtb)]
        tmp_loss = [hat_L_loss(i, y) for i in yt]

        ans_fair = [hat_L_fair(fens, fqtb), np.mean(tmp_fair).tolist()]
        ans_loss = [hat_L_loss(fens, y), np.mean(tmp_loss).tolist()]

        # Theorem 3.3 (Second order oracle bound)
        #   L_fair(MV_rho) <= 4E_rho^2[ L_fair(f,f')]
        tmp_fair = [[tandem_fair(yt[i], yqtb[i], yt[j], yqtb[j]
                                 ) for j in range(self._nb_cls)
                     ] for i in range(self._nb_cls)]
        tmp_loss = [[tandem_loss(i, j, y) for j in yt] for i in yt]

        tmp_fair = np.mean(np.mean(tmp_fair, axis=1), axis=0).tolist()
        tmp_loss = np.mean(np.mean(tmp_loss, axis=1), axis=0).tolist()
        ans_fair.append(tmp_fair)
        ans_loss.append(tmp_loss)

        # Lemma 3.2
        #   E_D[ E_rho[ ell_fair(f,bmx)]^2] = E_rho^2[ L_fair(f,f')]
        tmp_fair = np.mean([ell_fair_x(
            i, j) for i, j in zip(yt, yqtb)], axis=0).tolist()
        tmp_loss = np.mean([ell_loss_x(
            i, y) for i in yt], axis=0).tolist()
        tmp_fair = np.mean([i**2 for i in tmp_fair]).tolist()
        tmp_loss = np.mean([i**2 for i in tmp_loss]).tolist()
        ans_fair.append(tmp_fair)
        ans_loss.append(tmp_loss)

        # Lemma 3.4, calc later when plotting
        tmp_fair = check_zero(ans_fair[2] - ans_fair[1] + 1. / 4)
        tmp_fair = (ans_fair[2] - ans_fair[1]**2) / tmp_fair
        tmp_loss = check_zero(ans_loss[2] - ans_loss[1] + 1 / 4.)
        tmp_loss = (ans_loss[2] - ans_loss[1]**2) / tmp_loss
        ans_fair.append(tmp_fair)
        ans_loss.append(tmp_loss)

        # Theorem 3.5, calc later
        return ans_fair + ans_loss  # 5+5

    def prepare_trial(self):
        csv_row_1 = unique_column(8 + 25 + 1 + 10 * 2)
        csv_row_2c = [
            'L_fair (trn)'] + [''] * 4 + ['L_loss (trn)'] + [''] * 4 + [
            'L_fair (tst)'] + [''] * 4 + ['L_loss (tst)'] + [''] * 4
        csv_row_3c = ['MVrho', 'Erho', 'Erho2', 'EDErho', 'calc'] * 4
        return csv_row_1, csv_row_2c, csv_row_3c


class PartB_TheoremsLemma(EnsembleSetup):
    def __init__(self, name_ens, abbr_cls, nb_cls):
        super().__init__(name_ens, abbr_cls, nb_cls)

    def schedule_content(self, y, yt, fens, yqtb, fqtb):
        yt = [j.tolist() for j in yt]
        # ans_fair, ans_loss = [], []
        ans_fair = [hat_L_fair(fens, fqtb)]
        ans_loss = [hat_L_loss(fens, y)]

        # Theorem 3.1
        tmp_fair = [hat_L_fair(i, j) for i, j in zip(yt, yqtb)]
        tmp_loss = [hat_L_loss(i, y) for i in yt]
        tmp_fair = np.mean(tmp_fair).tolist()
        tmp_loss = np.mean(tmp_loss).tolist()
        ans_fair.extend([tmp_fair, 2. * tmp_fair])
        ans_loss.extend([tmp_loss, 2. * tmp_loss])

        # Theorem 3.3
        tmp_fair = [[tandem_fair(yt[i], yqtb[i], yt[j], yqtb[j]
                                 ) for j in range(self._nb_cls)
                     ] for i in range(self._nb_cls)]
        tmp_loss = [[tandem_loss(i, j, y) for j in yt] for i in yt]
        tmp_fair = np.mean(np.mean(tmp_fair, axis=1), axis=0).tolist()
        tmp_loss = np.mean(np.mean(tmp_loss, axis=1), axis=0).tolist()
        ans_fair.extend([tmp_fair, 4. * tmp_fair])
        ans_loss.extend([tmp_loss, 4. * tmp_loss])

        # Lemma 3.2
        tmp_fair = np.mean([ell_fair_x(
            i, j) for i, j in zip(yt, yqtb)], axis=0).tolist()
        tmp_loss = np.mean([ell_loss_x(
            i, y) for i in yt], axis=0).tolist()
        tmp_fair = np.mean([i**2 for i in tmp_fair]).tolist()
        tmp_loss = np.mean([i**2 for i in tmp_loss]).tolist()
        ans_fair.append(tmp_fair)
        ans_loss.append(tmp_loss)

        # Lemma 3.4
        tmp_fair = check_zero(ans_fair[3] - ans_fair[1] + 1. / 4)
        tmp_fair = (ans_fair[3] - ans_fair[1]**2) / tmp_fair
        tmp_loss = check_zero(ans_loss[3] - ans_loss[1] + 1 / 4.)
        tmp_loss = (ans_loss[3] - ans_loss[1]**2) / tmp_loss
        ans_fair.append(tmp_fair)
        ans_loss.append(tmp_loss)

        # Theorem 3.5
        return ans_fair + ans_loss  # 1+2*2+1+1 =7

    def prepare_trial(self):
        csv_row_1 = unique_column(8 + 26 + 7 * 4)
        csv_row_2c = [
            'L_fair (trn)'] + [''] * 6 + ['L_loss (trn)'] + [''] * 6 + [
            'L_fair (tst)'] + [''] * 6 + ['L_loss (tst)'] + [''] * 6
        csv_row_3c = [
            'MVrho', 'Erho', 'x2', 'Erho2', 'x4', 'EDErho', 'calc'] * 4
        return csv_row_1, csv_row_2c, csv_row_3c


class PartC_TheoremsLemma(PartB_TheoremsLemma):
    def __init__(self, name_ens, abbr_cls, nb_cls):
        super().__init__(name_ens, abbr_cls, nb_cls)

    def schedule_content(self, y, yt, fens, yqtb, fqtb):
        yt = [j.tolist() for j in yt]
        # ans_fair, ans_loss = [], []
        ans_fair = [hat_L_fair(fens, fqtb)]
        ans_loss = [hat_L_loss(fens, y)]
        coef = self._weight

        # Theorem 3.1
        tmp_fair = [hat_L_fair(i, j) for i, j in zip(yt, yqtb)]
        tmp_loss = [hat_L_loss(i, y) for i in yt]
        tmp_fair = np.sum(np.multiply(coef, tmp_fair)).tolist()
        tmp_loss = np.sum(np.multiply(coef, tmp_loss)).tolist()
        ans_fair.extend([tmp_fair, 2. * tmp_fair])
        ans_loss.extend([tmp_loss, 2. * tmp_loss])

        # Theorem 3.3
        tmp_fair = [[tandem_fair(yt[i], yqtb[i], yt[j], yqtb[j]
                                 ) for j in range(self._nb_cls)
                     ] for i in range(self._nb_cls)]
        tmp_loss = [[tandem_loss(i, j, y) for j in yt] for i in yt]
        tmp_fair = np.sum(np.multiply(tmp_fair, coef), axis=1)
        tmp_fair = np.sum(np.multiply(tmp_fair, coef), axis=0)
        tmp_loss = np.sum(np.multiply(tmp_loss, coef), axis=1)
        tmp_loss = np.sum(np.multiply(tmp_loss, coef), axis=0)
        tmp_fair, tmp_loss = tmp_fair.tolist(), tmp_loss.tolist()
        ans_fair.extend([tmp_fair, 4. * tmp_fair])
        ans_loss.extend([tmp_loss, 4. * tmp_loss])

        # Lemma 3.2
        coef = np.array([coef]).T
        tmp_fair = [ell_fair_x(i, j) for i, j in zip(yt, yqtb)]
        tmp_loss = [ell_loss_x(i, y) for i in yt]
        tmp_fair = np.sum(coef * np.array(tmp_fair), axis=0)
        tmp_loss = np.sum(coef * np.array(tmp_loss), axis=0)
        tmp_fair = np.mean(tmp_fair ** 2).tolist()
        tmp_loss = np.mean(tmp_loss ** 2).tolist()
        ans_fair.append(tmp_fair)
        ans_loss.append(tmp_loss)

        # Theorem 3.4
        tmp_fair = check_zero(ans_fair[3] - ans_fair[1] + 1. / 4)
        tmp_loss = check_zero(ans_loss[3] - ans_loss[1] + 1. / 4)
        tmp_fair = (ans_fair[3] - ans_fair[1]**2) / tmp_fair
        tmp_loss = (ans_loss[3] - ans_loss[1]**2) / tmp_loss
        ans_fair.append(tmp_fair)
        ans_loss.append(tmp_loss)

        # Theorem 3.5

        return ans_fair + ans_loss  # 1+2*2+1+1 =7

    def prepare_trial(self):
        csv_row_1 = unique_column(8 + 26 + 7 * 4)
        csv_row_2c = [
            'L_fair (trn)'] + [''] * 6 + ['L_loss (trn)'] + [''] * 6 + [
            'L_fair (tst)'] + [''] * 6 + ['L_loss (tst)'] + [''] * 6
        csv_row_3c = [
            'MVrho', 'Erho', 'x2', 'Erho2', 'x4', 'EDErho', 'calc'] * 4
        return csv_row_1, csv_row_2c, csv_row_3c


# Generalisation bounds
# -------------------------------------


class PartK_PACGeneralisation(EnsembleSetup):
    def __init__(self, name_ens, abbr_cls, nb_cls):
        super().__init__(name_ens, abbr_cls, nb_cls)

    def schedule_content(self,  # y, yt, fens, yqtb, fqtb
                         y_trn, y_insp, yq_insp, fens_trn, fqtb_trn,
                         y_tst, y_pred, yq_pred, fens_tst, fqtb_tst,
                         delta=1 - 1e-6):
        y_insp = [j.tolist() for j in y_insp]
        y_pred = [j.tolist() for j in y_pred]
        yq_insp = [j.tolist() for j in yq_insp]
        yq_pred = [j.tolist() for j in yq_pred]
        # coef = self._weight

        # Theorem 3.6
        m_n = len(y_trn)
        trn_bnd = np.log(self._nb_cls / 1)
        trn_bnd = np.sqrt(trn_bnd / (2 * m_n))

        trn_fair = hat_L_fair(fens_trn, fqtb_trn)
        tst_fair = hat_L_fair(fens_tst, fqtb_tst)
        trn_loss = hat_L_loss(fens_trn, y_trn)
        tst_loss = hat_L_loss(fens_tst, y_tst)

        ans = [trn_bnd,
               trn_fair, tst_fair, trn_fair + trn_bnd,
               trn_loss, tst_loss, trn_loss + trn_bnd]

        # Theorem 3.6
        ens_bnd = np.log(self._nb_cls / delta)
        ens_bnd = np.sqrt(ens_bnd / (2. * m_n))
        ans.extend([ens_bnd,
                    trn_fair, tst_fair, trn_fair + ens_bnd,
                    trn_loss, tst_loss, trn_loss + ens_bnd])

        # Theorem 3.5
        ind_bnd = np.log(1. / 1.)
        ind_bnd = np.sqrt(ind_bnd / (2. * m_n))
        ans.append(ind_bnd)

        for i in range(self._nb_cls):
            trn_fair = hat_L_fair(y_insp[i], yq_insp[i])
            tst_fair = hat_L_fair(y_pred[i], yq_pred[i])
            trn_loss = hat_L_loss(y_insp[i], y_trn)
            tst_loss = hat_L_loss(y_pred[i], y_tst)

            ans.extend([trn_fair, tst_fair, trn_fair + ind_bnd,
                        trn_loss, tst_loss, trn_loss + ind_bnd])

        # Lemma 3.7 / C.1
        return ans

    def prepare_trial(self):
        length = self._nb_cls * 6 + (7 * 2 + 1)
        csv_row_1 = unique_column(8 + 2 + length)

        csv_row_2c_a = ['trn_bnd'] + [
            ''] * 6 + ['ens_bnd'] + [''] * 6 + ['ind_bnd']
        csv_row_3c_a = [
            '', 'fair', '', '', 'loss', '', ''] * 2 + ['']

        csv_row_2c_b, csv_row_3c_b = [], []
        for i in range(self._nb_cls):
            csv_row_2c_b.extend(['cls#{}'.format(i + 1)] + [''] * 5)
            csv_row_3c_b.extend(['fair', '', '', 'loss', '', ''])

        csv_row_2c = csv_row_2c_a + csv_row_2c_b
        csv_row_3c = csv_row_3c_a + csv_row_3c_b
        del csv_row_2c_a, csv_row_2c_b
        del csv_row_3c_a, csv_row_3c_b  # (7*2+1)+6*?

        return csv_row_1, csv_row_2c, csv_row_3c


# PAC-Bayesian bounds
# -------------------------------------


# =====================================
# Properties
# =====================================


# Trial Part B.  To demonstrate that
#   for an ensemble, 
# -------------------------------------


class PartD_ImprovedPruning(EnsembleSetup):
    def __init__(self, name_ens, abbr_cls, nb_cls, nb_pru,
                 lam, epsilon, rho, alpha, L, R):
        # , logger=None):
        super().__init__(name_ens, abbr_cls, nb_cls, nb_pru)
        self._set_pru_prop = ['EPAF-C', 'EPAF-D', 'POEPAF']

        self._set_pru_name = [
            'ES', 'KL', 'KL+', 'KPk', 'KPz',  # 'KP', 'KP+',
            'RE', 'CM', 'OO', 'GMA', 'LCS',
            'DREP', 'drepm', 'SEP', 'OEP', 'PEP',
            # 'PEP+', 'pepre', 'pepr+'
        ]
        self._set_pru_late = [
            'MRMC-MRMR', 'MRMC-MRMC', 'MRMC-ALL',
            'MRMREP', 'mRMR-EP', 'Disc-EP',
            'TSP-AP', 'TSP-DP', 'TSP-AP+DP', 'TSP-DP+AP',
            # "TSPrev-DP", "TSPrev-AD", "TSPrev-DA",
        ]

        self._lam = lam
        self._epsilon = epsilon
        self._rho = rho  # nb_pru/nb_cls
        self._alpha = alpha  # 1-lam
        self._L_steps = L
        self._R_steps = R

        self._rho = max(rho, float(nb_pru) / nb_cls)
        self._alpha = max(alpha, 1. - lam)
        # self._logger = logger

    @property
    def lam(self):
        return self._lam

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def rho(self):
        return self._rho

    @property
    def alpha(self):
        return self._alpha

    @property
    def L(self):
        return self._L_steps

    @property
    def R(self):
        return self._R_steps

    '''
    @property
    def logger(self):
        return self._logger
    @logger.setter
    def logger(self, value):
        self._logger = value
    '''

    def prepare_trial(self):
        length = len(self._set_pru_name + self._set_pru_late)
        csv_row_1 = unique_column(8 + 26 + 24 * (length + 5) - 1)

        csv_row_2c_a = [
            'Ensem:trn'] + [''] * 10 + ['Ensem:tst'] + [''] * 10 + [
            'Ensem']
        csv_row_2c_b, csv_row_2c_c = [], []
        for name_pru in [  # , 'POEPAF.2'
                'EPAF-C', 'EPAF-D.2', 'EPAF-D.3', 'POEPAF.1']:
            csv_row_2c_b.extend(
                ['{} :trn'.format(name_pru)] + [''] * 10 +
                ['{} :tst'.format(name_pru)] + [''] * 10 + [name_pru, ''])
        for name_pru in self._set_pru_name + self._set_pru_late:
            csv_row_2c_c.extend(
                ['{} :trn'.format(name_pru)] + [''] * 10 +
                ['{} :tst'.format(name_pru)] + [''] * 10 + [name_pru, ''])
        csv_row_2c = csv_row_2c_a + csv_row_2c_b + csv_row_2c_c
        del csv_row_2c_a, csv_row_2c_b, csv_row_2c_c

        csv_row_3c_a = [
            'Acc', 'P', 'R', 'F1',
            'L_fair(MV)', 'L_acc(MV)', 'L()',
            'sub_fair', 'E(L_acc)', 'L(MVrho)', 'E(L_fair)'
        ] * 2 + ['ut.calc']  # 11*2+1 =23
        csv_row_3c_b = [
            'Acc', 'P', 'R', 'F1',
            'G1', 'G2', '', '', '', 'L_obj(MV)', ''
        ] * 2 + ['ut', 'ut.calc']  # 'ut.-pru'  11*2+2 =24
        csv_row_3c = csv_row_3c_a + csv_row_3c_b * (length + 4)
        del csv_row_3c_b, csv_row_3c_a

        return csv_row_1, csv_row_2c, csv_row_3c

    def schedule_content(self,
                         y_trn, y_insp, yq_insp,
                         y_tst, y_pred, yq_pred,
                         positive_label, X=None, indices=None):
        y_insp = [j.tolist() for j in y_insp]
        y_pred = [j.tolist() for j in y_pred]
        yq_insp = [j.tolist() for j in yq_insp]
        yq_pred = [j.tolist() for j in yq_pred]
        # logging.info("")
        # elegant_print("", self._logger)

        since = time.time()
        tmp_trn = self.calculate_fair_quality_pruned(
            y_trn, y_insp, yq_insp, self._weight, positive_label,
            self._lam)
        tmp_tst = self.calculate_fair_quality_pruned(
            y_tst, y_pred, yq_pred, self._weight, positive_label,
            self._lam)
        ut_ens = (time.time() - since) / 60
        # since = time.time()
        ans_ens = tmp_trn + tmp_tst + [ut_ens]  # 22+1

        tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, y_tst, y_pred, yq_pred,
            "EPAF-C", positive_label)
        ans_ens.extend(tmp_pru)  # +24
        tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, y_tst, y_pred, yq_pred,
            "EPAF-D", positive_label, n_m=2)
        ans_ens.extend(tmp_pru)  # +24
        tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, y_tst, y_pred, yq_pred,
            "EPAF-D", positive_label, n_m=3)
        ans_ens.extend(tmp_pru)  # +24
        tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, y_tst, y_pred, yq_pred,
            "POEPAF", positive_label, dist=1)
        ans_ens.extend(tmp_pru)  # +24
        # ans_ens: 23+24*(5-1)

        ans_pru = []
        for name_pru in self._set_pru_name + self._set_pru_late:
            tmp_pru = self.compare_prune_val(
                y_trn, y_insp, yq_insp, y_tst, y_pred, yq_pred,
                name_pru, positive_label, X, indices)
            ans_pru.extend(tmp_pru)
        del y_insp, y_pred, yq_insp, yq_pred, since
        return ans_ens + ans_pru

    def propose_prune_val(self,
                          y_trn, y_insp, yq_insp,
                          y_tst, y_pred, yq_pred, 
                          name_pru, positive_label, dist=1, n_m=2):
        since = time.time()
        if name_pru == 'POEPAF':
            H = Pareto_Optimal_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam, dist)
        elif name_pru == 'EPAF-C':
            H = Centralised_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam)
        elif name_pru == 'EPAF-D':
            H = Distributed_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam, n_m)
        else:
            raise ValueError("No such pruning proposed `{}`".format(
                name_pru))

        tim_elapsed = time.time() - since
        since = time.time()

        ys_insp = np.array(y_insp)[H].tolist()
        ys_pred = np.array(y_pred)[H].tolist()
        yr_insp = np.array(yq_insp)[H].tolist()
        yr_pred = np.array(yq_pred)[H].tolist()
        coef = np.array(self._weight)[H].tolist()

        tmp_trn = self.calculate_fair_quality_pruned(
            y_trn, ys_insp, yr_insp, coef, positive_label, self._lam)
        tmp_tst = self.calculate_fair_quality_pruned(
            y_tst, ys_pred, yr_pred, coef, positive_label, self._lam)
        tmp_pru = [tim_elapsed / 60, (time.time() - since) / 60]
        del ys_insp, ys_pred, yr_insp, yr_pred, coef
        return tmp_trn + tmp_tst + tmp_pru

    def compare_prune_val(self,
                          y_trn, y_insp, yq_insp,
                          y_tst, y_pred, yq_pred,
                          name_pru, positive_label,
                          X=None, indices=None):
        since = time.time()
        ys_insp, P, seq = self.pruning_baseline(
            y_trn, y_insp, name_pru, self._epsilon, self._rho,
            self._alpha, self._L_steps, self._R_steps, X, indices)
        tim_elapsed = time.time() - since
        since = time.time()

        ys_pred = np.array(y_pred)[P].tolist()
        yr_insp = np.array(yq_insp)[P].tolist()
        yr_pred = np.array(yq_pred)[P].tolist()
        coef = np.array(self._weight)[P].tolist()

        tmp_trn = self.calculate_fair_quality_pruned(
            y_trn, ys_insp, yr_insp, coef, positive_label, self._lam)
        tmp_tst = self.calculate_fair_quality_pruned(
            y_tst, ys_pred, yr_pred, coef, positive_label, self._lam)
        tmp_pru = [tim_elapsed / 60, (time.time() - since) / 60]
        del ys_insp, ys_pred, yr_insp, yr_pred, coef
        return tmp_trn + tmp_tst + tmp_pru  # 11*2+2 =24


# Trial Part B.  Revised
# -------------------------------------


class PartG_ImprovedPruning(PartD_ImprovedPruning):
    def __init__(self, name_ens, abbr_cls, nb_cls, nb_pru,
                 lam, epsilon, rho, alpha, L, R):
        super().__init__(name_ens, abbr_cls, nb_cls, nb_pru,
                         lam, epsilon, rho, alpha, L, R)
        self._set_pru_name = [
            'ES', 'KL', 'KPk', 'KPz', 'RE', 'CM', 'OO',
            'GMA', 'LCS', 'DREP', 'SEP', 'OEP', 'PEP',
        ]
        self._set_pru_late = [
            'MRMC-MRMR', 'MRMC-MRMC', 'MRMREP',
            'Disc-EP', 'TSP-AP', 'TSP-DP',
        ]
        self._set_pru_prop = ['EPAF-C', 'EPAF-D', 'POEPAF', 'POPEP']

    def prepare_trial(self):
        length = len(self._set_pru_name) + len(self._set_pru_late)
        # csv_row_1 = unique_column(8 + 26 + 25 * (length + 6) - 1)
        csv_row_1 = unique_column(8 + 26 + 23 * (1 + 5 + length))

        csv_row_2c_a = ['', 'Ensem', ''] + [
            'Ensem:trn'] + [''] * 9 + ['Ensem:tst'] + [''] * 9
        csv_row_2c_b, csv_row_2c_c = [], []
        for name_pru in [
                'EPAF-C', 'EPAF-D.2', 'EPAF-D.3', 'POEPAF.1', 'POPEP']:
            csv_row_2c_b.extend(
                [name_pru, '', '']
                + ['{} :trn'.format(name_pru)] + [''] * 9
                + ['{} :tst'.format(name_pru)] + [''] * 9)
        for name_pru in self._set_pru_name + self._set_pru_late:
            csv_row_2c_c.extend(
                [name_pru, '', '']
                + ['{} :trn'.format(name_pru)] + [''] * 9
                + ['{} :tst'.format(name_pru)] + [''] * 9)
        csv_row_2c = csv_row_2c_a + csv_row_2c_b + csv_row_2c_c
        del csv_row_2c_a, csv_row_2c_b, csv_row_2c_c

        csv_row_3c_a = ['', 'ut.calc', 'us'] + [
            'La(MV)', 'Lf(MV)', 'L(MV)', 'E[La(f)]', 'E[Lf(f,fp)]',
            'E[Lf(f)]', 'Acc', 'P', 'R', 'F1'] * 2  # 'L()', 'sub_fair',
        csv_row_3c_b = ['ut', 'ut.calc', 'us'] + [
            'G1', 'G2', 'Lo(MV)', '', '', '', 'Acc', 'P', 'R', 'F1'] * 2
        csv_row_3c = csv_row_3c_a + csv_row_3c_b * (5 + length)
        del csv_row_3c_a, csv_row_3c_b  # 23*(1+5+leng)

        csv_row_4c_a = ['sens', '', ''] + [
            'unaware', '', 'GF-one', '', 'GF-two', '', 'GF-thr', '',
            'manual', ''] * 2
        csv_row_4c_b = ['', '', ''] + ([
            'gones', 'gzero'] + [''] * 6 + ['gones', 'gzero']) * 2
        csv_row_4c = csv_row_4c_a + csv_row_4c_b * (5 + length)
        del csv_row_4c_a, csv_row_4c_b

        return csv_row_1, csv_row_2c, [csv_row_3c, csv_row_4c]

    def schedule_content(self,
                         y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                         y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                         fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                         positive_label, X=None, indices=None):
        """
          y_trn /tst  : list,               shape= (nb_y,)
          y_insp/pred : list of np.ndarray, shape= (nb_cls, nb_y)
         yq_insp/pred : list of np.ndarray, shape= (nb_cls, nb_y)
         tag_trn/tst  : list of np.ndarray, shape= (1/2, nb_y)
          jt_trn/tst  : list of np.ndarray, shape= (0/1, nb_y)
        fens_trn/tst  : list,               shape= (nb_y,)
        fqtb_trn/tst  : list,               shape= (nb_y,)
        positive_label: scalar
        """
        y_insp = [j.tolist() for j in y_insp]
        y_pred = [j.tolist() for j in y_pred]
        yq_insp = [j.tolist() for j in yq_insp]
        yq_pred = [j.tolist() for j in yq_pred]

        since = time.time()
        tmp_trn, trn_A1, trn_A2, trn_Jt = self.calc_simplified_fair_quality(
            y_trn, y_insp, yq_insp, fens_trn, fqtb_trn, tag_trn, jt_trn,
            self._weight, self._lam, positive_label)
        tmp_tst, tst_A1, tst_A2, tst_Jt = self.calc_simplified_fair_quality(
            y_tst, y_pred, yq_pred, fens_tst, fqtb_tst, tag_tst, jt_tst,
            self._weight, self._lam, positive_label)
        # ut_ens = (time.time() - since) / 60
        # us_ens = len(self._weight)
        # ans_ens = [ut_ens, us_ens]  # 2
        ans_ens = ['', (time.time() - since) / 60, len(self._weight)]  # 1+2
        ans_ens.extend(tmp_trn + tmp_tst)  # 3+10*2 =23

        ens_A1 = ['A1', '', ''] + trn_A1 + tst_A1  # 3+10*2 =23
        ens_A2 = ['A2', '', ''] + trn_A2 + tst_A2  # 3+10*2 =23
        ens_Jt = ['Jt', '', ''] + trn_Jt + tst_Jt  # 3+10*2 =23
        del trn_A1, trn_A2, trn_Jt
        del tst_A1, tst_A2, tst_Jt

        # ans_ens.extend([''] * 30 + tmp_pru)
        # ens_A1.extend([''] * 3 + tmp_A1)
        # ens_A2.extend([''] * 3 + tmp_A2)
        # ens_Jt.extend([''] * 3 + tmp_Jt)
        # ans_ens.extend(tmp_pru)  # +13
        # ens_A1.extend(tmp_A1)  # +10*2
        # ens_A2.extend(tmp_A2)  # +10*2
        # ens_Jt.extend(tmp_Jt)  # +10*2

        # Proposed
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-C", positive_label)
        ans_ens.extend(tmp_pru)  # +23
        ens_A1.extend([''] * 3 + tmp_A1)  # +3+10*2
        ens_A2.extend([''] * 3 + tmp_A2)  # +3+10*2
        ens_Jt.extend([''] * 3 + tmp_Jt)  # +3+10*2
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-D", positive_label, n_m=2)
        ans_ens.extend(tmp_pru)  # +23
        ens_A1.extend([''] * 3 + tmp_A1)  # +3+10*2
        ens_A2.extend([''] * 3 + tmp_A2)  # +3+10*2
        ens_Jt.extend([''] * 3 + tmp_Jt)  # +3+10*2
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-D", positive_label, n_m=3)
        ans_ens.extend(tmp_pru)  # +23
        ens_A1.extend([''] * 3 + tmp_A1)  # +3+10*2
        ens_A2.extend([''] * 3 + tmp_A2)  # +3+10*2
        ens_Jt.extend([''] * 3 + tmp_Jt)  # +3+10*2
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "POEPAF", positive_label, dist=1)
        ans_ens.extend(tmp_pru)  # +23
        ens_A1.extend([''] * 3 + tmp_A1)  # +3+10*2
        ens_A2.extend([''] * 3 + tmp_A2)  # +3+10*2
        ens_Jt.extend([''] * 3 + tmp_Jt)  # +3+10*2
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "POPEP", positive_label)
        ans_ens.extend(tmp_pru)  # +23
        ens_A1.extend([''] * 3 + tmp_A1)  # +3+10*2
        ens_A2.extend([''] * 3 + tmp_A2)  # +3+10*2
        ens_Jt.extend([''] * 3 + tmp_Jt)  # +3+10*2

        # Compared
        for name_pru in self._set_pru_name + self._set_pru_late:
            tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.compare_prune_val(
                y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                name_pru, positive_label, X, indices)
            ans_ens.extend(tmp_pru)  # +23
            ens_A1.extend([''] * 3 + tmp_A1)  # +3+10*2
            ens_A2.extend([''] * 3 + tmp_A2)  # +3+10*2
            ens_Jt.extend([''] * 3 + tmp_Jt)  # +3+10*2

        del y_insp, y_pred, yq_insp, yq_pred
        del tmp_A1, tmp_A2, tmp_Jt, tmp_pru, since
        # TEST  # size=(4, 23*(1+5+length))
        # return [ans_ens, ens_A1, ens_A2, ens_Jt]
        return ans_ens, ens_A1, ens_A2, ens_Jt

    # def calculate_fair_quality_pruned(self, y, ys, yr, wgt,
    #                                   lam=.5, pos=1, idx_priv=tuple()):
    #   pass  # aka. to be replaced with/by the next function

    def calc_simplified_fair_quality(self, y, ys, yr, fens, fqtb, tag, jt,
                                     wgt, lam=.5, pos_val=1):
        # abbreviated, reduced results to output, in order to save time
        """
           y     : list,         shape= (nb_y,)
          ys,  yr: list of list, shape= (nb_cls, nb_y)
         wgt     : list,         shape= (nb_cls,)
        fens,fqtb: list,         shape= (nb_y,)

                lam   : scalar
        positive_label: scalar
           ptb_priv[i]: np.ndarray of boolean, shape=(nb_y,)
        """
        # since = time.time()
        tmp = self.calc_reduced_fair_mvrho(
            y, ys, yr, fens, fqtb, wgt, lam, pos_val)

        y = np.array(y)
        ys, yr = np.array(ys), np.array(yr)
        fens, fqtb = np.array(fens), np.array(fqtb)

        ans_A1 = self.calc_reduced_fair_gather(y, fens, pos_val, tag[0])
        if not jt:
            ans_A2 = [''] * 10
            ans_Jt = [''] * 10
            return tmp, ans_A1, ans_A2, ans_Jt

        ans_A2 = self.calc_reduced_fair_gather(y, fens, pos_val, tag[1])
        ans_Jt = self.calc_reduced_fair_gather(y, fens, pos_val, jt[0])
        del y, ys, yr, fens, fqtb
        return tmp, ans_A1, ans_A2, ans_Jt  # 10,10*3 rows

    def calc_reduced_fair_mvrho(self, y, yt, yqtb, fens, fqtb,
                                wgt, lam=.5, pos_val=1):
        """
        y        : np.ndarray, shape= (nb_y,)
        yt  ,yqtb: np.ndarray, shape= (nb_cls, nb_y)
        fens,fqtb: np.ndarray, shape= (nb_y,)
              wgt: list,       shape= (nb_cls,)
        """
        # nb_cls = len(wgt)  # might not be self._weight if pruned

        # for below, less is better
        ans_fair = []
        ans_fair.append(hat_L_loss(fens, y))
        ans_fair.append(hat_L_fair(fens, fqtb))
        sub_no1 = E_rho_L_loss_f(yt, y, wgt)
        sub_no2 = Erho_sup_L_fair(yt, yqtb, wgt)
        G_mv = (sub_no1, sub_no2)
        ans_fair.append(_POAF_calc_eval(G_mv, lam))
        ans_fair.extend(G_mv)
        # return ans_fair  # 3+2 =5

        Acc, (_, p, r, f, _, _, _, _, _, _, _) = \
            self.calculate_sub_ensemble_metrics(y, fens, pos_val)
        # return ans_fair + [Acc, a, p, r, f]  # 5+5 =10
        ans_fair.append(E_rho_L_fair_f(yt, yqtb, wgt))
        del sub_no1, sub_no2, G_mv
        return ans_fair + [Acc, p, r, f]  # 6+4 =10

    def calc_reduced_fair_gather(self, y, fens,
                                 pos_val=1, idx_priv=tuple()):
        """
        y  :            np.ndarray
        hx :            np.ndarray
        positive_label: scalar
           ptb_priv[i]: np.ndarray of boolean, shape=(nb_y,)
                lam   : scalar
        """

        # ens_fair = self.calc_reduced_fair_group(
        #     y, yt, yqtb, fens, fq, wgt, pos_val, idx_priv)
        # def calc_reduced_fair_group(self, y, yt, yqtb, fens, fqtb,
        #                           wgt, pos_val, idx_priv):

        g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
            marginalised_pd_mat(y, fens, pos_val, idx_priv)
        cmp_fair = []
        cmp_fair.extend(unpriv_unaware(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_group_one(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_group_two(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_group_thr(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_manual(gones_Cm, gzero_Cm))
        # for above, more is better
        # 用这几个数值对比作图，看在敏感属性上会有多大区别

        # return cmp_fair + ans_fair  # 5*2+3+2 =15
        del g1_Cij, g0_Cij, gones_Cm, gzero_Cm
        return cmp_fair  # 5*2 =10

    def propose_prune_val(self,
                          y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                          y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                          name_pru, positive_label, dist=1, n_m=2):
        since = time.time()
        if name_pru == 'POEPAF':
            H = Pareto_Optimal_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam, dist)
        elif name_pru == 'EPAF-C':
            H = Centralised_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam)
        elif name_pru == 'EPAF-D':
            H = Distributed_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam, n_m)
        elif name_pru == 'POPEP':
            H = POAF_PEP(y_trn, y_insp, yq_insp, self._weight,
                         self._lam, self._nb_pru)
        else:
            raise ValueError("No such pruning proposed `{}`".format(
                name_pru))

        tim_elapsed = time.time() - since
        since = time.time()

        ys_insp = np.array(y_insp)[H].tolist()
        ys_pred = np.array(y_pred)[H].tolist()
        yr_insp = np.array(yq_insp)[H].tolist()
        yr_pred = np.array(yq_pred)[H].tolist()
        coef = np.array(self._weight)[H].tolist()

        fens_trn = self.majority_vote(y_trn, ys_insp, coef)
        fens_tst = self.majority_vote(y_tst, ys_pred, coef)
        fqtb_trn = self.majority_vote(y_trn, yr_insp, coef)
        fqtb_tst = self.majority_vote(y_tst, yr_pred, coef)

        (tmp_trn, trn_A1, trn_A2,
         trn_Jt) = self.calc_simplified_fair_quality(
            y_trn, ys_insp, yr_insp, fens_trn, fqtb_trn, tag_trn,
             jt_trn, coef, self._lam, positive_label)
        (tmp_tst, tst_A1, tst_A2,
         tst_Jt) = self.calc_simplified_fair_quality(
            y_tst, ys_pred, yr_pred, fens_tst, fqtb_tst, tag_tst,
             jt_tst, coef, self._lam, positive_label)
        del ys_insp, yr_insp, fens_trn, fqtb_trn
        del ys_pred, yr_pred, fens_tst, fqtb_tst
        del coef

        ans_A1 = trn_A1 + tst_A1  # 10*2 =20
        ans_A2 = trn_A2 + tst_A2  # 10*2 =20
        ans_Jt = trn_Jt + tst_Jt  # 10*2 =20
        del trn_A1, trn_A2, trn_Jt
        del tst_A1, tst_A2, tst_Jt

        tmp_pru = [tim_elapsed / 60, (time.time() - since) / 60, len(H)]
        tmp_pru.extend(tmp_trn + tmp_tst)  # 3+10*2 =23
        del tmp_trn, tmp_tst, since, tim_elapsed
        return ans_A1, ans_A2, ans_Jt, tmp_pru

    def compare_prune_val(self,
                          y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                          y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                          name_pru, positive_label, X=None, indices=None):
        since = time.time()
        ys_insp, P, seq = self.pruning_baseline(
            y_trn, y_insp, name_pru, self._epsilon, self._rho,
            self._alpha, self._L_steps, self._R_steps, X, indices)
        tim_elapsed = time.time() - since
        since = time.time()

        ys_pred = np.array(y_pred)[P].tolist()
        yr_insp = np.array(yq_insp)[P].tolist()
        yr_pred = np.array(yq_pred)[P].tolist()
        coef = np.array(self._weight)[P].tolist()

        fens_trn = self.majority_vote(y_trn, ys_insp, coef)
        fens_tst = self.majority_vote(y_tst, ys_pred, coef)
        fqtb_trn = self.majority_vote(y_trn, yr_insp, coef)
        fqtb_tst = self.majority_vote(y_tst, yr_pred, coef)

        (tmp_trn, trn_A1, trn_A2,
         trn_Jt) = self.calc_simplified_fair_quality(
            y_trn, ys_insp, yr_insp, fens_trn, fqtb_trn, tag_trn,
             jt_trn, coef, self._lam, positive_label)
        (tmp_tst, tst_A1, tst_A2, tst_Jt) = self.calc_simplified_fair_quality(
            y_tst, ys_pred, yr_pred, fens_tst, fqtb_tst, tag_tst,
            jt_tst, coef, self._lam, positive_label)
        del ys_insp, yr_insp, fens_trn, fqtb_trn
        del ys_pred, yr_pred, fens_tst, fqtb_tst
        del coef

        ans_A1 = trn_A1 + tst_A1  # 10*2 =20
        ans_A2 = trn_A2 + tst_A2  # 10*2 =20
        ans_Jt = trn_Jt + tst_Jt  # 10*2 =20
        del trn_A1, trn_A2, trn_Jt
        del tst_A1, tst_A2, tst_Jt

        tmp_pru = [tim_elapsed / 60, (time.time() - since) / 60, len(seq)]
        tmp_pru.extend(tmp_trn + tmp_tst)  # 3+10*2 =23
        del tmp_trn, tmp_tst, since, tim_elapsed
        return ans_A1, ans_A2, ans_Jt, tmp_pru


# Compared to PartG:
#   Complement H/seq to return

class PartH_ImprovedPruning(PartG_ImprovedPruning):
    def __init__(self, name_ens, abbr_cls, nb_cls, nb_pru,
                 lam, epsilon, rho, alpha, L, R):
        super().__init__(name_ens, abbr_cls, nb_cls, nb_pru,
                         lam, epsilon, rho, alpha, L, R)

    def prepare_trial(self):
        length = len(self._set_pru_name) + len(self._set_pru_late)
        num_cmp = 1 + 5 + length
        csv_row_1 = unique_column(8 + 26 + (23 + 1) * num_cmp - 1)

        csv_row_2c_a = ['', 'Ensem', ''] + [
            'Ensem:trn'] + [''] * 9 + ['Ensem:tst'] + [''] * 9
        csv_row_2c_b, csv_row_2c_c = [], []
        for name_pru in [
                'EPAF-C', 'EPAF-D.2', 'EPAF-D.3', 'POEPAF.1', 'POPEP']:
            csv_row_2c_b.extend(
                [name_pru, '', '', '']
                + ['{} :trn'.format(name_pru)] + [''] * 9
                + ['{} :tst'.format(name_pru)] + [''] * 9)
        for name_pru in self._set_pru_name + self._set_pru_late:
            csv_row_2c_c.extend(
                [name_pru, '', '', '']
                + ['{} :trn'.format(name_pru)] + [''] * 9
                + ['{} :tst'.format(name_pru)] + [''] * 9)
        csv_row_2c = csv_row_2c_a + csv_row_2c_b + csv_row_2c_c
        del csv_row_2c_a, csv_row_2c_b, csv_row_2c_c  # 23+24*5+24*?

        csv_row_3c_a = ['', 'ut.calc', 'us'] + [
            'La(MV)', 'Lf(MV)', 'Lo(MV)', 'E[La(f)]', 'E[Lf(f,fp)]',
            'E[Lf(f)]', 'Acc', 'P', 'R', 'F1'] * 2
        csv_row_3c_b = ['ut', 'ut.calc', 'us', 'seq'] + [
            'G1', 'G2', 'L(MV)', '', '', '', 'Acc', 'P', 'R', 'F1'] * 2
        csv_row_3c = csv_row_3c_a + csv_row_3c_b * (num_cmp - 1)
        del csv_row_3c_a, csv_row_3c_b  # 23+24*(5+?)

        csv_row_4c_a = ['sens', '', ''] + [
            'unaware:g1', 'unaware:g0',
            'GF-one :g1', 'GF-one :g0',
            'GF-two :g1', 'GF-two :g0',
            'GF-thr :g1', 'GF-thr :g0',
            'manual :g1', 'manual :g0'] * 2
        csv_row_4c_b = ['', '', '', ''] + ['g1', 'g0'] * 5 * 2
        csv_row_4c = csv_row_4c_a + csv_row_4c_b * (num_cmp - 1)
        del csv_row_4c_a, csv_row_4c_b  # 23+24*(5+?)

        return csv_row_1, csv_row_2c, [csv_row_3c, csv_row_4c]

    def schedule_content(self,
                         y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                         y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                         fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                         positive_label, X=None, indices=None):
        y_insp = [j.tolist() for j in y_insp]
        y_pred = [j.tolist() for j in y_pred]
        yq_insp = [j.tolist() for j in yq_insp]
        yq_pred = [j.tolist() for j in yq_pred]

        since = time.time()
        (tmp_trn, trn_A1, trn_A2,
         trn_Jt) = self.calc_simplified_fair_quality(
            y_trn, y_insp, yq_insp, fens_trn, fqtb_trn, tag_trn,
            jt_trn, self._weight, self._lam, positive_label)
        (tmp_tst, tst_A1, tst_A2,
         tst_Jt) = self.calc_simplified_fair_quality(
            y_tst, y_pred, yq_pred, fens_tst, fqtb_tst, tag_tst,
            jt_tst, self._weight, self._lam, positive_label)
        ans_ens = ['', (time.time() - since) / 60, len(self._weight)]
        ans_ens.extend(tmp_trn + tmp_tst)  # 3+10*2 =23

        ens_A1 = ['A1', '', ''] + trn_A1 + tst_A1  # 3+10*2 =23
        ens_A2 = ['A2', '', ''] + trn_A2 + tst_A2  # 3+10*2 =23
        ens_Jt = ['Jt', '', ''] + trn_Jt + tst_Jt  # 3+10*2 =23
        del trn_A1, trn_A2, trn_Jt, tmp_trn
        del tst_A1, tst_A2, tst_Jt, tmp_tst

        # Proposed
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-C", positive_label)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend([''] * 4 + tmp_A1)  # +4+10*2
        ens_A2.extend([''] * 4 + tmp_A2)  # +4+10*2
        ens_Jt.extend([''] * 4 + tmp_Jt)  # +4+10*2
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-D", positive_label, n_m=2)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend([''] * 4 + tmp_A1)  # +4+10*2
        ens_A2.extend([''] * 4 + tmp_A2)  # +4+10*2
        ens_Jt.extend([''] * 4 + tmp_Jt)  # +4+10*2
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-D", positive_label, n_m=3)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend([''] * 4 + tmp_A1)  # +4+10*2
        ens_A2.extend([''] * 4 + tmp_A2)  # +4+10*2
        ens_Jt.extend([''] * 4 + tmp_Jt)  # +4+10*2
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "POEPAF", positive_label, dist=1)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend([''] * 4 + tmp_A1)  # +4+10*2
        ens_A2.extend([''] * 4 + tmp_A2)  # +4+10*2
        ens_Jt.extend([''] * 4 + tmp_Jt)  # +4+10*2
        tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "POPEP", positive_label)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend([''] * 4 + tmp_A1)  # +4+10*2
        ens_A2.extend([''] * 4 + tmp_A2)  # +4+10*2
        ens_Jt.extend([''] * 4 + tmp_Jt)  # +4+10*2

        # Compared
        for name_pru in self._set_pru_name + self._set_pru_late:
            tmp_A1, tmp_A2, tmp_Jt, tmp_pru = self.compare_prune_val(
                y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                name_pru, positive_label, X, indices)
            ans_ens.extend(tmp_pru)  # +24
            ens_A1.extend([''] * 4 + tmp_A1)  # +4+10*2
            ens_A2.extend([''] * 4 + tmp_A2)  # +4+10*2
            ens_Jt.extend([''] * 4 + tmp_Jt)  # +4+10*2

        del y_insp, y_pred, yq_insp, yq_pred
        del tmp_A1, tmp_A2, tmp_Jt, tmp_pru
        return ans_ens, ens_A1, ens_A2, ens_Jt

    def calc_abbreviated_pru_sub_qlt(
            self,
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            positive_label, H):
        ys_insp = np.array(y_insp)[H].tolist()
        ys_pred = np.array(y_pred)[H].tolist()
        yr_insp = np.array(yq_insp)[H].tolist()
        yr_pred = np.array(yq_pred)[H].tolist()
        coef = np.array(self._weight)[H].tolist()

        fens_trn = self.majority_vote(y_trn, ys_insp, coef)
        fens_tst = self.majority_vote(y_tst, ys_pred, coef)
        fqtb_trn = self.majority_vote(y_trn, yr_insp, coef)
        fqtb_tst = self.majority_vote(y_tst, yr_pred, coef)

        (tmp_trn, trn_A1, trn_A2,
         trn_Jt) = self.calc_simplified_fair_quality(
            y_trn, ys_insp, yr_insp, fens_trn, fqtb_trn, tag_trn,
            jt_trn, coef, self._lam, positive_label)
        (tmp_tst, tst_A1, tst_A2,
         tst_Jt) = self.calc_simplified_fair_quality(
            y_tst, ys_pred, yr_pred, fens_tst, fqtb_tst, tag_tst,
            jt_tst, coef, self._lam, positive_label)
        del ys_insp, yr_insp, fens_trn, fqtb_trn
        del ys_pred, yr_pred, fens_tst, fqtb_tst
        del coef

        ans_A1 = trn_A1 + tst_A1  # 10*2 =20
        ans_A2 = trn_A2 + tst_A2  # 10*2 =20
        ans_Jt = trn_Jt + tst_Jt  # 10*2 =20
        del trn_A1, trn_A2, trn_Jt
        del tst_A1, tst_A2, tst_Jt

        ans_pru = tmp_trn + tmp_tst  # 10*2 =20
        del tmp_trn, tmp_tst
        return ans_A1, ans_A2, ans_Jt, ans_pru

    def propose_prune_val(self,
                          y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                          y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                          name_pru, positive_label, dist=1, n_m=2):
        since = time.time()
        if name_pru == 'POEPAF':
            H = Pareto_Optimal_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam, dist)
        elif name_pru == 'EPAF-C':
            H = Centralised_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam)
        elif name_pru == 'EPAF-D':
            H = Distributed_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam, n_m)
        elif name_pru == 'POPEP':
            H = POAF_PEP(y_trn, y_insp, yq_insp, self._weight,
                         self._lam, self._nb_pru)
        else:
            raise ValueError("No such pruning proposed `{}`".format(name_pru))

        tim_elapsed = time.time() - since
        since = time.time()

        (ans_A1, ans_A2, ans_Jt,
         ans_pru) = self.calc_abbreviated_pru_sub_qlt(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            positive_label, H)
        tmp_pru = [tim_elapsed / 60, (time.time() - since) / 60,
                   len(H), H]
        tmp_pru.extend(ans_pru)  # 4+20 =24

        del since, tim_elapsed, ans_pru
        return ans_A1, ans_A2, ans_Jt, tmp_pru

    def compare_prune_val(self,
                          y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                          y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                          name_pru, positive_label,
                          X=None, indices=None):
        since = time.time()
        _, P, seq = self.pruning_baseline(
            y_trn, y_insp, name_pru, self._epsilon, self._rho,
            self._alpha, self._L_steps, self._R_steps, X, indices)
        tim_elapsed = time.time() - since
        since = time.time()

        (ans_A1, ans_A2, ans_Jt,
         ans_pru) = self.calc_abbreviated_pru_sub_qlt(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            positive_label, seq)  # P
        tmp_pru = [
            tim_elapsed / 60, (time.time() - since) / 60,
            len(seq), seq]
        tmp_pru.extend(ans_pru)  # 4+20 =24

        del since, tim_elapsed, ans_pru
        return ans_A1, ans_A2, ans_Jt, tmp_pru


# -------------------------------------


# Effect of $\lambda$ value
# -------------------------------------


class PartI_LambdaEffect(EnsembleSetup):
    def __init__(self, name_ens, abbr_cls, nb_cls, nb_pru):
        super().__init__(name_ens, abbr_cls, nb_cls, nb_pru)

    @property
    def abbr_cls(self):
        return self._abbr_cls

    def calc_reduced_fair_mvrho(self, y, yt, yqtb, fens, fqtb,
                                wgt, lam=.5, pos_val=1):
        ans_fair = [hat_L_loss(fens, y),
                    hat_L_fair(fens, fqtb)]
        G_mv = (E_rho_L_loss_f(yt, y, wgt),
                Erho_sup_L_fair(yt, yqtb, wgt))
        ans_fair.append(_POAF_calc_eval(G_mv, lam))
        ans_fair.extend(G_mv)  # 2+1+2 =5
        ans_fair.append(E_rho_L_fair_f(yt, yqtb, wgt))
        # for above, less is better
        # for below, more is better
        Acc, (a, p, r, f, _, _, _, _, _, _, _) = \
            self.calculate_sub_ensemble_metrics(y, fens, pos_val)
        del G_mv, a
        return ans_fair + [Acc, p, r, f]  # 6+4 =10

    def calc_reduced_fair_gather(self, y, fens, pos_val=1,
                                 idx_priv=tuple()):
        g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
            marginalised_pd_mat(y, fens, pos_val, idx_priv)
        cmp_fair = []
        cmp_fair.extend(unpriv_unaware(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_group_one(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_group_two(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_group_thr(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_manual(gones_Cm, gzero_Cm))
        # for above, more is better
        # 用这几个数值对比作图，看在敏感属性上会有多大区别
        del g1_Cij, g0_Cij, gones_Cm, gzero_Cm
        return cmp_fair  # 5*2 =10

    def calc_simplified_fair_quality(self, y, ys, yr, fens, fqtb,
                                     tag, jt, wgt, lam=.5, pos_val=1):
        # abbreviated, reduced results to output
        """
           y     :  list,         shape= (nb_y,)
          ys,  yr:  list of list, shape= (nb_cls, nb_y)
         wgt     :  list,         shape= (nb_cls,)
        fens,fqtb:  list,         shape= (nb_y,)

                  lam :  scalar
        positive_label:  scalar
           ptb_priv[i]:  np.ndarray of boolean, shape= (nb_y,)
        """
        # aka. def calc_reduced_fair_rho

        tmp = self.calc_reduced_fair_mvrho(
            y, ys, yr, fens, fqtb, wgt, lam, pos_val)
        y = np.array(y)
        ys, yr = np.array(ys), np.array(yr)
        fens, fqtb = np.array(fens), np.array(fqtb)

        ans_A1 = self.calc_reduced_fair_gather(y, fens, pos_val, tag[0])
        if not jt:
            ans_A2 = [''] * 10
            ans_Jt = [''] * 10
            return tmp, ans_A1, ans_A2, ans_Jt
        ans_A2 = self.calc_reduced_fair_gather(y, fens, pos_val, tag[1])
        ans_Jt = self.calc_reduced_fair_gather(y, fens, pos_val, jt[0])

        # aka. def calc_reduced_fair_gather
        del y, ys, yr, fens, fqtb
        return tmp, ans_A1, ans_A2, ans_Jt

    def calc_abbreviated_pru_sub_qlt(self,
                                     y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                                     y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                                     positive_label, H, lam):
        ys_insp = np.array(y_insp)[H].tolist()
        ys_pred = np.array(y_pred)[H].tolist()
        yr_insp = np.array(yq_insp)[H].tolist()
        yr_pred = np.array(yq_pred)[H].tolist()
        coef = np.array(self._weight)[H].tolist()

        fens_trn = self.majority_vote(y_trn, ys_insp, coef)
        fens_tst = self.majority_vote(y_tst, ys_pred, coef)
        fqtb_trn = self.majority_vote(y_trn, yr_insp, coef)
        fqtb_tst = self.majority_vote(y_tst, yr_pred, coef)

        (tmp_trn, trn_A1, trn_A2,
         trn_Jt) = self.calc_simplified_fair_quality(
            y_trn, ys_insp, yr_insp, fens_trn, fqtb_trn, tag_trn,
            jt_trn, coef, lam, positive_label)
        (tmp_tst, tst_A1, tst_A2,
         tst_Jt) = self.calc_simplified_fair_quality(
            y_tst, ys_pred, yr_pred, fens_tst, fqtb_tst, tag_tst,
            jt_tst, coef, lam, positive_label)
        del ys_insp, yr_insp, fens_trn, fqtb_trn
        del ys_pred, yr_pred, fens_tst, fqtb_tst
        del coef

        ans_A1 = trn_A1 + tst_A1
        ans_A2 = trn_A2 + tst_A2
        ans_Jt = trn_Jt + tst_Jt
        ans_pru = tmp_trn + tmp_tst  # 10*2 =20
        del trn_A1, trn_A2, trn_Jt, tmp_trn
        del tst_A1, tst_A2, tst_Jt, tmp_tst
        return ans_pru, ans_A1, ans_A2, ans_Jt

    def prepare_trial(self):
        length = 24 + 24 * 5
        csv_row_1 = unique_column(8 + 26 + length)

        csv_row_2c_a = ['lam', 'sens', 'Ensem', ''] + [
            'Ensem:trn'] + [''] * 9 + ['Ensem:tst'] + [''] * 9
        csv_row_2c_b = []
        for name_pru in ['EPAF-C', 'EPAF-D.2', 'EPAF-D.3',
                         'POEPAF.1', 'POPEP']:
            csv_row_2c_b.extend(
                [name_pru, '', '', '']
                + ['{} :trn'.format(name_pru)] + [''] * 9
                + ['{} :tst'.format(name_pru)] + [''] * 9)
        csv_row_2c = csv_row_2c_a + csv_row_2c_b
        del csv_row_2c_a, csv_row_2c_b  # 24+24*5

        csv_row_3c_a = ['', '', 'ut.calc', 'us'] + [
            'La(MV)', 'Lf(MV)', 'Lo(MV)', 'E[La(f)]', 'E[Lf(f,fp)]',
            'E[Lf(f)]', 'Acc', 'P', 'R', 'F1'] * 2
        csv_row_3c_b = ['ut', 'ut.calc', 'us', 'seq'] + [
            'G1', 'G2', 'L(MV)', '', '', '', 'Acc',
            'P', 'R', 'F1'] * 2
        csv_row_3c = csv_row_3c_a + csv_row_3c_b * 5
        del csv_row_3c_a, csv_row_3c_b  # 24+24*5

        return csv_row_1, csv_row_2c, csv_row_3c

    def schedule_content(self,
                         y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                         y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                         fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                         positive_label, nb_lam=11, logger=None):
        y_insp = [j.tolist() for j in y_insp]
        y_pred = [j.tolist() for j in y_pred]
        yq_insp = [j.tolist() for j in yq_insp]
        yq_pred = [j.tolist() for j in yq_pred]
        lam_set = np.linspace(0, 1, nb_lam).tolist()
        elegant_print("\tEffect $\lambda$", logger)

        ans_ens, ans_A1, ans_A2, ans_Jt = [], [], [], []
        for lam in lam_set:
            (tmp_ens, tmp_A1, tmp_A2,
             tmp_Jt) = self.schedule_subroutine(
                y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                positive_label, lam)
            elegant_print("\t\tlam = {}".format(lam), logger)
            ans_ens.append(tmp_ens)
            ans_A1.append(tmp_A1)
            ans_A2.append(tmp_A2)
            ans_Jt.append(tmp_Jt)
        return ans_ens, ans_A1, ans_A2, ans_Jt

    def schedule_subroutine(self,
                            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                            fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                            positive_label, lam):
        since = time.time()
        (tmp_trn, trn_A1, trn_A2,
         trn_Jt) = self.calc_simplified_fair_quality(
            y_trn, y_insp, yq_insp, fens_trn, fqtb_trn, tag_trn,
            jt_trn, self._weight, lam, positive_label)
        (tmp_tst, tst_A1, tst_A2,
         tst_Jt) = self.calc_simplified_fair_quality(
            y_tst, y_pred, yq_pred, fens_tst, fqtb_tst, tag_tst,
            jt_tst, self._weight, lam, positive_label)
        ans_ens = [lam, '', (time.time() - since) / 60, len(self._weight)]
        ans_ens.extend(tmp_trn + tmp_tst)  # 4+10*2 =24

        ens_A1 = [lam, 'A1', '', ''] + trn_A1 + tst_A1
        ens_A2 = [lam, 'A2', '', ''] + trn_A2 + tst_A2
        ens_Jt = [lam, 'Jt', '', ''] + trn_Jt + tst_Jt
        del trn_A1, trn_A2, trn_Jt, tmp_trn
        del tst_A1, tst_A2, tst_Jt, tmp_tst

        # Proposed
        tmp_pru, tmp_A1, tmp_A2, tmp_Jt = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-C", positive_label, lam=lam)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend(tmp_A1)
        ens_A2.extend(tmp_A2)
        ens_Jt.extend(tmp_Jt)

        tmp_pru, tmp_A1, tmp_A2, tmp_Jt = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-D", positive_label, lam=lam, n_m=2)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend(tmp_A1)
        ens_A2.extend(tmp_A2)
        ens_Jt.extend(tmp_Jt)

        tmp_pru, tmp_A1, tmp_A2, tmp_Jt = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-D", positive_label, lam=lam, n_m=3)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend(tmp_A1)
        ens_A2.extend(tmp_A2)
        ens_Jt.extend(tmp_Jt)

        tmp_pru, tmp_A1, tmp_A2, tmp_Jt = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "POEPAF", positive_label, lam=lam, dist=1)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend(tmp_A1)
        ens_A2.extend(tmp_A2)
        ens_Jt.extend(tmp_Jt)

        tmp_pru, tmp_A1, tmp_A2, tmp_Jt = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "POPEP", positive_label, lam=lam)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend(tmp_A1)
        ens_A2.extend(tmp_A2)
        ens_Jt.extend(tmp_Jt)

        del tmp_pru, tmp_A1, tmp_A2, tmp_Jt
        return ans_ens, ens_A1, ens_A2, ens_Jt  # each 24+24*5

    def propose_prune_val(self,
                          y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                          y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                          name_pru, positive_label, lam, dist=1, n_m=2):
        since = time.time()
        H = self.pruning_proposed(
            y_trn, y_insp, yq_insp, name_pru, lam, dist, n_m)
        tim_elapsed = time.time() - since
        since = time.time()
        (ans_pru, ans_A1, ans_A2,
         ans_Jt) = self.calc_abbreviated_pru_sub_qlt(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            positive_label, H, lam)
        tmp_pru = [tim_elapsed / 60, (time.time() - since) / 60,
                   len(H), H]

        tmp_pru.extend(ans_pru)  # 4+20 =24
        ans_A1 = [''] * 4 + ans_A1
        ans_A2 = [''] * 4 + ans_A2
        ans_Jt = [''] * 4 + ans_Jt
        del since, tim_elapsed, ans_pru
        return tmp_pru, ans_A1, ans_A2, ans_Jt


class PartJ_LambdaEffect(PartI_LambdaEffect):
    def __init__(self, name_ens, abbr_cls, nb_cls, nb_pru):
        super().__init__(name_ens, abbr_cls, nb_cls, nb_pru)

    def prepare_trial(self):
        csv_row_1 = unique_column(8 + 2 + (24 + 24 * 4))

        csv_row_2c_a = ['lam', 'sens', 'Ensem', ''] + [
            'Ensem:trn'] + [''] * 9 + ['Ensem:tst'] + [''] * 9
        csv_row_2c_b = []
        for name_pru in [  # 'EPAF-D.3',
                'EPAF-C', 'EPAF-D.2', 'POEPAF.1', 'POPEP']:
            csv_row_2c_b.extend(
                [name_pru, '', '', '']
                + ['{} :trn'.format(name_pru)] + [''] * 9
                + ['{} :tst'.format(name_pru)] + [''] * 9)
        csv_row_2c = csv_row_2c_a + csv_row_2c_b
        del csv_row_2c_a, csv_row_2c_b  # 24+24*4

        csv_row_3c_a = ['', '', 'ut.calc', 'us'] + [
            'La(MV)', 'Lf(MV)', 'Lo(MV)', 'E[La(f)]', 'E[Lf(f,fp)]',
            'E[Lf(f)]', 'Acc', 'P', 'R', 'F1'] * 2
        csv_row_3c_b = ['ut', 'ut.calc', 'us', 'seq'] + [
            'G1', 'G2', 'L(MV)', '', '', '', 'Acc', 'P', 'R',
            'F1'] * 2
        csv_row_3c = csv_row_3c_a + csv_row_3c_b * 4
        del csv_row_3c_a, csv_row_3c_b  # 24+24*5

        return csv_row_1, csv_row_2c, csv_row_3c

    def schedule_subroutine(self,
                            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                            fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                            positive_label, lam):
        since = time.time()
        (tmp_trn, trn_A1, trn_A2,
         trn_Jt) = self.calc_simplified_fair_quality(
            y_trn, y_insp, yq_insp, fens_trn, fqtb_trn, tag_trn,
            jt_trn, self._weight, lam, positive_label)
        (tmp_tst, tst_A1, tst_A2,
         tst_Jt) = self.calc_simplified_fair_quality(
            y_tst, y_pred, yq_pred, fens_tst, fqtb_tst, tag_tst,
            jt_tst, self._weight, lam, positive_label)
        ans_ens = [lam, '', (time.time() - since) / 60, len(self._weight)]
        ans_ens.extend(tmp_trn + tmp_tst)  # 4+10*2 =24

        ens_A1 = [lam, 'A1', '', ''] + trn_A1 + tst_A1
        ens_A2 = [lam, 'A2', '', ''] + trn_A2 + tst_A2
        ens_Jt = [lam, 'Jt', '', ''] + trn_Jt + tst_Jt
        del trn_A1, trn_A2, trn_Jt, tmp_trn
        del tst_A1, tst_A2, tst_Jt, tmp_tst

        # Proposed
        tmp_pru, tmp_A1, tmp_A2, tmp_Jt = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-C", positive_label, lam=lam)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend(tmp_A1)
        ens_A2.extend(tmp_A2)
        ens_Jt.extend(tmp_Jt)

        tmp_pru, tmp_A1, tmp_A2, tmp_Jt = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "EPAF-D", positive_label, lam=lam, n_m=2)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend(tmp_A1)
        ens_A2.extend(tmp_A2)
        ens_Jt.extend(tmp_Jt)

        tmp_pru, tmp_A1, tmp_A2, tmp_Jt = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "POEPAF", positive_label, lam=lam, dist=1)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend(tmp_A1)
        ens_A2.extend(tmp_A2)
        ens_Jt.extend(tmp_Jt)

        tmp_pru, tmp_A1, tmp_A2, tmp_Jt = self.propose_prune_val(
            y_trn, y_insp, yq_insp, tag_trn, jt_trn,
            y_tst, y_pred, yq_pred, tag_tst, jt_tst,
            "POPEP", positive_label, lam=lam)
        ans_ens.extend(tmp_pru)  # +24
        ens_A1.extend(tmp_A1)
        ens_A2.extend(tmp_A2)
        ens_Jt.extend(tmp_Jt)

        del tmp_pru, tmp_A1, tmp_A2, tmp_Jt
        return ans_ens, ens_A1, ens_A2, ens_Jt  # each 24+24*5


# -------------------------------------
# legacy

# class PartB_Ensembles(EnsembleSetup):
# class PartC_ImprovedFairness(EnsembleSetup):
# class PartD_ImprovedFairness(EnsembleSetup):


class PartE_ImprovedFairness(EnsembleSetup):
    def __init__(self, name_ens, abbr_cls, nb_cls,
                 nb_pru=None, lam=.5):
        super().__init__(name_ens, abbr_cls, nb_cls)
        # To demonstrate my pruning could improve fairness
        # In last experiment I have demonstrated that it could
        # improve accuracy as well
        if nb_pru is None:
            nb_pru = nb_cls
        self._nb_pru = nb_pru
        self._lam = lam

    @property
    def lam(self):
        return self._lam

    @property
    def nb_pru(self):
        return self._nb_pru

    def propose_prune_val(self,
                          y_trn, y_insp, yq_insp,
                          name_pru, dist=1, n_m=2):
        since = time.time()

        if name_pru == 'POEPAF':
            H = Pareto_Optimal_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam, dist)
        elif name_pru == 'EPAF-C':
            H = Centralised_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam)
        elif name_pru == 'EPAF-D':
            H = Distributed_EPAF_Pruning(
                y_trn, y_insp, yq_insp, self._weight, self._nb_pru,
                self._lam, n_m)
        elif name_pru == "POPEP":
            H = POAF_PEP(y_trn, y_insp, yq_insp, self._weight,
                         self._lam, self._nb_pru)
        else:
            raise ValueError("Wrong parameter `{}`".format(name_pru))

        time_elapsed = time.time() - since
        return H, time_elapsed / 60

    def prepare_trial(self):
        csv_row_1 = unique_column((8 + 26) + (6 + 33 * 3 * 2 * 5))

        csv_row_2c_a = ['Time Cost (min)'] + [''] * 5
        csv_row_2c_ca = (['Ensem trn:A1'] + [''] * 32 +
                         ['Ensem trn:A2'] + [''] * 32 +
                         ['Ensem trn:jt'] + [''] * 32)  # 33*3 =99
        csv_row_2c_cb = (['Ensem tst:A1'] + [''] * 32 +
                         ['Ensem tst:A2'] + [''] * 32 +
                         ['Ensem tst:jt'] + [''] * 32)  # 33*3 =99
        csv_row_2c_cc = (['EPAF-C .trn'] + [''] * 98 +
                         ['EPAF-C .tst'] + [''] * 98 +
                         ['EPAF-D:2 .trn'] + [''] * 98 +
                         ['EPAF-D:2 .tst'] + [''] * 98 +
                         ['EPAF-D:3 .trn'] + [''] * 98 +
                         ['EPAF-D:3 .tst'] + [''] * 98 +
                         ['POEPAF .trn'] + [''] * 98 +
                         ['POEPAF .tst'] + [''] * 98)  # 99*2*4
        csv_row_2c_c = csv_row_2c_ca + csv_row_2c_cb + csv_row_2c_cc
        del csv_row_2c_ca, csv_row_2c_cb, csv_row_2c_cc  # 99*2*5
        csv_row_2c = csv_row_2c_a + csv_row_2c_c  # 6+99*10
        del csv_row_2c_a, csv_row_2c_c

        csv_row_3c_a = ['EPAF-C', 'EPAF-D:2', 'EPAF-D:3', 'POEPAF:1'
                        ] + ['ut.pru', 'ut.calc']
        csv_row_3c_ba = ['unpriv'] + [''] * 7 + ['manual', ''] + [
            'L_fair(MV)', 'L_loss(MV)', 'L(MV)']  # 8+2+3 =13
        csv_row_3c_bb = ['bar|unpriv'] + [''] * 7 + ['bar|manual', ''] + [
            'Erho[Lfair(f)]', 'Erho[Lloss(f)]', 'Erho[L(f)]']  # same 13
        csv_row_3c_bc = ['Erho^2[Lfair(f,fp)]', 'Erho^2[Lloss(f,fp)]',
                         'cal_L_obj_v1', 'cal_L_obj_v2',
                         'ED_Erho[ell_fair(f)]', 'ED_Erho[ell_loss(f)]',
                         '_bi_objectives(.,.)']  # 4+3 =7
        csv_row_3c_b = csv_row_3c_ba + csv_row_3c_bb + csv_row_3c_bc
        del csv_row_3c_ba, csv_row_3c_bb, csv_row_3c_bc  # 13*2+7 =33
        csv_row_3c_da = ['unpriv'] + [''] * 7 + ['manual', ''] + [
            'Lf(MV)', 'La(MV)', 'Lo(MV)']  # 10+3 =13
        csv_row_3c_db = ['bar()'] + [''] * 9 + ['E[Lf]', 'E[La]', 'E[Lo]']
        csv_row_3c_dc = ['E[# Lf]', 'E[# La]', 'Lo(MV)', 'Lo(MV)',
                         'ED[Ef^2]', 'ED[Ea^2]', '@G(lam)']  # 7
        csv_row_3c_d = csv_row_3c_da + csv_row_3c_db + csv_row_3c_dc
        del csv_row_3c_da, csv_row_3c_db, csv_row_3c_dc  # 13*2+7 =33
        csv_row_3c = (csv_row_3c_a + csv_row_3c_b * 3 * 2 +
                      csv_row_3c_d * 3 * 2 * 4)  # 6+99*2+99*2*4
        del csv_row_3c_a, csv_row_3c_b, csv_row_3c_d

        return csv_row_1, csv_row_2c, csv_row_3c

    def schedule_content(self,
                         y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                         y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                         fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                         positive_label):
        """
          y_trn /tst  : list,               shape= (nb_y,)
          y_insp/pred : list of np.ndarray, shape= (nb_cls, nb_y)
         yq_insp/pred : list of np.ndarray, shape= (nb_cls, nb_y)
         tag_trn/tst  : list of np.ndarray, shape= (1/2, nb_y)
          jt_trn/tst  : list of np.ndarray, shape= (0/1, nb_y)
        fens_trn/tst  : list,               shape= (nb_y,)
        fqtb_trn/tst  : list,               shape= (nb_y,)
        positive_label: scalar
        """
        since = time.time()
        H1, ut1 = self.propose_prune_val(y_trn, y_insp, yq_insp,
                                         'EPAF-C')
        H2, ut2 = self.propose_prune_val(y_trn, y_insp, yq_insp,
                                         'EPAF-D', n_m=2)
        H3, ut3 = self.propose_prune_val(y_trn, y_insp, yq_insp,
                                         'EPAF-D', n_m=3)
        H4, ut4 = self.propose_prune_val(y_trn, y_insp, yq_insp,
                                         'POEPAF', dist=1)
        time_elapsed = time.time() - since
        since = time.time()

        y_trn, y_tst = np.array(y_trn), np.array(y_tst)
        fens_trn, fqtb_trn = np.array(fens_trn), np.array(fqtb_trn)
        fens_tst, fqtb_tst = np.array(fens_tst), np.array(fqtb_tst)
        y_insp, yq_insp = np.array(y_insp), np.array(yq_insp)
        y_pred, yq_pred = np.array(y_pred), np.array(yq_pred)

        ans = []
        for H in [None, H1, H2, H3, H4]:
            tmp_trn = self.calc_fair_measure_routine(
                y_trn, y_insp, yq_insp, fens_trn, fqtb_trn, tag_trn,
                jt_trn, positive_label, H)
            tmp_tst = self.calc_fair_measure_routine(
                y_tst, y_pred, yq_pred, fens_tst, fqtb_tst, tag_tst,
                jt_tst, positive_label, H)
            ans.extend(tmp_trn + tmp_tst)  # 99+99

        tmp = [ut1, ut2, ut3, ut4, time_elapsed / 60]
        time_elapsed = time.time() - since
        tmp.append(time_elapsed / 60)
        return tmp + ans  # 6+99*2*5

    def calc_fair_measure_routine(self, y, yt, yqtb, fens, fqtb,
                                  tag, jt, positive_label, H=None):
        """
        y   : list,               shape= (nb_y,)
        yt  : list of np.ndarray, shape= (nb_cls, nb_y)
        yqtb: list of np.ndarray, shape= (nb_cls, nb_y)
        fens: list,               shape= (nb_y,)
        fqtb: list,               shape= (nb_y,)

        tag : list,  at most 2 ndarray, shape= (1,nb_y) or (2,nb_y)
        jt  : list, empty or 1 ndarray, shape= (0,)     or (1,nb_y)
        wgt : list,               shape= (nb_cls)
        """
        # y, fens, fqtb = np.array(y), np.array(fens), np.array(fqtb)
        if H is None:
            wgt = self._weight
        else:
            yt, yqtb = yt[H], yqtb[H]
            wgt = np.array(self._weight)[H].tolist()
        # TODO: something wrong here, didn't update fens/fqtb

        ans = []
        tmp = self.calc_fair_measure_gather(
            y, yt, yqtb, fens, fqtb, wgt, positive_label, tag[0])
        ans.extend(tmp)  # 33
        if not jt:
            ans.extend([''] * 33 * 2)
            return ans

        tmp = self.calc_fair_measure_gather(
            y, yt, yqtb, fens, fqtb, wgt, positive_label, tag[1])
        ans.extend(tmp)  # 33
        tmp = self.calc_fair_measure_gather(
            y, yt, yqtb, fens, fqtb, wgt, positive_label, jt[0])
        ans.extend(tmp)  # 33
        return ans  # 33*3 =99

    def calc_fair_measure_by_group(self, y, hx, hx_qtb,
                                   pos_val, idx_priv):
        """
        y :             np.ndarray, shape=(nb_y,)
        hx:             np.ndarray, shape=(nb_y,)
                        could be an individual or ensemble classifier
        positive_label: scalar
        ptb_priv[i]   : np.ndarray of boolean, shape=(nb_y,)
        lam           : scalar
        """
        g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
            marginalised_pd_mat(y, hx, pos_val, idx_priv)
        cmp_fair = []
        cmp_fair.extend(unpriv_unaware(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_group_one(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_group_two(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_group_thr(gones_Cm, gzero_Cm))
        cmp_fair.extend(unpriv_manual(gones_Cm, gzero_Cm))
        # above: more is better

        # below: less is better
        ans_fair = []
        ans_fair.append(hat_L_fair(hx, hx_qtb))  # 1-tmp \in [0,1]
        ans_fair.append(hat_L_loss(hx, y))       # 1-acc \in [0,1]
        ans_fair.append(hat_L_objt(hx, hx_qtb, y, self._lam))

        return cmp_fair + ans_fair  # 5*2+3 =13

    def calc_fair_measure_gather(self, y, yt, yqtb, fens, fqtb,
                                 wgt, pos_val, idx_priv):
        """
        y         : np.ndarray, shape= (nb_y,)
        yt  , yqtb: np.ndarray, shape= (nb_cls, nb_y)
        fens, fqtb: np.ndarray, shape= (nb_y,)
              wgt : list,       shape= (nb_cls,)

        元素乘法：np.multiply(a,b)
        矩阵乘法：np.dot(a,b) 或 np.matmul(a,b)
        """
        nb_cls = len(wgt)  # wgt = self._weight

        tmp_fair = [self.calc_fair_measure_by_group(
            y, hx, hq, pos_val, idx_priv) for hx, hq in zip(yt, yqtb)]
        tmp_fair = list(zip(*tmp_fair))
        tmp_fair = [np.dot(i, wgt).tolist() for i in tmp_fair]

        ens_fair = self.calc_fair_measure_by_group(
            y, fens, fqtb, pos_val, idx_priv)

        extra = []  # tandem_e,extra_f_ = []  # extra/ens
        extra.append(Erho_sup_L_fair(yt, yqtb, wgt, nb_cls))
        extra.append(Erho_sup_L_loss(yt, y, wgt, nb_cls))
        extra.extend([cal_L_obj_v1(yt, yqtb, y, wgt, self._lam),
                      cal_L_obj_v2(yt, yqtb, y, wgt, self._lam)])
        # tmp_G           aka. E_rho_L_fair_f, E_rho_L_loss_f
        tmp_G = (ED_Erho_I_fair(yt, yqtb, wgt),
                 ED_Erho_I_fair(yt, y, wgt))
        extra.extend(tmp_G)
        extra.append(_bi_objectives(tmp_G, self._lam))

        # ens_fair[-3:-1] aka. L_fair_MV_rho , L_loss_MV_rho
        # tmp_fair[-3:-1] aka. E_rho_L_fair_f, E_rho_L_loss_f
        return ens_fair + tmp_fair + extra  # 13*2+(4+3) =33


class PartF_ImprovedFairness(PartE_ImprovedFairness):
    def __init__(self, name_ens, abbr_cls, nb_cls,
                 nb_pru=None, lam=.5):
        # basically same as `PartE_ImprovedFairness`, just with
        # different presentation format.
        super().__init__(name_ens, abbr_cls, nb_cls, nb_pru, lam)

    def prepare_trial(self):
        csv_row_1 = unique_column((8 + 26) + (7 + 33 * 2 * 5))

        csv_row_2c_a = ['Time Cost (min)'] + [''] * 5 + ['Group:Attr']
        csv_row_2c_c = (['Ensem :trn'] + [''] * 32 +
                        ['Ensem :tst'] + [''] * 32 +
                        ['EPAF-C .trn'] + [''] * 32 +
                        ['EPAF-C .tst'] + [''] * 32 +
                        ['EPAF-D:2 .trn'] + [''] * 32 +
                        ['EPAF-D:2 .tst'] + [''] * 32 +
                        ['EPAF-D:3 .trn'] + [''] * 32 +
                        ['EPAF-D:3 .tst'] + [''] * 32 +
                        ['POEPAF .trn'] + [''] * 32 +
                        ['POEPAF .tst'] + [''] * 32)
        csv_row_2c = csv_row_2c_a + csv_row_2c_c  # 7+33*2*5
        del csv_row_2c_a, csv_row_2c_c

        csv_row_3c_a = ['EPAF-C', 'EPAF-D:2', 'EPAF-D:3', 'POEPAF'
                        ] + ['ut.pru', 'ut.calc'] + ['']  # 7
        csv_row_3c_ba = ['unpriv'] + [''] * 7 + ['manual', ''] + [
            'L_fair(MV)', 'L_loss(MV)', 'L(MV)']  # 8+2+3 =13
        csv_row_3c_bb = ['bar|unpriv'] + [''] * 7 + ['bar|manual', ''] + [
            'Erho[Lfair(f)]', 'Erho[Lloss(f)]', 'Erho[L(f)]']  # same 13
        csv_row_3c_bc = ['Erho^2[Lfair(f,fp)]', 'Erho^2[Lloss(f,fp)]',
                         'cal_L_obj_v1', 'cal_L_obj_v2',
                         'ED_Erho[ell_fair(f)]', 'ED_Erho[ell_loss(f)]',
                         '_bi_objectives(.,.)']  # 4+3 =7
        csv_row_3c_b = csv_row_3c_ba + csv_row_3c_bb + csv_row_3c_bc
        del csv_row_3c_ba, csv_row_3c_bb, csv_row_3c_bc  # 13*2+7 =33
        csv_row_3c_da = ['unpriv'] + [''] * 7 + ['manual', ''] + [
            'Lf(MV)', 'La(MV)', 'Lo(MV)']  # 10+3 =13
        csv_row_3c_db = ['bar()'] + [''] * 9 + ['E[Lf]', 'E[La]', 'E[Lo]']
        csv_row_3c_dc = ['E[# Lf]', 'E[# La]', 'Lo(MV)', 'Lo(MV)',
                         'ED[Ef^2]', 'ED[Ea^2]', '@G(lam)']  # 7
        csv_row_3c_d = csv_row_3c_da + csv_row_3c_db + csv_row_3c_dc
        del csv_row_3c_da, csv_row_3c_db, csv_row_3c_dc  # 13*2+7 =33
        csv_row_3c = (csv_row_3c_a + csv_row_3c_b * 2 +
                      csv_row_3c_d * 2 * 4)  # 7+33*2+33*2*4
        del csv_row_3c_a, csv_row_3c_b, csv_row_3c_d

        return csv_row_1, csv_row_2c, csv_row_3c

    def schedule_content(self,
                         y_trn, y_insp, yq_insp, tag_trn, jt_trn,
                         y_tst, y_pred, yq_pred, tag_tst, jt_tst,
                         fens_trn, fqtb_trn, fens_tst, fqtb_tst,
                         positive_label):
        since = time.time()
        H1, ut1 = self.propose_prune_val(y_trn, y_insp, yq_insp,
                                         'EPAF-C')
        H2, ut2 = self.propose_prune_val(y_trn, y_insp, yq_insp,
                                         'EPAF-D', n_m=2)
        H3, ut3 = self.propose_prune_val(y_trn, y_insp, yq_insp,
                                         'EPAF-D', n_m=3)
        H4, ut4 = self.propose_prune_val(y_trn, y_insp, yq_insp,
                                         'POEPAF', dist=1)
        time_elapsed = time.time() - since
        since = time.time()

        y_trn, y_tst = np.array(y_trn), np.array(y_tst)
        fens_trn, fqtb_trn = np.array(fens_trn), np.array(fqtb_trn)
        fens_tst, fqtb_tst = np.array(fens_tst), np.array(fqtb_tst)
        y_insp, yq_insp = np.array(y_insp), np.array(yq_insp)
        y_pred, yq_pred = np.array(y_pred), np.array(yq_pred)

        attr_A1, attr_A2, attr_Jt = ['A1'], ['A2'], ['Jt']
        for H in [None, H1, H2, H3, H4]:

            trn_A1, trn_A2, trn_Jt = self.calc_fair_measure_routine(
                y_trn, y_insp, yq_insp, fens_trn, fqtb_trn, tag_trn, jt_trn,
                positive_label, H)
            tst_A1, tst_A2, tst_Jt = self.calc_fair_measure_routine(
                y_tst, y_pred, yq_pred, fens_tst, fqtb_tst, tag_tst, jt_tst,
                positive_label, H)

            attr_A1.extend(trn_A1 + tst_A1)  # +33*2
            attr_A2.extend(trn_A2 + tst_A2)  # +33*2
            attr_Jt.extend(trn_Jt + tst_Jt)  # +33*2

        tmp = [ut1, ut2, ut3, ut4, time_elapsed / 60]
        time_elapsed = time.time() - since
        tmp.append(time_elapsed / 60)  # list, size=6
        return tmp, attr_A1, attr_A2, attr_Jt  # list,1+33*2*5

    def calc_fair_measure_routine(self, y, yt, yqtb, fens, fqtb,
                                  tag, jt, positive_label, H=None):
        if H is None:
            wgt = self._weight
        else:
            yt, yqtb = yt[H], yqtb[H]
            wgt = np.array(self._weight)[H].tolist()

        ans_A1 = self.calc_fair_measure_gather(
            y, yt, yqtb, fens, fqtb, wgt, positive_label, tag[0])
        if not jt:
            ans_A2 = [''] * 33
            ans_Jt = [''] * 33
            return ans_A1, ans_A2, ans_Jt

        ans_A2 = self.calc_fair_measure_gather(
            y, yt, yqtb, fens, fqtb, wgt, positive_label, tag[1])
        ans_Jt = self.calc_fair_measure_gather(
            y, yt, yqtb, fens, fqtb, wgt, positive_label, jt[0])
        return ans_A1, ans_A2, ans_Jt  # list, (33,)


# -------------------------------------

# -------------------------------------
