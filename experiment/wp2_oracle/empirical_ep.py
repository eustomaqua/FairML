# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority vote
#


import numpy as np
# import pandas as pd
import time

from fairml.widget.utils_const import unique_column
# from fairml.widget.utils_saver import elegant_print

from fairml.discriminative_risk import (
    hat_L_fair, E_rho_L_fair_f,  # ell_fair_x,tandem_fair,
    hat_L_loss, E_rho_L_loss_f,  # ell_loss_x,tandem_loss,
    Erho_sup_L_fair)  # Erho_sup_L_loss, ED_Erho_I_fair,
# hat_L_objt, cal_L_obj_v1, cal_L_obj_v2)
from fairml.dr_pareto_optimal import (
    Pareto_Optimal_EPAF_Pruning, Centralised_EPAF_Pruning,
    Distributed_EPAF_Pruning, POAF_PEP, _POAF_calc_eval)
# from fairml.dr_pareto_optimal import _bi_objectives  # as _POAF_calc_eval

from experiment.wp2_oracle.fetch_expt import EnsembleSetup
# from fairml.facils.fairness_group import (
#     unpriv_group_one, unpriv_group_two, unpriv_group_thr,
#     marginalised_pd_mat, unpriv_unaware, unpriv_manual)
from fairml.facils.metric_fair import marginalised_pd_mat
from fairml.facils.metric_fair import prev_unpriv_grp_one \
    as unpriv_group_one
from fairml.facils.metric_fair import prev_unpriv_grp_two \
    as unpriv_group_two
from fairml.facils.metric_fair import prev_unpriv_grp_thr \
    as unpriv_group_thr
from fairml.facils.metric_fair import prev_unpriv_unaware \
    as unpriv_unaware
from fairml.facils.metric_fair import prev_unpriv_manual \
    as unpriv_manual


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
