# coding: utf-8


from fairml.widget.utils_remark import AVAILABLE_ABBR_CLS
# Experiments
from fairml.widget.utils_const import _get_tmp_document, check_zero
from fairml.datasets import (
    DATASETS, DATASET_NAMES, RAW_EXPT_DIR)
# from fairml.facils.metrics_cont import (
#     calc_accuracy, calc_Acc, calc_PR, calc_F1, calc_4Rate,
#     calc_confusion)
# Experiments

import pandas as pd
import numpy as np
import numba
# import csv
# import logging
import os
# import sys
# import time
CURR_EXPT_DIR = os.path.join(RAW_EXPT_DIR, 'wp2_oracle')


# =====================================
# Metrics
# =====================================


# -------------------------------------
# TP, FP, FN, TN
# -------------------------------------
'''
|                      | predicted label positive  |  negative |
|true label is positive| true positive (TP)|false negative (FN)|
|true label is negative|false positive (FP)| true negative (TN)|
'''
# y, hx: list of scalars (as elements)


def calc_confusion(y, hx, pos=1):
    TP = np.logical_and(np.equal(y, pos), np.equal(hx, pos))
    FP = np.logical_and(np.not_equal(y, pos), np.equal(hx, pos))
    FN = np.logical_and(np.equal(y, pos), np.not_equal(hx, pos))
    TN = np.logical_and(np.not_equal(y, pos), np.not_equal(hx, pos))
    TP = np.sum(TP).tolist()  # TP = float(np.sum(TP))
    FP = np.sum(FP).tolist()  # FP = float(np.sum(FP))
    FN = np.sum(FN).tolist()  # FN = float(np.sum(FN))
    TN = np.sum(TN).tolist()  # TN = float(np.sum(TN))
    return TP, FP, FN, TN


def calc_accuracy(y, hx):
    n = len(y)
    t = np.sum(np.equal(y, hx)).tolist()
    # n = float(len(y))
    # t = float(np.sum(np.equal(y, hx)))
    return t / n  # == (TP+TN)/N


@numba.jit(nopython=True)
def calc_Acc(TP, FP, FN, TN):
    N = TP + FP + FN + TN
    accuracy_ = (TP + TN) / N
    return accuracy_, N


def calc_PR(TP, FP, FN):
    # precision = TP / (TP + FP)  # 查准率,精确率
    # recall = TP / (TP + FN)     # 查全率,召回率
    precision = TP / check_zero(TP + FP)
    recall = TP / check_zero(TP + FN)
    F1 = 2 * TP / check_zero(2 * TP + FP + FN)
    return precision, recall, F1


def calc_4Rate(TP, FP, FN, TN):
    '''
    TPR = TP / (TP + FN)  # 真正率,召回率,命中率 hit rate
    FPR = FP / (TN + FP)  # 假正率=1-特异度,误报/虚警/误检率 false alarm
    FNR = FN / (TP + FN)  # 漏报率 miss rate，也称为漏警率、漏检率
    TNR = TN / (TN + FP)  # 特异度 specificity
    # expect FPR,FNR smaller, TNR larger
    '''
    TPR = TP / check_zero(TP + FN)
    FPR = FP / check_zero(TN + FP)
    FNR = FN / check_zero(TP + FN)
    TNR = TN / check_zero(TN + FP)
    return TPR, FPR, FNR, TNR


def calc_F1(P, R, beta=1):
    if beta == 1:
        F1 = 2 * P * R / check_zero(P + R)
        return F1

    beta2 = beta ** 2
    denom = check_zero(beta2 * P + R)
    fbeta = (1 + beta2) * P * R / denom
    return fbeta


# ==================================
# Experiments
# ==================================


class DataSetup:
    def __init__(self, data_type):
        self._data_type = data_type
        self._log_document = data_type

        # ['ricci', 'german', 'adult', 'ppc', 'ppvc']
        if data_type == 'ppr':     # or data_type == 'ppc':
            self._data_type = DATASET_NAMES[-2]
        elif data_type == 'ppvr':  # or data_type == 'ppvc':
            self._data_type = DATASET_NAMES[-1]
        elif data_type not in ['ricci', 'german', 'adult']:
            raise ValueError("Wrong dataset `{}`".format(data_type))
        idx = DATASET_NAMES.index(self._data_type)

        self._dataset = DATASETS[idx]
        self._data_frame = self._dataset.load_raw_dataset()

        if data_type == "ricci":
            self.saIndex = [2]  # 'Race' -2
        elif data_type == "german":
            self.saIndex = [3, 5]  # ['sex', 'age'] [, 12]
        elif data_type == "adult":
            self.saIndex = [2, 3]  # ['race', 'sex'] [7, 8]
        elif data_type == "ppr":
            self.saIndex = [0, 2]  # ['sex', 'race'] [0, 3]
        elif data_type == "ppvr":
            self.saIndex = [0, 2]  # ['sex', 'race'] [0, 3]
        self.saValue = self._dataset.get_privileged_group('numerical-binsensitive')
        # self.saValue = 0  # 1 means the privileged group
        self.saValue = [0 for sa in self.saValue if sa == 1]

    @property
    def data_type(self):
        return self._data_type

    # no use?
    @property
    def trial_type(self):
        return self._trial_type

    @property
    def log_document(self):
        return self._log_document

    # # ----------- mu -----------
    # def prepare_mu_datasets(self, ratio=.5, logger=None):
    #   pass
    # # ----------- tr -----------
    # # ----------- bi -----------
    # def prepare_bi_datasets(self, ratio=.5, logger=None):
    #   pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_frame(self):
        return self._data_frame


class GraphSetup:
    def __init__(self,
                 name_ens,
                 nb_cls,
                 nb_pru=None,
                 nb_iter=5,
                 figname=''):
        self._name_ens = name_ens
        self._nb_cls = nb_cls
        if nb_pru is None:
            nb_pru = nb_cls
        self._nb_pru = nb_pru
        self._nb_iter = nb_iter  # 5
        self._figname = figname

    @property
    def trial_type(self):
        return self._trial_type

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
    def nb_iter(self):
        return self._nb_iter

    @property
    def figname(self):
        return self._figname

    @property
    def raw_dframe(self):
        return self._raw_dframe

    @raw_dframe.setter
    def raw_dframe(self, value):
        self._raw_dframe = value

    def get_raw_filename(self, trial_type = 'mCV_expt3'):
        nmens_tmp = _get_tmp_document(self._name_ens, self._nb_cls)
        filename = "{}_iter{}_pms.xlsx".format(nmens_tmp, self._nb_iter)
        if trial_type:
            trial_type += "_"
        return os.path.join(  # osp.join(RAW_EXPT_DIR,
            CURR_EXPT_DIR, "{}{}".format(trial_type, filename))

    def load_raw_dataset(self, filename):
        dframe = {}
        for _, v in enumerate(AVAILABLE_ABBR_CLS):  # _:k
            dframe[v] = pd.read_excel(filename, v)
        self._raw_dframe = dframe
        return dframe

    def recap_sub_data(self, dframe, nb_row=3):
        nb_set = [len(v) for k, v in dframe.items()]
        assert len(set(nb_set)) == 1, "{}-{}".format(
            self._name_ens, self._nb_cls)
        nb_set = list(set(nb_set))[0]
        nb_set = (nb_set - nb_row + 1) // (self._nb_iter + 1)

        id_set = [i * (
            self._nb_iter + 1) + nb_row - 1 for i in range(nb_set)]
        index = [[i + j for j in range(
            1, self._nb_iter + 1)] for i in id_set]
        return nb_set, id_set, index

    def fetch_sub_data(self, df, index, tag_col, tag_ats='ua'):
        # index = np.concatenate(index).tolist()
        data = df.iloc[index][tag_col]
        # data = dframe.iloc[index, tag_col]

        if tag_ats == 'us':
            data.fillna(self._nb_cls, inplace=True)
        elif tag_ats == 'ua':
            data *= 100.
        data = data.values.astype('float')  # .to_numpy()
        return data  # np.ndarray

    def stats_sub_calc(self, data, nb_set, nb_col):
        avg = np.zeros((nb_set, nb_col))
        std = np.zeros((nb_set, nb_col))
        var = np.zeros((nb_set, nb_col))

        for i in range(nb_set):
            loc_idx = np.arange(self._nb_iter) + i * self._nb_iter
            now_set = data[loc_idx]

            avg[i] = now_set.mean(axis=0)
            std[i] = now_set.std(axis=0, ddof=1)
            var[i] = now_set.var(axis=0, ddof=1)
        return avg, std, var

    def merge_sub_data(self, dframe, index, tag_col, tag_ats='us'):
        nb_set = np.shape(index)[0]
        index = np.concatenate(index).tolist()
        nb_col = len(tag_col)

        raw_data = {}
        avg, std, var = {}, {}, {}

        for k, v in enumerate(AVAILABLE_ABBR_CLS):
            # dframe[v].iloc[index][tag_col]
            raw_data[v] = self.fetch_sub_data(
                dframe[v], index, tag_col, tag_ats)

            avg[v], std[v], var[v] = self.stats_sub_calc(
                raw_data[v], nb_set, nb_col)

        raw = [v for k, v in raw_data.items()]
        raw = np.concatenate(raw, axis=0)
        avg = np.concatenate([v for k, v in avg.items()], axis=0)
        std = np.concatenate([v for k, v in std.items()], axis=0)
        var = np.concatenate([v for k, v in var.items()], axis=0)
        return raw, avg, std, var

    # Pandas
    # def schedule_content(self):
    #   raise NotImplementedError

    def schedule_mspaint(self, raw_dframe, tag_col):
        raise NotImplementedError

    def subdraw_spliting(self):
        raise NotImplementedError

    def subdraw_asawhole(self):
        raise NotImplementedError

    def prepare_graph(self):
        raise NotImplementedError


# Experiments
# ----------------------------------


def pd_concat_divide_set(i_dframe, tag_col):
    #                      dat_dframe
    # aka. def pd_concat_divide_avg(i_dframe, tag_col):

    i_avg = {i: i_dframe[i].mean() for i in tag_col}
    i_avg = pd.DataFrame(i_avg, index=[0])

    i_std = {i: [i_dframe[i].std(ddof=1)] for i in tag_col}
    i_var = {i: [i_dframe[i].var(ddof=1)] for i in tag_col}
    i_std = pd.DataFrame(i_std)
    i_var = pd.DataFrame(i_var)

    return i_avg, i_std, i_var


def pd_concat_divide_sht(sht_dframe, tag_col, nb_set, index):
    # sht_dframe = [sht_dframe.iloc[i][tag_col] for i in index]
    sht_dframe = [sht_dframe.loc[i][tag_col] for i in index]
    # Doesn't matter that much, as it is the sheet

    tmp = list(map(pd_concat_divide_set,
                   sht_dframe, [tag_col] * nb_set))
    t_avg, t_std, t_var = zip(*tmp)

    t_avg = pd.concat(t_avg, ignore_index=True)
    t_std = pd.concat(t_std, ignore_index=True)
    t_var = pd.concat(t_var, ignore_index=True)

    t_raw = pd.concat(sht_dframe, ignore_index=False)
    return t_avg, t_std, t_var, t_raw


def pd_concat_divide_raw(raw_dframe, tag_col, nb_set, index):
    keys = AVAILABLE_ABBR_CLS  # list(raw_dframe.keys())
    sht_dframe = {k: pd_concat_divide_sht(
        v, tag_col, nb_set, index) for k, v in raw_dframe.items()}

    s_avg = [sht_dframe[i][0] for i in keys]
    s_std = [sht_dframe[i][1] for i in keys]
    s_var = [sht_dframe[i][2] for i in keys]
    s_raw = [sht_dframe[i][3] for i in keys]

    s_avg = pd.concat(s_avg, ignore_index=False)
    s_std = pd.concat(s_std, ignore_index=False)
    s_var = pd.concat(s_var, ignore_index=False)
    s_raw = pd.concat(s_raw, ignore_index=False)

    return s_avg, s_std, s_var, s_raw


def pd_concat_sens_raw(raw_dframe, tag_col,
                       nb_set, index):
    # keys = AVAILABLE_ABBR_CLS
    ind_A1 = [np.add(i, 1) for i in index]
    ind_A2 = [np.add(i, 2) for i in index[1:]]
    ind_Jt = [np.add(i, 3) for i in index[1:]]

    A1_avg, A1_std, A1_var, A1_raw = pd_concat_divide_raw(
        raw_dframe, tag_col, nb_set, ind_A1)
    A1_data = (A1_avg, A1_std, A1_var, A1_raw)

    A2_avg, A2_std, A2_var, A2_raw = pd_concat_divide_raw(
        raw_dframe, tag_col, nb_set, ind_A2)
    Jt_avg, Jt_std, Jt_var, Jt_raw = pd_concat_divide_raw(
        raw_dframe, tag_col, nb_set, ind_Jt)
    A2_data = (A2_avg, A2_std, A2_var, A2_raw)
    Jt_data = (Jt_avg, Jt_std, Jt_var, Jt_raw)
    return A1_data, A2_data, Jt_data
