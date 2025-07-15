# coding: utf-8


from fairml.datasets import DATASETS, DATASET_NAMES, RAW_EXPT_DIR
from fairml.widget.utils_const import _get_tmp_document
from fairml.widget.utils_remark import (
    AVAILABLE_ABBR_CLS, AVAILABLE_NAME_PRUNE, LATEST_NAME_PRUNE)

# Experiments
from fairml.widget.utils_const import _get_tmp_name_ens
from fairml.widget.utils_wpclf import INDIVIDUALS
from fairml.datasets import DATASETS, DATASET_NAMES, preprocess
from fairml.preprocessing import (
    adversarial, transform_X_and_y, transform_unpriv_tag)
from fairml.facils.data_classify import EnsembleAlgorithm
from fairml.facils.metrics_cont import (
    calc_accuracy, calc_Acc, calc_PR, calc_F1, calc_4Rate,
    calc_confusion)

from fairml.facils.fairness_group import marginalised_pd_mat
from fairml.facilc.ensem_pruning import \
    contrastive_pruning_methods as exist_pruning_basics
from fairml.facilc.ensem_prulatest import \
    contrastive_pruning_lately as exist_pruning_latest
from fairml.discriminative_risk import (
    hat_L_fair, E_rho_L_fair_f, Erho_sup_L_fair,
    hat_L_loss, E_rho_L_loss_f)
from fairml.dr_pareto_optimal import (
    Pareto_Optimal_EPAF_Pruning, _bi_objectives, POAF_PEP,
    Centralised_EPAF_Pruning, Distributed_EPAF_Pruning)
# Experiments

import pandas as pd
import numpy as np
import os
CURR_EXPT_DIR = os.path.join(RAW_EXPT_DIR, 'wp2_oracle')


# ==================================
# Experiments
# ==================================


class DataSetup:
    def __init__(self, data_type):
        self._data_type = data_type
        self._log_document = data_type

        # ['ricci', 'german', 'adult', 'ppc', 'ppvc']
        if data_type == 'ppr':
            self._data_type = DATASET_NAMES[-2]
        elif data_type == 'ppvr':
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
        elif data_type == "ppc":
            self.saIndex = [0, 2]  # ['sex', 'race'] [0, 3]
        elif data_type == "ppvc":
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
                                         ind_priv=list()):
        # pos : dataset.positive_label
        # priv: dataset.get_positive_class_val('numerical-binsensitive')
        # sens: an element in `belongs_priv` or `belongs_priv_with_joint`

        # # y : pd.DataFrame  --> `.to_numpy()`
        # # hx: np.ndarray    --> `.tolist()`
        # y,hx,ind_priv: np.ndarray  # indicator_of_priv

        g1_Cij, g0_Cij, gones_Cm, gzero_Cm = \
            marginalised_pd_mat(y, hx, pos, ind_priv)
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
                                        X_trn, y_trn, X_val, X_tst):
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
        weig = [np.sum(
            coef * np.equal(yt, i), axis=0).tolist() for i in vY]
        loca = np.array(weig).argmax(axis=0).tolist()
        fens = [vY[i] for i in loca]
        return fens

    def pruning_baseline(self, y, yt, name_pru, epsilon=1e-6, rho=.4,
                         alpha=0.5, L=3, R=2, X=None, indices=None):
        # rho = float(self._nb_pru) / self._nb_cls  # default: 0.4

        if name_pru in AVAILABLE_NAME_PRUNE + ['KP']:
            ys, P, seq, _ = exist_pruning_basics(
                name_pru, self._nb_cls, self._nb_pru, y, yt, epsilon, rho)

        elif name_pru in LATEST_NAME_PRUNE:
            kwargs = {}
            if name_pru.startswith("TSP"):
                kwargs["indices"] = indices
            elif name_pru.startswith("TSPrev"):
                kwargs["X_trn"] = X
                kwargs["X_val"] = X

            ys, _, P, seq = exist_pruning_latest(
                name_pru, self._nb_cls, self._nb_pru, y, [], yt, [],
                alpha, L, R, **kwargs)

        else:
            raise ValueError("No such pruning named `{}`".format(name_pru))
        return ys, P, seq

    def pruning_proposed(self, y, yt, yq, name_pru, lam, dist=1, n_m=2):
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
            H = POAF_PEP(y, yt, yq, self._weight, lam, self._nb_pru)
        else:
            raise ValueError("No such pruning proposed `{}`".format(
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
        Acc, (_, p, r, f, _, _, _, _, _, _, _) = \
            self.calculate_sub_ensemble_metrics(y, fens, pos)
        tmp_pru.extend([Acc, p, r, f])

        # L_fair(MV_rho), L_acc(MV_rho)
        sub_whole = (hat_L_fair(fens, fqtb), hat_L_loss(fens, y))
        tmp_pru.extend(sub_whole)
        tmp_pru.append(_bi_objectives(sub_whole, lam))

        # objective
        sub_split = (
            Erho_sup_L_fair(ys, yr, wgt), E_rho_L_loss_f(ys, y, wgt))
        tmp_pru.extend(sub_split)
        tmp_pru.append(_bi_objectives(sub_split, lam))
        tmp_pru.append(E_rho_L_fair_f(ys, yr, wgt))

        return tmp_pru  # 4+(2+1)+(2+1)+1 =4+3*2+1 =11

    def prepare_trial(self):
        return [], [], []

    def schedule_content(self):
        raise NotImplementedError


# Experimental design
# -------------------------------------


class ExperimentSetup(DataSetup):  # that is, ExptSetting()
    def __init__(self, trial_type, data_type,
                 name_ens, abbr_cls, nb_cls, nb_pru,
                 nb_iter=5, ratio=.5, screen=True, logged=False):
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
        self._iterator = EnsembleSetup(
            name_ens, abbr_cls, nb_cls, nb_pru)

    # def __del__(self):
    #     pass

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
            '''
            X_trn = X.iloc[i_trn]
            y_trn = y.iloc[i_trn]
            Xp_trn = Xp.iloc[i_trn]
            X_tst = X.iloc[i_tst]
            y_tst = y.iloc[i_tst]
            Xp_tst = Xp.iloc[i_tst]
            '''

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


# ----------------------------------
