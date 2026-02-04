# coding: utf-8
# Usage: empirical results' representation


import json
import os
import time
import numpy as np
# import pdb
import pandas as pd

from pyfair.facil.utils_const import DTY_FLT, _get_tmp_document
from pyfair.facil.utils_saver import (
    get_elogger, rm_ehandler, elegant_print)
from pyfair.facil.utils_timer import elegant_durat

from pyfair.marble.draw_hypos import (
    _encode_sign, Friedman_init,  # Friedman_test,
    cmp_paired_wtl, cmp_paired_avg, comp_t_sing, comp_t_prep)
from pyfair.granite.draw_chart import (
    multiple_scatter_chart, multiple_scatter_alternative,
    analogous_confusion, analogous_confusion_alternative,
    analogous_confusion_extended)
from pyfair.granite.draw_graph import (
    Friedman_chart, stat_chart_stack, multiple_hist_chart,
    scatter_id_chart)

from pyfair.granite.draw_addtl import (
    FairGBM_scatter, FairGBM_tradeoff_v1, FairGBM_tradeoff_v2,
    FairGBM_tradeoff_v3, multi_lin_reg_without_distr)
from experiment.wp2_oracle.fvote_draw import PlotC_TheoremsLemma
from experiment.wp2_oracle.fetch_data import DataSetup, GraphSetup
from experiment.wp2_oracle.fvote_addtl import _little_helper


# =====================================
# Experiments (re-presentation)
# =====================================


# -------------------------------------
# oracle_plot.py
# -------------------------------------
# oracle_replot.py


class PlotD_Measures(GraphSetup):
    def __init__(self):
        pass

    def prepare_graph(self, res_data):
        new_data = np.zeros_like(res_data).transpose(1, 2, 3, 0)
        nb_iter, nb_attr, nb_ens, nb_eval = np.shape(res_data)
        for i in range(nb_attr):
            for j in range(nb_ens):
                for k in range(nb_eval):
                    new_data[i, j, k] = res_data[:, i, j, k]
        del nb_iter, nb_attr, nb_ens, nb_eval
        return new_data

    def schedule_mspaint(self, res_data, res_all, figname=""):
        new_data = np.array(res_data)
        new_data = self.prepare_graph(new_data)[:, :, 56:, :]

        # data_name, binary, nb_cls, _, nb_iter, _ = res_all[0]
        _, _, _, _, nb_iter, _ = res_all[0]
        sensitive_attributes = res_all[1]
        ensemble_methods = res_all[-1]
        ensemble_methods = [i.replace(
            'FPR', 'fpr').replace('FNR', 'fnr') if (
            'FairGBM' in i) else i for i in ensemble_methods]
        idx = [0, 1, 2, 3, 4, 6]
        ensemble_methods = [ensemble_methods[i] for i in idx]

        for sa, sens_attr in enumerate(sensitive_attributes):
            fgn = "{}_{}_".format(figname, sens_attr)
            curr = new_data[sa][idx]
            del fgn, curr

        # num_s, num_e, num_v, _ = new_data.shape
        num_s, num_e, _, _ = new_data.shape
        alt_data = np.concatenate([
            new_data[i] for i in range(num_s)], axis=2)
        alt_data = np.concatenate([
            alt_data[i] for i in range(num_e)], axis=1)
        X, Ys = alt_data[26], alt_data[[50, 51, 52, 54], :]
        annots = (r"$\Delta$(Accuracy)", "Fairness Measure")
        annotZs = ('DP', 'EO', 'PQP', 'DR')
        fgn = figname + "_correlation"
        # multiple_scatter_chart(X, Ys, annots, annotZs, fgn)
        multiple_scatter_chart(X, Ys, annots, annotZs, fgn,
                               ind_hv='v', identity=False)

        Mat = alt_data[[26, 27, 28, 29, 32, 33, 50, 51, 52, 54]]
        key = ["Acc", "P", "R", "f1", "Sen", "Spe"]
        key = [r"$\Delta$({})".format(i) for i in key
               ] + ["DP", "EO", "PQP", "DR"]
        fgn = figname + "_confusion"
        # pdb.set_trace()
        # analogous_confusion(Mat, key, fgn, normalize=False)

        idx_A = [0, 1, 2, 3, 5]
        Mat_A, Mat_B = Mat[idx_A], Mat[6:]
        key_A, key_B = [key[i] for i in idx_A], key[6:]
        analogous_confusion_extended(
            Mat_B, Mat_A, key_B, key_A, figname + '_confusion_alt',
            cmap_name='PuBu', rotate=0, figsize='M-WS')
        return

    def plot_multiple_hist_chart(self, curr, ens, mark="acc.norm",
                                 fgn="", ddof=1):
        if mark.startswith("acc"):  # without sen/spe
            idx = [0, 1, 2, 3]      # normal
            # idx = [13, 14, 15, 16]  # adversarial
            # idx = [26, 27, 28, 29]  # abs()
            key = ["Accuracy", "Precision", "Recall", "f1_score"]
        elif mark.startswith("sen"):
            idx = [6, 7, 8, 10, 11]
            # idx = [19, 20, 21, 23, 24]
            # idx = [32, 33, 34, 36, 37]
            key = ["Sensitivity", "Specificity", "G_mean",
                   "Matthew", "Cohen"]
        elif mark == "fair":
            idx = [50, 51, 52, 54]
            key = ["DP", "EO", "PQP", "DR"]  # or "FQ (ours)"

        '''
        if mark.endswith("norm"):
            pass
        elif mark.endswith("advr"):
            idx = [i + 13 for i in idx]
        elif mark.endswith("abs_"):
            idx = [i + 26 for i in idx]
        # mode = "ascend" if mark == "fair" else "descend"
        '''
        if mark.endswith("advr"):
            idx = [i + 13 for i in idx]
        elif mark.endswith("abs_"):
            idx = [i + 26 for i in idx]
        # elif mark.endswith("norm"): pass

        new_curr = curr[:, idx, :]
        Ys_avg = new_curr.mean(axis=2).T
        Ys_std = new_curr.std(axis=2, ddof=ddof).T
        multiple_hist_chart(Ys_avg, Ys_std, key, '', ens,
                            figname=fgn + mark, rotate=20)
        return


class GatherD_Measures(PlotD_Measures):
    def schedule_mspaint(self, res_data, res_all,
                         optional_data, figname=""):
        # each res_data.shape (#iter,#attr,#ens,#eval) =(5,1|2,7,113)
        # each new_data.shape (#attr,#ens,#eval,#iter) =(1|2,7,113,5)
        new_data = [self.prepare_graph(np.array(
            res_data[i]))[:, :, 56:, :] for i in optional_data]

        alt_data = np.concatenate(new_data, axis=0)
        num_s, num_e, _, _ = alt_data.shape  # num_v =.shape[2]
        alt_data = np.concatenate([
            alt_data[i] for i in range(num_s)], axis=2)
        alt_data = np.concatenate([
            alt_data[i] for i in range(num_e)], axis=1)

        X, Ys = alt_data[26], alt_data[[50, 51, 52, 54], :]
        annots = (r"$\Delta$(Accuracy)", "Fairness Measure")
        annotZs = ('DP', 'EO', 'PQP', 'DR')
        fgn = figname + "_correlation"
        multiple_scatter_chart(X, Ys, annots, annotZs, fgn,
                               ind_hv='v', identity=False)

        Mat = alt_data[[26, 27, 28, 29, 32, 33, 50, 51, 52, 54]]
        key = ["Acc", "P", "R", "f1", "Sen", "Spe"]
        key = [r"$\Delta$(%s)" % i for i in key
               ] + ["DP", "EO", "PQP", "DR"]
        fgn = figname + "_confusion"

        Mat_A, Mat_B = Mat[: 6], Mat[6:]
        key_A, key_B = key[: 6], key[6:]
        Mat_C = Mat_A[[0, 1, 2, 3, 5]]
        key_C = [key_A[i] for i in [0, 1, 2, 3, 5]]  # delta(Sen)
        analogous_confusion_extended(
            Mat_B, Mat_C, key_B, key_C, figname + '_confusion_alt',
            cmap_name='PuBu', rotate=0, figsize='M-WS')
        return


class PlotE_Measures(GraphSetup):
    def __init__(self):
        pass

    def prepare_graph(self, res_data):
        new_data = np.zeros_like(res_data).transpose(1, 2, 3, 0)
        nb_iter, nb_attr, nb_ens, nb_eval = np.shape(res_data)
        for i in range(nb_attr):
            for j in range(nb_ens):
                for k in range(nb_eval):
                    new_data[i, j, k] = res_data[:, i, j, k]
        del nb_iter, nb_attr, nb_ens, nb_eval
        return new_data

    def schedule_mspaint(self, res_data, res_all, figname="",
                         jt=False):
        new_data = np.array(res_data)
        idx = list(range(56, 112)) + list(
            range(168, 224)) + list(range(280, 336)) + [337 - 1, ]
        new_data = self.prepare_graph(new_data)[:, :, idx, :]

        # data_name, binary, nb_cls, _, nb_iter, _ = res_all[0]
        _, _, nb_cls, _, nb_iter, _ = res_all[0]
        ensemble_methods = res_all[-1]
        ensemble_methods = [
            i.replace('FPR', 'fpr').replace('FNR', 'fnr')
            if 'FairGBM' in i else i for i in ensemble_methods]
        idx = [0, 1, 2, 3, 4, 6]
        ensemble_methods = [ensemble_methods[i] for i in idx]

        sensitive_attributes = res_all[1]
        for sa, sens_attr in enumerate(sensitive_attributes):
            fgn = "{}_{}_".format(figname, sens_attr)
            curr = new_data[sa][idx]
            del fgn, curr

        attr_A = list(range(56))
        attr_B = list(range(56, 112))
        attr_J = list(range(112, 168))
        test_A = new_data[:, :, attr_A, :].astype(DTY_FLT)
        if len(sensitive_attributes) > 1:
            test_B = new_data[:, :, attr_B, :].astype(DTY_FLT)
            test_J = new_data[:, :, attr_J, :].astype(DTY_FLT)

        num_s, num_e, _, _ = new_data.shape  # num_v =.shape[2]
        test_A = np.concatenate([
            test_A[i] for i in range(num_s)], axis=2)
        test_A = np.concatenate([
            test_A[i] for i in range(num_e)], axis=1)
        if len(sensitive_attributes) > 1:
            test_B = np.concatenate([
                test_B[i] for i in range(num_s)], axis=2)
            test_B = np.concatenate([
                test_B[i] for i in range(num_e)], axis=1)
            test_J = np.concatenate([
                test_J[i] for i in range(num_s)], axis=2)
            test_J = np.concatenate([
                test_J[i] for i in range(num_e)], axis=1)

        annots = (r"$\Delta$(Accuracy)", "Fairness Measure")
        annotZs = ('DP', 'EO', 'PQP', 'DR')
        fgn_r = figname + "_correlation"
        key = ["Acc", "P", "R", "f1", "Sen", "Spe"]
        key = [r"$\Delta$(%s)" % i for i in key] + [
            "DP", "EO", "PQP", "DR"]
        fgn_f = figname + "_confusion"

        if len(sensitive_attributes) == 1:
            alt_data = [test_A]
        elif not jt:
            alt_data = [test_A, test_B]
        else:
            alt_data = [test_A, test_B, test_J]
            sensitive_attributes.append('Joint')
        fs = "_{}".format(str(jt)[0])
        fgn_r += fs
        fgn_f += fs

        X = [i[26] for i in alt_data]
        Ys = [i[[50, 51, 52, 54], :] for i in alt_data]
        multiple_scatter_alternative(
            X, Ys, sensitive_attributes, annots, annotZs,
            fgn_r, box=True)
        Mat = [i[[26, 27, 28, 29, 32, 33,
                  50, 51, 52, 54]] for i in alt_data]
        analogous_confusion_alternative(
            Mat, sensitive_attributes, key, fgn_f)
        return


class GatherE_Measures(PlotE_Measures):
    def handle_each_set(self, res_data, res_all, jt=False):
        idx = list(range(56, 112)) + list(range(
            168, 224)) + list(range(280, 336)) + [337 - 1, ]
        new_data = self.prepare_graph(np.array(res_data))[:, :, idx, :]
        num_s, num_e, _, _ = new_data.shape  # num_v =.shape[2]
        sensitive_attributes = res_all[1]

        attr_A = list(range(56))
        attr_B = list(range(56, 112))
        attr_J = list(range(112, 168))
        test_A = new_data[:, :, attr_A, :].astype(DTY_FLT)
        if len(sensitive_attributes) > 1:
            test_B = new_data[:, :, attr_B, :].astype(DTY_FLT)
            test_J = new_data[:, :, attr_J, :].astype(DTY_FLT)
        # test_*.shape: (1|2,7,56,5)= (#attr,#ens,#eval'',#iter)

        test_A = np.concatenate([
            test_A[i] for i in range(num_s)], axis=2)
        test_A = np.concatenate([
            test_A[i] for i in range(num_e)], axis=1)
        if len(sensitive_attributes) > 1:
            test_B = np.concatenate([
                test_B[i] for i in range(num_s)], axis=2)
            test_B = np.concatenate([
                test_B[i] for i in range(num_e)], axis=1)
            test_J = np.concatenate([
                test_J[i] for i in range(num_s)], axis=2)
            test_J = np.concatenate([
                test_J[i] for i in range(num_e)], axis=1)

        if len(sensitive_attributes) == 1:
            return test_A
        alt_data = [test_A, test_B]
        if jt:
            alt_data.append(test_J)
        return np.concatenate(alt_data, axis=1)

    def schedule_mspaint(self, res_data, res_all,
                         optional_data, figname="", jt=False):
        alt_data = [self.handle_each_set(
            res_data[i], res_all[i], jt) for i in optional_data]
        alt_data = np.concatenate(alt_data, axis=1)

        annots = (r"$\Delta$(Accuracy)", "Fairness Measure")
        annotZs = ('DP', 'EO', 'PQP', 'DR')
        fgn_r = figname + "_correlation" + "_{}".format(str(jt)[0])
        key = ["Acc", "P", "R", "f1", "Sen", "Spe"]
        key = [r"$\Delta$(%s)" % i for i in key] + [
            "DP", "EO", "PQP", "DR"]
        fgn_f = figname + "_confusion" + "_{}".format(str(jt)[0])

        X, Ys = alt_data[26], alt_data[[50, 51, 52, 54], :]
        multiple_scatter_chart(X, Ys, annots, annotZs,
                               fgn_r, ind_hv='v', identity=False)
        Mat = alt_data[[26, 27, 28, 29, 32, 33, 50, 51, 52, 54]]
        analogous_confusion(Mat, key, fgn_f, normalize=False)
        return


"""
def _little_helper(name, sensitive_attributes=list()):
    name_ens_set = ['Bagging', 'AdaBoostM1', 'SAMME']
    if "Entire" in name:
        return name.replace(" & Entire", "")
    for i in name_ens_set:
        if i in name:
            name = name.replace("{} & ".format(i), "")
            break
    name = name.replace(":2", "")
    name = name.replace(":3", "")
    name = name.replace("POEPAF", "POAF")
    name = name.replace("POPEP", "POAF")
    for i in sensitive_attributes:
        if i in name:
            name = name.replace(" via {}".format(i), "")
            break
    name = name.replace(" (fpr)", "")
    name = name.replace(" (fnr)", "")
    name = name.replace("LightGBM", "lightGBM")
    return name
"""


class PlotF_Prunings(GraphSetup):
    def __init__(self):
        pass

    def prepare_graph(self, res_data):
        new_data = np.zeros_like(res_data).transpose(1, 2, 0)
        nb_iter, nb_ens, nb_eval = np.shape(res_data)
        for i in range(nb_ens):
            for j in range(nb_eval):
                new_data[i, j] = res_data[:, i, j]
        del nb_iter, nb_ens, nb_eval
        return new_data

    def schedule_mspaint(self, res_data, res_all, figname="",
                         jt=False, logger=None):
        new_data = np.array(res_data)
        idx = list(range(56, 112)) + list(range(168, 224)) + list(
            range(280, 336)) + list(range(336, 339))
        new_data = self.prepare_graph(new_data)[:, idx, :]

        # data_name, binary, nb_cls, nb_pru, nb_iter, _ = res_all[0]
        data_name, _, _, _, _, _ = res_all[0]
        sensitive_attributes = res_all[1]
        name_ens_set, name_pru_set = res_all[3], res_all[4]  # -4/-3
        domestic_key = ['{} & {}'.format(
            i, j) for i in name_ens_set for j in name_pru_set]
        domestic_key.extend(res_all[-2])
        for sa in sensitive_attributes:
            domestic_key.extend([
                '{} via {}'.format(i, sa) for i in res_all[-1]])
        domestic_key = [
            _little_helper(i, sensitive_attributes) for i in domestic_key]

        e_i = [0, ]
        p_i = [0, 1, 3, 5]
        f_i = [0, 3]
        s_i_set = [0, 1] if data_name != 'ricci' else [0]
        j_i = [20]
        tmp_1 = np.arange(3 * 6).reshape(3, 6)
        tmp_1 = tmp_1[e_i, :][:, p_i].reshape(-1).tolist()
        for s_i in s_i_set:
            tmp_2 = np.arange(2 * 4).reshape(2, 4) + 21
            tmp_2 = tmp_2[[s_i], :][:, f_i].reshape(-1).tolist()
            choose_idx = j_i + tmp_2 + tmp_1
            choose_key = [domestic_key[i] for i in choose_idx]
            alt_data = new_data[choose_idx, :, :]
            elegant_print(["sens_attr: {} {}".format(
                s_i, sensitive_attributes[s_i]), "\n"], logger)
            self.plot_multiple_hist_chart(s_i, 0, alt_data, choose_key, figname, logger=logger)
            self.plot_multiple_hist_chart(s_i, 3, alt_data, choose_key, figname, logger=logger)
            self.plot_multiple_hist_chart(s_i, 54, alt_data, choose_key, figname, logger=logger)
        del tmp_1, tmp_2, j_i, s_i, f_i, e_i, p_i
        del choose_idx, choose_key, alt_data

    def plot_multiple_hist_chart(self, sa_i, pt_i, alt_data,
                                 choose_key, fgn="", ddof=0,
                                 logger=None):
        if sa_i == 0:
            tag_choice = list(range(0, 56)) + [168, 169, 170]
        elif sa_i == 1:
            tag_choice = list(range(56, 112)) + [168, 169, 170]
        elif sa_i == 2:
            tag_choice = list(range(112, 168 + 3))
        tag = {
            0: "Accuracy (%)",
            1: "Precision",
            2: "Recall",
            3: "f1_score",
            4: "fpr",
            5: "fnr",
            6: "Sensitivity",
            7: "Specificity",
            8: "G_mean",
            9: "dp",
            10: "Matthew",
            11: "Cohen",
            12: "random_acc",

            39: "unaware",  # 39,40, 49 =39+10
            41: "DP",       # 41,42, 50 =41+9
            43: "EO",       # 43,44, 51 =53+8
            45: "PQP",      # 45,46, 52 =45+7
            47: "manual",   # 47,48, 53 =47+6
            49: "unaware",
            50: "DP",
            51: "EO",
            52: "PQP",
            53: "manual",
            # 54: r"$\hat{L}_{fair}$",
            # 55: r"$\hat{L}_{err}$",  # loss
            54: r"$L_{fair}$",
            55: r"$L_{err}$",
            -3: r"$T_{ens}$",
            -2: r"$T_{pru}$",
            -1: r"$T_{all}$",  # "Time Cost (sec)",
        }
        fgn += "_sa{}".format(sa_i)
        alt_data = alt_data[:, tag_choice, :]
        alt_data = alt_data.astype(DTY_FLT)

        if 0 <= pt_i < 13:
            Ys = alt_data[:, [pt_i, pt_i + 13], :]
            if pt_i == 0:
                Ys *= 100
            Ys_avg = Ys.mean(axis=2)
            Ys_std = Ys.std(axis=2, ddof=ddof)
            multiple_hist_chart(
                Ys_avg, Ys_std, choose_key, tag[pt_i],
                ["Raw", "Disturbed"],
                fgn + "_{}{}".format(pt_i, tag[pt_i][:3]))
            outputs = []
            rez = 2 if pt_i == 0 else 4
            for oi, ok in enumerate(["Raw", "Disturbed"]):
                otmp = "{}\n".format(ok)
                for t1, t2 in zip(Ys_avg[:, oi], Ys_std[:, oi]):
                    otmp += " & {}".format(_encode_sign(t1, t2, rez))
                outputs.append(otmp)

        elif pt_i >= 39:
            tmp = [50, 51, 52, 54]  # [50, 51, 52, 54, 55]
            Ys = alt_data[:, tmp, :]
            Ys_avg = Ys.mean(axis=2)
            Ys_std = Ys.std(axis=2, ddof=ddof)
            multiple_hist_chart(Ys_avg, Ys_std, choose_key,
                                "Fairness Measure",
                                [tag[t] for t in tmp],
                                fgn + "_{}{}".format(54, "fair"))
            multiple_hist_chart(Ys_avg.T, Ys_std.T,
                                [tag[t] for t in tmp],
                                "Fairness Measure",
                                choose_key,
                                fgn + "_{}{}".format(55, "fair"))
            outputs = []
            for oi, ok in enumerate([tag[t] for t in tmp]):
                otmp = "{}\n".format(ok)
                for t1, t2 in zip(Ys_avg[:, oi], Ys_std[:, oi]):
                    otmp += " & {}".format(_encode_sign(t1, t2, 4))
                outputs.append(otmp)

        elif pt_i < 0:
            Ys = alt_data[:, [56, 57, 58], :]
            Ys_avg = Ys.mean(axis=2)
            Ys_std = Ys.std(axis=2, ddof=ddof)
            multiple_hist_chart(Ys_avg, Ys_std, choose_key,
                                "Time Cost (sec)",
                                [tag[t] for t in [-3, -2, -1]],
                                fgn + "_{}{}".format(56, "time"))
            outputs = []
            for oi, ok in enumerate([tag[t] for t in [-3, -2, -1]]):
                otmp = "{}\n".format(ok)
                for t1, t2 in zip(Ys_avg[:, oi], Ys_std[:, oi]):
                    otmp += " & {}".format(_encode_sign(t1, t2, 4))
                outputs.append(otmp)
        elegant_print(tag[pt_i], logger)
        elegant_print(choose_key, logger)
        elegant_print(outputs, logger)
        elegant_print("\n", logger)
        return


class GatherF_Prunings(PlotF_Prunings):
    def merge_sub_data(self, res_all, logger=None):
        name_ens_set = ['Bagging', 'AdaBoostM1', 'SAMME']
        name_pru_set = ['Entire', 'EPAF-C', 'EPAF-D:2', 'EPAF-D:3',
                        'POEPAF', 'POPEP']
        name_fa1_set = ['bagging', 'adaboost', 'LightGBM']
        name_fa2_set = ['FairGBM (fpr)', 'FairGBM (fnr)',
                        'FairGBM (fpr,fnr)', 'AdaFair']

        domestic_key = ['{} & {}'.format(
            i, j) for i in name_ens_set for j in name_pru_set]
        domestic_key.extend(name_fa1_set)
        sensitive_attributes = res_all[1]
        for sa in sensitive_attributes:
            domestic_key.extend([
                '{} via {}'.format(i, sa) for i in name_fa2_set])
        domestic_key = [_little_helper(
            i, sensitive_attributes) for i in domestic_key]

        e_i = [0, ]
        p_i = [0, 1, 2, 5]
        f_i = [0, 3]
        s_i = [0, 1]
        j_i = [20]
        ret_key, ret_idr, ret_idc = {}, {}, {}
        tmp_1 = np.arange(3 * 6).reshape(3, 6)
        tmp_1 = tmp_1[e_i, :][:, p_i].reshape(-1).tolist()
        for s_i, s_k in enumerate(sensitive_attributes):
            tmp_2 = np.arange(2 * 4).reshape(2, 4) + 21
            tmp_2 = tmp_2[[s_i], :][:, f_i].reshape(-1).tolist()
            choose_idx = tmp_1 + j_i + tmp_2
            choose_key = [domestic_key[i] for i in choose_idx]
            ret_key[s_k] = choose_key
            ret_idr[s_k] = choose_idx

            if s_i == 0:
                ret_idc[s_k] = list(range(0, 56))
            elif s_i == 1:
                ret_idc[s_k] = list(range(56, 112))
            ret_idc[s_k] += [168, 169, 170]
        return ret_key, ret_idr, ret_idc, sensitive_attributes

    def schedule_mspaint(self, res_data, res_all,
                         optional_data, figname="",
                         jt=False, logger=None):
        idx = list(range(56, 112)) + list(range(168, 224)) + list(
            range(280, 336)) + list(range(336, 339))
        new_data = {i: self.prepare_graph(np.array(
            res_data[i]))[:, idx, :] for i in optional_data}
        elegant_print("mCV_expt6\n", logger)
        self._tag = {
            0: "Accuracy (%)",
            1: "Precision",
            2: "Recall",
            3: "f1_score",
            4: "fpr",
            5: "fnr",
            6: "Sensitivity",
            7: "Specificity",
            8: "G_mean",
            9: "dp",
            10: "Matthew",
            11: "Cohen",
            12: "random_acc",
            13: "Disturbed Acc (%)",
            16: "Disturbed f1",

            39: "unaware",  # 39,40, 49 =39+10
            41: "DP",       # 41,42, 50 =41+9
            43: "EO",       # 43,44, 51 =53+8
            45: "PQP",      # 45,46, 52 =45+7
            47: "manual",   # 47,48, 53 =47+6
            49: "unaware",
            50: "DP",
            51: "EO",
            52: "PQP",
            53: "manual",
            54: r"$L_{fair}$",  # 54: r"$\hat{L}_{fair}$",
            55: r"$L_{err}$",   # 55: r"$\hat{L}_{err}$",  # loss
            -3: r"$T_{ens}$",
            -2: r"$T_{pru}$",
            -1: r"$T_{all}$",  # "Time Cost (sec)",
        }

        self.tabulate_output(new_data, res_all, optional_data, 0,
                             figname, logger=logger)
        self.tabulate_output(new_data, res_all, optional_data, 3,
                             figname, logger=logger)
        self.tabulate_output(new_data, res_all, optional_data, 50,
                             figname, logger=logger)
        self.tabulate_output(new_data, res_all, optional_data, 51,
                             figname, logger=logger)
        self.tabulate_output(new_data, res_all, optional_data, 52,
                             figname, logger=logger)
        self.tabulate_output(new_data, res_all, optional_data, 54,
                             figname, logger=logger)

    def tabulate_output(self, new_data, res_all,
                        optional_data, pt_i,
                        fgn="", ddof=0, logger=None):
        Ys_avg = np.zeros((9, 7))
        Ys_std = np.zeros((9, 7))
        Ys_i = 0
        rez = 2 if pt_i in [0, 13, 26] else 4
        reorder = [4, 5, 6, 0, 1, 2, 3]
        mode = "ascend" if pt_i >= 39 else "descend"

        for i in optional_data:
            ret_key, ret_r, ret_c, sens_attr = self.merge_sub_data(
                res_all[i], logger=logger)
            for _, sk in enumerate(sens_attr):  # for si,sk in
                alt_data = new_data[i][:, ret_c[sk]]
                alt_data = alt_data[ret_r[sk], :][:, pt_i]
                alt_data = alt_data.astype(DTY_FLT)
                alt_data = alt_data[reorder, :]
                if pt_i in [0, 13, 26]:
                    alt_data *= 100.
                Ys_avg[Ys_i] = alt_data.mean(axis=1)
                Ys_std[Ys_i] = alt_data.std(axis=1, ddof=ddof)

                re_key = [ret_key[sk][ri] for ri in reorder]
                if pt_i == 0:
                    elegant_print(["\n"] + re_key, logger)
                else:
                    elegant_print("\n", logger)
                elegant_print("data sens_attr: {} {}".format(i, sk), logger)
                otmp = "{}\n".format(self._tag[pt_i])
                for t1, t2 in zip(Ys_avg[Ys_i], Ys_std[Ys_i]):
                    otmp += " & {}".format(_encode_sign(t1, t2, rez))
                elegant_print(otmp, logger)

                otmp_1 = "{}\n".format('wtl')
                otmp_2 = "{}\n".format('avg')
                proposed = alt_data[6, :]
                sign_A, G_A = comp_t_sing(proposed, 5, rez)
                for j in range(6):
                    compared = alt_data[j, :]
                    sign_B, G_B = comp_t_sing(compared, 5, rez)
                    mk_mu, mk_s2 = comp_t_prep(proposed, compared)
                    otmp_1 += " & {}".format(cmp_paired_wtl(
                        G_A, G_B, mk_mu, mk_s2, mode))
                    otmp_2 += " & {}".format(cmp_paired_avg(
                        G_A, G_B, mode))
                otmp_1 += " & ---"
                otmp_2 += " & ---"
                elegant_print([otmp_1, otmp_2], logger)
                del otmp_1, otmp_2, otmp, compared, proposed
                del sign_A, sign_B, G_A, G_B, mk_mu, mk_s2

                Ys_i += 1
        # mode = "ascend" if pt_i >= 39 else "descend"
        _, idx_bar = Friedman_init(Ys_avg, mode=mode)  # rank,
        fgn += "_{}{}".format(pt_i, self._tag[pt_i][:3])
        if pt_i in [0, 54]:
            Friedman_chart(idx_bar, re_key, fgn + "_fried5",
                           alpha=.05, logger=logger, anotCD=True)
        otmp = np.mean(idx_bar, axis=0).tolist()
        otmp = ["{:.4f}".format(j) for j in otmp]
        otmp = " & ".join(otmp)
        elegant_print("avg.rank & " + otmp, logger)
        del otmp
        annots = {
            0: r"aggr.rank.accuracy",
            1: r"aggr.rank.precision",
            2: r"aggr.rank.recall",
            3: r"aggr.rank.f1_score",
            50: r"aggr.rank.DP",
            51: r"aggr.rank.EO",
            52: r"aggr.rank.PQP",
            54: r"aggr.rank.$L_{fair}$",
            55: r"aggr.rank.$L_{err}$",
        }
        kwargs = {"cmap_name": 'GnBu', "rotation": 35}  # 65
        if 50 <= pt_i <= 52:
            kwargs["cmap_name"] = 'Greens'
        elif pt_i >= 54:
            kwargs["cmap_name"] = 'Greens'
        elif 0 <= pt_i < 13:
            kwargs["cmap_name"] = 'Blues'
        if pt_i in annots.keys():
            kwargs["annots"] = annots[pt_i]
        stat_chart_stack(idx_bar, re_key, fgn + "_stack", **kwargs)
        elegant_print("\n\n", logger)


# =====================================
# Experiments (representation)
# =====================================


class Renew_PlotF_Prunings(PlotF_Prunings):
    def renew_schedule_msgraph(self, res_data, res_all, optional_data,
                               figname='', jt=False, logger=None):
        return


class Renew_GatherF_Prunings(GatherF_Prunings):
    def renew_schedule_msgraph(self,
                               res_data, res_all, optional_data,
                               figname='', jt=False, logger=None):
        idx = list(range(56, 112)) + list(range(168, 224)) + list(
            range(280, 336)) + list(range(336, 339))
        new_data = {i: self.prepare_graph(np.array(
            res_data[i]))[:, idx, :] for i in optional_data}
        elegant_print("renew mCV_expt6\n", logger)

        # for i in [0, 3, 50, 51, 52, 54]:
        #   self.renew_tabulate_output(
        #       new_data, res_all, optional_data, i, figname, logger=logger)
        _, Ys_model, Ys_acc = self.renew_retrieve_dat(
            new_data, res_all, optional_data, 0)
        # _, _, Ys_f1s = self.renew_retrieve_dat(
        #     new_data, res_all, optional_data, 3)
        _, _, Ys_DP = self.renew_retrieve_dat(
            new_data, res_all, optional_data, 50)
        _, _, Ys_EO = self.renew_retrieve_dat(
            new_data, res_all, optional_data, 51)
        _, _, Ys_PQP = self.renew_retrieve_dat(
            new_data, res_all, optional_data, 52)
        _, _, Ys_DR = self.renew_retrieve_dat(
            new_data, res_all, optional_data, 54)
        Ys_annot = ['DP', 'EO', 'PQP', 'DR']
        self.renew_graph_scatter(Ys_acc, [
            Ys_DP, Ys_EO, Ys_PQP, Ys_DR], Ys_model, Ys_annot,
            verbose=False)
        return

    def renew_retrieve_dat(self, new_data, res_all, optional_data,
                           pt_i, fgn='', ddof=0, logger=None):
        Ys_avg, Ys_std = np.zeros((9, 7)), np.zeros((9, 7))
        Ys_i, reorder = 0, [4, 5, 6, 0, 1, 2, 3]
        # rez = 2 if pt_i in [0, 13, 26] else 4
        # mode = 'ascend' if pt_i >= 39 else 'descend'
        Ys_entire = np.zeros((9, 7, 5))
        for i in optional_data:
            ret_key, ret_r, ret_c, sen_att = self.merge_sub_data(
                res_all[i], logger=logger)
            for _, sk in enumerate(sen_att):  # for si,sk in
                alt_data = new_data[i][:, ret_c[sk]]
                alt_data = alt_data[ret_r[sk], :][:, pt_i]
                alt_data = alt_data.astype(DTY_FLT)[reorder, :]
                # if pt_i in [0, 13, 26]:
                #   alt_data *= 100.
                Ys_avg[Ys_i] = alt_data.mean(axis=1)
                Ys_std[Ys_i] = alt_data.std(axis=1, ddof=ddof)
                Ys_entire[Ys_i] = alt_data
                Ys_i += 1
        Ys_entire_trans = Ys_entire.transpose(
            1, 0, 2).reshape(-1, 5 * 9)
        Ys_models = ret_key[sen_att[0]]
        Ys_models = [Ys_models[i] for i in reorder]
        # rank, idx_bar = Friedman_init(Ys_avg, mode=mode)
        return Ys_entire, Ys_models, Ys_entire_trans

    def renew_graph_scatter(self, X, Ys, Ys_model, Ys_annot,
                            verbose):
        # fig = plt.figure(dpi=300)
        # ax = fig.add_subplot(111)
        # multi_lin_reg_with_distr()
        # scatter_with_marginal_distrib()
        label_x = 'Performance (Accuracy)'  # f1_score
        kw_balance = {'alpha_loc': 'b4', 'alpha_rev': True}
        n = len(Ys_model)  # Ys.shape  (4, #model, #iteration)
        for i in range(4):
            FairGBM_scatter(
                X, Ys[i], Ys_model, (
                    label_x, 'Fairness ({})'.format(Ys_annot[i])),
                figname='exp6_fairgbm_scat_0acc_bias{}'.format(i))
            FairGBM_tradeoff_v3(
                X, Ys[i], Ys_model, ('error rate', Ys_annot[i]),
                figname='exp6_fairgbm_pct_0acc_bias{}'.format(i),
                num_gap=100, **kw_balance)
            if not verbose:
                continue
            FairGBM_tradeoff_v2(
                X, Ys[i], Ys_model, (r'$\alpha$', 'error rate'),
                figname='exp6_fairgbm_bal_0acc_bias{}'.format(i),
                num_gap=100, **kw_balance)
        if not verbose:
            return

        for i in range(4):
            new_X = X[:, 5:]      # (#model, #iteration')
            new_Y = Ys[i][:, 5:]  # (#model, #iteration')
            new_idx = np.argsort(new_X, axis=1)[:, -1]
            X_new = np.array([new_X[j][new_idx[j]] for j in range(n)])
            Y_new = np.array([new_Y[j][new_idx[j]] for j in range(n)])
            FairGBM_tradeoff_v3(
                new_X, new_Y, Ys_model, ('error rate', Ys_annot[i]),
                figname='exp6_fairep_pct_0acc_bias{}'.format(i),
                num_gap=100, **kw_balance)
            FairGBM_scatter(
                new_X, new_Y, Ys_model, (
                    label_x, 'Fairness ({})'.format(Ys_annot[i])),
                figname='exp6_fairep_scat_0acc_bias{}'.format(i))

            if not verbose:
                continue
            FairGBM_tradeoff_v1(
                X_new, Y_new, Ys_model, ('error rate', Ys_annot[i]),
                figname='exp6_fairep_sin_0acc_bias{}'.format(i),
                **kw_balance)  # 95% percentage
            FairGBM_tradeoff_v2(
                new_X, new_Y, Ys_model, (r'$\alpha$', 'error rate'),
                figname='exp6_fairep_bal_0acc_bias{}'.format(i),
                num_gap=100, **kw_balance)
        # pdb.set_trace()
        # fig = _setup_figsize(fig, 'L-WS', invt=False)
        # _setup_figshow(fig, figname + '_')
        return


# =====================================
# Benchmarks
# =====================================


class FVre_Drawing(DataSetup):
    def __init__(self, trial_type, nb_cls, nb_pru=None,
                 nb_iter=5, ratio=.5, lam=.5, data_type='ricci',
                 name_ens="AdaBoostM1", abbr_cls="DT",
                 partition=False, nb_lam=9,
                 gather=False, screen=True, logged=False):
        super().__init__(data_type)
        self._trial_type = trial_type
        self._nb_cls = nb_cls
        self._nb_pru = nb_pru
        self._nb_iter = nb_iter
        self._ratio = ratio
        self._lam = lam
        self._name_ens = name_ens
        self._abbr_cls = abbr_cls
        self._screen = screen
        self._logged = logged
        self._pn = partition

        self._log_document = "_".join([
            trial_type, "{}vs{}".format(nb_cls, nb_pru),
            "iter{}".format(nb_iter), self._log_document,
            "ratio{}".format(int(ratio * 100)), "pms"])
        if trial_type.endswith('expt6'):
            self._log_document += "_lam{}".format(int(lam * 100))
        if trial_type[-5:] in ('expt3'):
            nmens_tmp = _get_tmp_document(name_ens, nb_cls)
            self._log_document = "_".join([
                trial_type, nmens_tmp, "paint"])

        if self._trial_type.endswith('expt5'):
            self._iterator = PlotE_Measures() if not gather else GatherE_Measures()
        elif self._trial_type.endswith('expt6'):  # UPDATE
            # self._iterator = PlotF_Prunings() if not gather else GatherF_Prunings()
            self._iterator = Renew_PlotF_Prunings(
            ) if not gather else Renew_GatherF_Prunings()
        elif self._trial_type.endswith('expt4'):
            self._iterator = PlotD_Measures() if not gather else GatherD_Measures()
        elif trial_type.endswith('expt3'):
            self._iterator = PlotC_TheoremsLemma(
                name_ens, nb_cls,
                nb_iter=nb_iter, trial_type=trial_type)
        self._figname = "{}_{}".format(trial_type, data_type)
        if gather:
            self._figname = "{}_entire".format(trial_type)

    def trial_one_process(self, prefix):
        since = time.time()
        # START
        if self._trial_type[-5:] in ('expt6', 'expt4'):
            self.trial_one_proc_json(prefix=prefix)
        elif self._trial_type[-5:] in ('expt3',):
            self.trial_one_proc_csvr(prefix=prefix)
        # END
        tim_elapsed = time.time() - since
        elegant_print("\tDrawing: time cost {:.6f} minutes".format(
            tim_elapsed / 60), None)  # logger)
        return

    def trial_one_proc_csvr(self, prefix=''):
        trial_type = self._iterator.trial_type
        filename = self._iterator.get_raw_filename(trial_type)
        filename = filename.split('/')[-1]
        if prefix:
            filename = os.path.join(prefix, filename)
        # assert os.path.exists(filename), filename
        raw_dframe = self._iterator.load_raw_dataset(filename)
        if self._trial_type.endswith('expt10'):
            self._iterator.schedule_mspaint(raw_dframe, None)
        else:
            self._iterator.schedule_mspaint(raw_dframe, self._pn)
        return

    def trial_one_proc_json(self, prefix=''):
        # START
        if not prefix:
            json_rf = open(self._log_document + ".json", 'r')
        else:
            json_rf = open(os.path.join(
                prefix, self._log_document + '.json'), 'r')
        # json_rf = open(self._log_document + ".json", 'r')
        content = json_rf.read()
        json_reader = json.loads(content)
        res_all = json_reader['res_all']
        res_data = json_reader['res_data']
        # START

        if self._trial_type.endswith('expt5'):
            self._iterator.schedule_mspaint(
                res_data, res_all, self._figname, jt=True)
        elif self._trial_type.endswith('expt6'):
            if os.path.exists(self._figname + ".txt"):
                os.remove(self._figname + ".txt")
            logger, formatter, fileHandler = get_elogger(
                "fvote", self._figname + ".txt")
            self._iterator.schedule_mspaint(
                res_data, res_all, self._figname, False, logger)
            # pdb.set_trace()
            rm_ehandler(logger, formatter, fileHandler)
        else:
            self._iterator.schedule_mspaint(res_data, res_all,
                                            self._figname)

        # END
        json_rf.close()  # file to read
        del json_rf, content, json_reader
        del res_all, res_data  # , new_data
        # if self._trial_type.endswith('expt6'):
        #     rm_ehandler(logger, formatter, fileHandler)
        # END
        return

    def trial_gather_process(self, prefix=''):
        since = time.time()
        # START

        optional_data = ["ricci", "german", "adult", "ppr", "ppvr"]
        res_all, res_data = {}, {}
        for data_type in optional_data:
            log_document = "_".join([
                self._trial_type,
                "{}vs{}".format(self._nb_cls, self._nb_pru),
                "iter{}".format(self._nb_iter),
                data_type,
                "ratio{}".format(int(self._ratio * 100)),
                "pms"])
            if self._trial_type.endswith('expt6'):
                log_document += "_lam{}".format(int(self._lam * 100))

            if not prefix:
                json_rf = open(log_document + ".json", 'r')
            else:
                json_rf = open(os.path.join(
                    prefix, log_document + '.json'), 'r')
            content = json_rf.read()
            json_reader = json.loads(content)
            res_all[data_type] = json_reader['res_all']
            res_data[data_type] = json_reader['res_data']
            json_rf.close()
            del json_rf, content, json_reader

        if self._trial_type.endswith('expt5'):
            self._iterator.schedule_mspaint(
                res_data, res_all, optional_data,
                self._figname, True)  # jt=True)
        elif self._trial_type.endswith('expt6'):
            if os.path.exists(self._figname + ".txt"):
                os.remove(self._figname + ".txt")
            logger, formatter, fileHandler = get_elogger(
                "fvote", self._figname + ".txt")
            self._iterator.schedule_mspaint(
                res_data, res_all, optional_data,
                # self._figname, jt=False, logger=logger)
                self._figname, False, logger)
            self._iterator.renew_schedule_msgraph(
                res_data, res_all, optional_data, self._figname,
                False, logger)  # jt=False, logger=logger)
            rm_ehandler(logger, formatter, fileHandler)
        else:
            self._iterator.schedule_mspaint(
                res_data, res_all, optional_data, self._figname)
        del res_data, res_all, optional_data
        # if self._trial_type.endswith('expt6'):
        #     rm_ehandler(logger, formatter, fileHandler)

        # END
        tim_elapsed = time.time() - since
        elegant_print("Time period in total: {}".format(
            elegant_durat(tim_elapsed)), logger)
        return


# =====================================
# Experiments (corrected bounds)
# =====================================


# -------------------------------------
# fvote_draw.py
#   PlotC_TheoremsLemma
#   PlotK_PACGeneralisation


class CorrFigCK_bounds(GraphSetup):
    def load_raw_dataset(self, filename, sheetname):
        filepath = self.get_raw_filename('')  # 'mCV_exp11g'
        filepath = os.path.dirname(filepath)
        filepath = os.path.join(filepath, f'{filename}.xlsx')
        # import pandas as pd  # return dframe
        self._cmap_name = 'muted'
        return pd.read_excel(filepath, sheetname)

    def recap_sub_data(self, dframe, nb_row=4,
                       nc_norm=11, nc_sens=0):
        each_att = nc_sens * self._nb_iter
        each_gen = nc_norm * self._nb_iter
        each_set = each_att * 2 + each_gen + 1
        nb_set = len(dframe) - nb_row + 1
        nb_set = (nb_set + each_att) // each_set

        tmp = each_set - each_att  # i.e. tmp_diff
        id_set = [i * each_set + tmp for i in range(nb_set)]
        id_set = [i + nb_row - 1 for i in [0] + id_set]
        del tmp, each_gen, each_att, each_set
        return nb_set, id_set

    def prepare_graph(self):
        from pyfair.facil.utils_const import unique_column
        num = (5 + 12) * 2 + (6 + 4 * self._nb_cls)
        csv_row_1 = unique_column(12 + 1 + num)

        pms = csv_row_1[: 12 + 1]  # params last: Ensem/ut
        tmp_loss = csv_row_1[13: 13 + 10]
        tmp_bias = csv_row_1[23: 23 + 24]
        tag_pac = csv_row_1[47: 47 + 6]
        tmp_pac_alt = csv_row_1[47 + 6:]

        tag_trn = tmp_loss[:5] + tmp_bias[:12]
        tag_tst = tmp_loss[5:] + tmp_bias[12:]
        tag_pac_alt = []
        for i in range(self._nb_cls):
            tag_pac_alt.append(tmp_pac_alt[i * 4: (i + 1) * 4])
        del tmp_loss, tmp_bias, tmp_pac_alt, pms
        return tag_trn, tag_tst, tag_pac, tag_pac_alt

    # def plot_scatter_chart(self, df, tgx, tgy, fgn, ant=('X', 'Y')):
    #     X = df[tgx].values.astype(DTY_FLT)  # tag_X
    #     Y = df[tgy].values.astype(DTY_FLT)  # tag_Y
    #     kws = {"annots": ant, "identity": True, "figsize": 'L-WS'}
    #     if "lem32" in fgn:            # figname
    #         kws['diff'] = 0.05        # -.05
    #         kws['base'] = 0.05
    #     kws['locate'] = 'upper left'  # kwargs
    #     scatter_id_chart(X, Y, fgn, **kws)  # 'man_'+
    #     return

    def plot_scatter_chart(self, df_X, df_Y, fgn, ant=('X', 'Y')):
        X = df_X.values.astype(DTY_FLT)
        Y = df_Y.values.astype(DTY_FLT)
        if ('_ck' in fgn) or ('_rl' in fgn) or (
                '_pac' in fgn) or ('_lem' in fgn):
            multi_lin_reg_without_distr(
                X, [Y], [''], ant, fgn,  # +'_alt', # [ant[1]],
                figsize='S-NT',  # if '_pac' in fgn else 'S-WS',
                snspec='sty8a')  # 'sty4','sty3a','sty3b'
        else:  # thm1|3|4
            multi_lin_reg_without_distr(
                X, [Y], [''], ant, fgn, figsize='S-NT',
                snspec='sty3e')  # 'sty3b')
        if '_lem' in fgn:
            multi_lin_reg_without_distr(X, [Y], [
                ''], ant, fgn + '_als', snspec='sty8b')  # 'sty3a')
        kws = {"annots": ant, "identity": True, "figsize": 'L-WS'}
        if 'lem2' in fgn:
            kws['diff'] = .05
            kws['base'] = .05
        kws['locate'] = 'upper left'
        rmk = '' if ('_err' in fgn) or ('_re' in fgn) else '_prev'
        scatter_id_chart(X, Y, f"{fgn}{rmk}", **kws)
        return

    def obtn_dat_bnds(self, dframe, nb_set, id_set, tag):
        k = 0
        df_raw = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag]
        for k in range(1, nb_set):
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[k + 1]][tag]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw.reset_index(drop=True)

    def obtn_dat_pac_coll(self, dframe, tag_pac_alt,
                          tag_pac):  # collect.
        df_raw = dframe[tag_pac_alt[0] + [tag_pac[-1]]]
        for j in tag_pac_alt[1:]:
            columns = {t2: t1 for t1, t2 in zip(tag_pac_alt[0], j)}
            df_tmp = dframe[j + [
                tag_pac[-1], ]].rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw  # .reset_index(drop=True)

    def schedule_mspaint(self, raw_dframe, omit=True):
        nb_set, id_set = self.recap_sub_data(raw_dframe)
        tag_trn, tag_tst, tag_pac, tag_alt = self.prepare_graph()

        from pyfair.facil.utils_const import _get_tmp_name_ens
        tmp = _get_tmp_name_ens(self._name_ens)
        tag, fgn = tag_tst, f'{self._figname}_{tmp}_tst'
        # tag, fgn = tag_trn, f'{self._figname}_{tmp}_trn'
        df_raw = self.obtn_dat_bnds(raw_dframe, nb_set, id_set, tag)
        # df_set = self.obtn_dat_bnds(
        #     raw_dframe, 1, [id_set[1], id_set[2]], tag_tst)
        self.verify_theorem31(df_raw, tag, fgn, True)
        self.verify_theorem33(df_raw, tag, fgn, True)
        self.verify_lemma32(df_raw, tag, fgn)
        self.verify_theorem34(df_raw, tag, fgn, True)
        # self.verify_bounds(df_raw, tag, fgn)  # df_tmp,

        tmp_pac_alt = np.array(tag_alt).reshape(-1).tolist()
        df_raw = self.obtn_dat_bnds(
            raw_dframe, nb_set, id_set, tag_pac + tmp_pac_alt)
        df_tmp = self.obtn_dat_pac_coll(df_raw, tag_alt, tag_pac)
        self.verify_theorem36(df_raw, tag_pac, fgn[:-4])
        self.verify_theorem35(
            df_tmp, tag_alt[0] + [tag_pac[-1], ], fgn[:-4])
        # pdb.set_trace()

        rmk = '_prev'  # ''
        if omit:
            os.remove(f'{fgn}_ck_thm1{rmk}.pdf')
            os.remove(f'{fgn}_ck_thm3{rmk}.pdf')
            # os.remove(f'{fgn}_ck_thm4{rmk}.pdf')
            os.remove(f'{fgn}_rl_col7{rmk}.pdf')
            os.remove(f'{fgn}_rl_col8{rmk}.pdf')
            os.remove(f'{fgn[:-4]}_err_thm5.pdf')
            os.remove(f'{fgn[:-4]}_err_thm6.pdf')
            os.remove(f'{fgn[:-4]}_pac_thm5{rmk}.pdf')
            os.remove(f'{fgn[:-4]}_pac_thm6{rmk}.pdf')
            # os.remove(f'{fgn}_lem2_als.pdf')

        # os.remove(f'{fgn}_lem2{rmk}.pdf')
        # os.remove(f'{fgn}_err_thm1.pdf')
        # os.remove(f'{fgn}_err_thm3.pdf')
        # os.remove(f'{fgn}_err_thm4.pdf')
        os.remove(f'{fgn}_thm1{rmk}.pdf')
        os.remove(f'{fgn}_thm3{rmk}.pdf')
        os.remove(f'{fgn}_thm4{rmk}.pdf')
        return

    def schedule_mspaint_gather(self, raw_dfs, verbose=False):
        df_bnd, df_pac, df_pac_alt = [], [], []
        _, tag, tag_pac, tag_alt = self.prepare_graph()
        tmp_pac_alt = np.array(tag_alt).reshape(-1).tolist()
        for i, raw_dframe in enumerate(raw_dfs):
            nb_set, id_set = self.recap_sub_data(raw_dframe)
            fgn = f'{self._figname}_tst'
            df_bnd.append(self.obtn_dat_bnds(
                raw_dframe, nb_set, id_set, tag))
            tmp = self.obtn_dat_bnds(
                raw_dframe, nb_set, id_set, tag_pac + tmp_pac_alt)
            df_pac.append(tmp)
            df_pac_alt.append(self.obtn_dat_pac_coll(
                tmp, tag_alt, tag_pac))
        df_bnd = pd.concat(df_bnd, axis=0).reset_index(drop=True)
        df_pac = pd.concat(df_pac, axis=0).reset_index(drop=True)
        df_pac_alt = pd.concat(df_pac_alt, axis=0)

        self.verify_theorem31(df_bnd, tag, fgn, True)
        self.verify_theorem33(df_bnd, tag, fgn, True)
        self.verify_theorem34(df_bnd, tag, fgn, True)
        self.verify_lemma32(df_bnd, tag, fgn)
        os.remove(f'{fgn}_ck_thm1_prev.pdf')
        os.remove(f'{fgn}_ck_thm3_prev.pdf')
        os.remove(f'{fgn}_rl_col7_prev.pdf')
        os.remove(f'{fgn}_rl_col8_prev.pdf')
        if not verbose:
            os.remove(f'{fgn}_thm1_prev.pdf')
            os.remove(f'{fgn}_thm3_prev.pdf')
            os.remove(f'{fgn}_thm4_prev.pdf')
        fgn = fgn[:-4]
        self.verify_theorem36(df_pac, tag_pac, fgn)
        self.verify_theorem35(df_pac_alt, tag_alt[
            0] + [tag_pac[-1], ], fgn)
        os.remove(f'{fgn}_err_thm5.pdf')
        os.remove(f'{fgn}_err_thm6.pdf')
        os.remove(f'{fgn}_pac_thm5_prev.pdf')
        os.remove(f'{fgn}_pac_thm6_prev.pdf')
        del df_bnd, df_pac, df_pac_alt
        del fgn  # pdb.set_trace()
        return

    def verify_theorem31(self, df, tag, fgn, omit=False):
        annots = [
            r"$2\mathbb{E}_\rho[\mathcal{L}_{bias}(f)]$",
            r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$"]
        # figname = fgn + "_thm31"  # "31_"+kw
        df_X = df[tag[5 + 1]] * 2.  # tag[6]
        self.plot_scatter_chart(
            df_X, df[tag[5]], fgn + '_thm1', ant=annots)
        self.plot_scatter_chart(
            df[tag[5]], df_X, fgn + '_re_thm1', annots[::-1])

        annots[0] = r"$2\mathbb{E}\left[\frac{ \phi_\rho(\mathbf{x}) }{ \gamma_\rho(\mathbf{x}) }\right]$"
        annots.append(r'$f(x)=x$')
        df_Y, df_X = df[tag[5]], df[tag[5 + 5 + 0]] * 2.
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_ck_thm1', ant=annots)
        annots[0] = r"$\frac{2}{\gamma_0} \mathbb{E}_\rho[\mathcal{L}_{bias}(f)]+\eta$"
        df_X = df[tag[5 + 5 + 3 + 0]]
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_rl_col7', ant=annots)
        del df_Y, df_X  # '_rl_col5'

        annots = (
            r"$2\mathbb{E}_\rho[\mathcal{L}_{err}(f)]$",
            r"$\mathcal{L}_{err}(\mathbf{wv}_\rho)$")
        self.plot_scatter_chart(
            # df, tag[1], tag[0], fgn + '_err_thm1', annots)
            df[tag[1]] * 2., df[tag[0]], fgn + '_err_thm1', annots)
        if omit:
            os.remove(f"{fgn}_err_thm1.pdf")
            os.remove(f"{fgn}_re_thm1.pdf")
        return

    def verify_theorem33(self, df, tag, fgn, omit=False):
        annots = [
            r"$4\mathbb{E}_{\rho^2}[\mathcal{L}_{bias}(f,f^\prime)]$",
            r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$"]
        df_X = df[tag[5 + 3]] * 4.  # tag[8]
        self.plot_scatter_chart(
            df_X, df[tag[5]], fgn + '_thm3', ant=annots)
        self.plot_scatter_chart(
            df[tag[5]], df_X, fgn + '_re_thm3', annots[::-1])

        annots[0] = r"$4\mathbb{E}\left[\frac{ \phi_\rho(\mathbf{x})^2 }{ \gamma_\rho(\mathbf{x})^2 }\right]$"
        annots.append(r'$f(x)=x$')
        df_Y, df_X = df[tag[5]], df[tag[5 + 5 + 1]] * 4.
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_ck_thm3', ant=annots)
        annots[0] = r"$\frac{4}{\gamma_0^2} \mathbb{E}_{\rho^2}[\mathcal{L}_{bias}(f,f')]+\eta$"
        df_X = df[tag[5 + 5 + 3 + 3]]
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_rl_col8', ant=annots)
        del df_Y, df_X  # '_rl_col6'

        annots = (
            r"$4\mathbb{E}_{\rho^2}[\mathcal{L}_{err}(f,f^\prime)]$",
            r"$\mathcal{L}_{err}(\mathbf{wv}_\rho)$")
        self.plot_scatter_chart(
            # df, tag[3], tag[0], fgn + '_err_thm3', annots)
            df[tag[3]] * 4., df[tag[0]], fgn + '_err_thm3', annots)
        if omit:
            os.remove(f"{fgn}_err_thm3.pdf")
            os.remove(f"{fgn}_re_thm3.pdf")
        return

    def verify_lemma32(self, df, tag, fgn, omit=True):
        annots = (
            r"$\mathbb{E}_{\rho^2}[\mathcal{L}_{bias}(f,f^\prime)]$",
            r"$\mathbb{E}_\mathcal{D}[\mathbb{E}_\rho[\ell_{bias}(f,\mathbf{x})]^2]$")
        df_X, df_Y = df[tag[5 + 3]], df[tag[5 + 2]]  # tag[7],tag[8]
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_lem2', ant=annots)
        self.plot_scatter_chart(
            df_Y, df_X, fgn + '_re_lem2', annots[::-1])
        del df_X, df_Y

        annots = (
            r"$\mathbb{E}_{\rho^2}[\mathcal{L}_{err}(f,f^\prime)]$",
            r"$\mathbb{E}_\mathcal{D}[\mathbb{E}_\rho[\ell_{err}(f,\mathbf{x})]^2]$")
        self.plot_scatter_chart(
            # df, tag[3], tag[2], fgn + '_err_lem2', annots)
            df[tag[3]], df[tag[2]], fgn + '_err_lem2', annots)
        if omit:
            os.remove(f"{fgn}_err_lem2.pdf")
            os.remove(f"{fgn}_re_lem2.pdf")
            # os.remove(f"{fgn}_err_lem2_alt.pdf")
            # os.remove(f"{fgn}_re_lem2_alt.pdf")
            os.remove(f"{fgn}_err_lem2_als.pdf")
            os.remove(f"{fgn}_re_lem2_als.pdf")
        os.remove(f'{fgn}_lem2_prev.pdf')
        os.remove(f'{fgn}_lem2_als.pdf')
        return

    def verify_theorem34(self, df, tag, fgn, omit=False):
        annots = [
            r"$\mathrm{RHS}$ in Theorem 3.3",  # $\mathbf{RHS}$
            r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$"]
        annots[0] = r"$\frac{ \mathbb{E}_{\rho^2}[\mathcal{L}_{bias}(f,f')] -\mathbb{E}_\rho[\mathcal{L}_{bias}(f)]^2 }{ \mathbb{E}_{\rho^2}[\mathcal{L}_{bias}(f,f')] -\mathbb{E}_\rho[\mathcal{L}_{bias}(f)]+\frac{1}{4} }$"
        df_Y, df_X = df[tag[5]], df[tag[5 + 4]]  # tag[9]
        df_Z = (df[tag[5 + 1]] < 1. / 2).values
        self.plot_scatter_chart(
            df_X[df_Z], df_Y[df_Z], fgn + '_thm4', ant=annots)
        self.plot_scatter_chart(
            df_Y[df_Z], df_X[df_Z], fgn + '_re_thm4', annots[::-1])

        # annots[0] = r"$\mathrm{RHS}$ in Theorem 3.3"
        annots[0] = r"$\frac{ \mathbb{E}_\mathcal{D}[\phi_\rho(\mathbf{x})^2/\gamma_\rho(\mathbf{x})^2] -\mathbb{E}_\mathcal{D}[\phi_\rho(\mathbf{x})/\gamma_\rho(\mathbf{x})]^2 }{ \mathbb{E}_\mathcal{D}[\phi_\rho(\mathbf{x})^2/\gamma_\rho(\mathbf{x})^2] -\mathbb{E}_\mathcal{D}[\phi_\rho(\mathbf{x})/\gamma_\rho(\mathbf{x})]+\frac{1}{4} }$"
        annots.append(r'$f(x)=x$')
        df_X = df[tag[5 + 5 + 2]]  # df_Y,=df[tag[5]],
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_ckp_thm4', ant=annots)
        df_Z = (df[tag[5 + 5 + 0]] < 1. / 2).values
        # pdb.set_trace()
        df_Y, df_X = df_Y[df_Z], df_X[df_Z]
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_ck_thm4', ant=annots)
        del df_Y, df_X, df_Z

        annots = [
            r"$\mathrm{RHS}$ in Theorem 3.4",  # $\mathbf{RHS}$
            r"$\mathcal{L}_{err}(\mathbf{wv}_\rho)$"]
        annots[0] = r"$\frac{ \mathbb{E}_{\rho^2}[\mathcal{L}_{err}(f,f')] -\mathbb{E}_\rho[\mathcal{L}_{err}(f)]^2 }{ \mathbb{E}_{\rho^2}[\mathcal{L}_{err}(f,f')] -\mathbb{E}_\rho[\mathcal{L}_{err}(f)]+\frac{1}{4} }$"
        self.plot_scatter_chart(
            # df, tag[4], tag[0], fgn + '_err_thm4', annots)
            df[tag[4]], df[tag[0]], fgn + '_err_thm4', annots)
        if omit:
            os.remove(f"{fgn}_err_thm4.pdf")
            os.remove(f"{fgn}_re_thm4.pdf")
            os.remove(f"{fgn}_ckp_thm4_prev.pdf")  # _ckp_thm4
            os.remove(f"{fgn}_ckp_thm4.pdf")   # _ckp_thm4_alt
        return

    # def verify_bounds(self, df, tag, fgn, omit=False):
    #     import matplotlib.pyplot as plt
    #     from pyfair.facil.draw_prelim import (
    #         _setup_figsize, _setup_figshow, _setup_config)
    #
    #     annots = (
    #         r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$",
    #         r"$2\mathbb{E}_\rho[\mathcal{L}_{bias}(f)]$",
    #         r"$4\mathbb{E}_{\rho^2}[\mathcal{L}_{bias}(f,f^\prime)]$",
    #         r"$\mathrm{RHS}$ in Theorem 3.4")
    #     X = df[tag[5 + 0]].values.astype(DTY_FLT)
    #     Ys = df[[tag[5 + 1], tag[5 + 3], tag[5 + 4]]
    #             ].values.astype(DTY_FLT)
    #     fig = plt.figure(figsize=_setup_config['L-NT'])
    #     plt.plot(X, X, 'k--', label=r'$f(x)=x$')
    #     plt.plot(X, Ys[:, 0] * 2., '.', label=annots[1])
    #     plt.plot(X, Ys[:, 1] * 4., '.', label=annots[2])
    #     plt.plot(X, Ys[:, 2], '.', label=annots[3])
    #     plt.xlabel(annots[0])
    #     plt.legend(loc='best')
    #     _setup_figsize(fig, figsize='M-WS')
    #     _setup_figshow(fig, figname=fgn + '_all')
    #
    #     annots = (
    #         r"$\mathcal{L}_{err}(\mathbf{wv}_\rho)$",
    #         r"$2\mathbb{E}_\rho[\mathcal{L}_{err}(f)]$",
    #         r"$4\mathbb{E}_{\rho^2}[\mathcal{L}_{err}(f,f^\prime)]$",
    #         r"$\mathrm{RHS}$ in Theorem 3.4")
    #     X = df[tag[0]].values.astype(DTY_FLT)
    #     Ys = df[[tag[1], tag[3], tag[4]]].values.astype(DTY_FLT)
    #     # ind = np.arange(len(X))
    #     fig = plt.figure(figsize=_setup_config['L-NT'])
    #     # plt.plot(ind, X, 'r.', label=annots[0])
    #     # plt.plot(ind, Ys[:, 0], '.', label=annots[1])
    #     # plt.plot(ind, Ys[:, 1], '.', label=annots[2])
    #     # plt.plot(ind, Ys[:, 2], '.', label=annots[3])
    #     plt.plot(X, X, 'k--', label=f'$f(x)=x$')  # 'b--'
    #     plt.plot(X, Ys[:, 0], '.', label=annots[1])
    #     plt.plot(X, Ys[:, 1], '.', label=annots[2])
    #     plt.plot(X, Ys[:, 2], '.', label=annots[3])
    #     plt.xlabel(annots[0])
    #     plt.legend(loc='best')
    #     _setup_figsize(fig, figsize='M-WS')
    #     _setup_figshow(fig, figname=fgn + '_err_all')
    #     return

    def verify_theorem36(self, df, tag, fgn):  # ,omit=False):
        annots = [
            r"$\hat{\mathcal{L}}_{bias}(\mathbf{wv}_\rho) +\sqrt{\frac{1}{2n}+\ln\frac{|\mathcal{F}|}{\delta}}$",
            # r"$\mathrm{RHS}$ in Theorem 3.6",
            r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$", ]
        df_Y, df_X = df[tag[2]], df[tag[3]] + df[tag[4]]
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_pac_thm5', ant=annots)

        annots = [
            r"$\hat{\mathcal{L}}_{err}(\mathbf{wv}_\rho) +\sqrt{\frac{1}{2n}+\ln\frac{|\mathcal{F}|}{\delta}}$",
            r"$\mathcal{L}_{err}(\mathbf{wv}_\rho)$", ]
        df_Y, df_X = df[tag[0]], df[tag[1]] + df[tag[4]]
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_err_thm5', annots)
        return

    def verify_theorem35(self, df, tag, fgn):
        annots = [
            r"$\hat{\mathcal{L}}_{bias}(f,S) +\sqrt{\frac{1}{2n}+\ln\frac{1}{\delta}}$",
            # r"$\mathrm{RHS}$ in Theorem 3.5",
            r"$\mathcal{L}_{bias}(f)$", ]  # r"$\mathcal{L}_{fair}(f)$",
        df_Y, df_X = df[tag[2]], df[tag[3]] + df[tag[4]]
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_pac_thm6', ant=annots)

        annots = [
            r"$\hat{\mathcal{L}}_{err}(f,S) +\sqrt{\frac{1}{2n}+\ln\frac{1}{\delta}}$",
            r"$\mathcal{L}_{err}(f)$", ]
        df_Y, df_X = df[tag[0]], df[tag[1]] + df[tag[4]]
        self.plot_scatter_chart(
            df_X, df_Y, fgn + '_err_thm6', annots)
        return


# -------------------------------------
