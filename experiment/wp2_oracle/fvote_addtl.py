# coding: utf-8

import numpy as np
import pdb

from fairml.widget.utils_const import DTY_FLT, _get_tmp_name_ens
from fairml.widget.utils_saver import (
    get_elogger, rm_ehandler, elegant_print)

from fairml.facil.draw_hypos import (  # .graph.utils_hypos
    _encode_sign, Friedman_init, Friedman_test,
    cmp_paired_wtl, cmp_paired_avg, comp_t_sing, comp_t_prep)
from fairml.facil.draw_graph import (
    Friedman_chart, stat_chart_stack, multiple_hist_chart)
from fairml.facil.draw_chart import (
    multiple_scatter_chart, analogous_confusion, single_hist_chart,
    multiple_scatter_alternative, analogous_confusion_alternative,
    analogous_confusion_extended)

from fairml.facil.draw_addtl import (
    _subproc_pl_lin_reg, _subproc_pl_lin_reg_alt,
    line_reg_with_marginal_distr, scatter_with_marginal_distrib,
    FairGBM_scatter, FairGBM_tradeoff_v1, FairGBM_tradeoff_v2,
    FairGBM_tradeoff_v3)
from experiment.wp2_oracle.fetch_data import GraphSetup


# =====================================
# Experiments (additional)
# =====================================


# -------------------------------------
# To verify the proposed measure

class PlotA_Measures(GraphSetup):
    def __init__(self):
        pass

    def prepare_graph(self, res_data):
        # res_data.shape (#iter, #sen_att, #name_ens, #eval)
        new_data = np.zeros_like(res_data).transpose(1, 2, 3, 0)
        nb_iter, nb_attr, nb_ens, nb_eval = np.shape(res_data)
        for i in range(nb_attr):
            for j in range(nb_ens):
                for k in range(nb_eval):
                    new_data[i, j, k] = res_data[:, i, j, k]
        del nb_iter, nb_attr, nb_ens, nb_eval
        return new_data

    def schedule_mspaint(self, res_data, res_all, figname=""):
        # rew_data.shape: (#iter, #sen_att, #name_ens, #eval)
        # new_data.shape: (#sen_att, #name_ens, #eval, #iter)
        new_data = self.prepare_graph(np.array(res_data))[:, :, 30:, :]

        data_name, binary, nb_cls, _, nb_iter, _ = res_all[0]
        sensitive_attributes = res_all[1]
        ensemble_methods = res_all[-1]
        idx = [0, 1, 2, 3, 4, 6]
        ensemble_methods = np.array(ensemble_methods)[idx].tolist()

        for sa, sens_attr in enumerate(sensitive_attributes):
            # .shape: (#name_ens, #eval, #iter)
            fgn = "{}_{}_".format(figname, sens_attr)
            curr = new_data[sa][idx]  # shape (#ens',30,5)
            self.plot_multiple_hist_chart(curr, ensemble_methods, "acc", fgn)
            self.plot_multiple_hist_chart(curr, ensemble_methods, "fair", fgn)
            del curr, fgn

    def plot_multiple_hist_chart(self, curr, ens, mark="acc", fgn=""):
        if mark == "acc":
            idx = [0, 1, 2, 3]  # without sen/spe
            key = ["Accuracy", "Precision", "Recall", "f1_score"]
        elif mark == "sen":
            idx = [6, 7, 8, 10, 11]  # data_imbalance
            key = ["Sensitivity", "Specificity", "G_mean", "Matthew", "Cohen"]
        elif mark == "fair":
            # idx = [23, 24, 25, 26, 27, 28]  # fairness
            # key = ["unaware", "DP", "EO", "PQP", "Manual", "DR"]
            idx = [24, 25, 26, 28]
            key = ["DP", "EO", "PQP", "DR"]  # or "FQ (ours)"
        mode = 'ascend' if mark == "fair" else "descend"

        new_curr = curr[:, idx, :]  # .shape (#alg,#eval,#iter)
        Ys_avg = new_curr.mean(axis=2).T  # .shape (#eval,#alg)
        Ys_std = new_curr.std(axis=2, ddof=1).T
        multiple_hist_chart(Ys_avg, Ys_std, key, '', ens,
                            figname=fgn + mark, rotate=20)
        return


class PlotD_Measures(PlotA_Measures):
    def schedule_mspaint(self, res_data, res_all, figname=""):
        # res_data.shape: (#iter, #sen_att, #name_ens, #eval)
        # new_data.shape: (#sen_att, #name_ens, #eval, #iter)
        new_data = np.array(res_data)
        new_data = self.prepare_graph(new_data)[:, :, 56:, :]

        data_name, binary, nb_cls, _, nb_iter, _ = res_all[0]
        sensitive_attributes = res_all[1]
        ensemble_methods = res_all[-1]
        ensemble_methods = [
            i.replace('FPR', 'fpr').replace('FNR', 'fnr') 
            if 'FairGBM' in i else i for i in ensemble_methods]
        idx = [0, 1, 2, 3, 4, 6]
        ensemble_methods = [ensemble_methods[i] for i in idx]

        for sa, sens_attr in enumerate(sensitive_attributes):
            fgn = "{}_{}_".format(figname, sens_attr)
            curr = new_data[sa][idx]  # shape (#ens,57,5)
            del fgn, curr

        num_s, num_e, num_v, _ = new_data.shape
        alt_data = np.concatenate([
            new_data[i] for i in range(num_s)], axis=2)
        # shape (7,57,5*2)= (#ens,#eval,#iter*#att)
        alt_data = np.concatenate([
            alt_data[i] for i in range(num_e)], axis=1)
        # shape   (57,70)= (#eval,#iter*#att*#ens)
        X, Ys = alt_data[26], alt_data[[50, 51, 52, 54], :]
        # annots = ('diff(Accuracy)', 'Fairness Measure')
        annots = (r"$\Delta$(Accuracy)", "Fairness Measure")
        annotZs = ('DP', 'EO', 'PQP', 'DR')
        fgn = figname + "_correlation"
        multiple_scatter_chart(X, Ys, annots, annotZs, fgn)
        multiple_scatter_chart(X, Ys, annots, annotZs, fgn,
                               ind_hv='v', identity=False)

        Mat = alt_data[[26, 27, 28, 29, 32, 33, 50, 51, 52, 54]]
        # key = ["Accuracy", "Precision", "Recall", "f1_score",
        #        "Sensitivity", "Specificity"]
        key = ["Acc", "P", "R", "f1", "Sen", "Spe"]
        key = [r"$\Delta$({})".format(i) for i in key
               ] + ["DP", "EO", "PQP", "DR"]
        fgn = figname + "_confusion"
        analogous_confusion(Mat, key, fgn, normalize=False)
        return

    # def plot_multiple_scatter(self):
    #   pass

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

        if mark.endswith("norm"):
            pass
        elif mark.endswith("advr"):
            idx = [i + 13 for i in idx]
        elif mark.endswith("abs_"):
            idx = [i + 26 for i in idx]
        mode = "ascend" if mark == "fair" else "descend"

        new_curr = curr[:, idx, :]  # .shape (#alg,#eval,#iter)
        Ys_avg = new_curr.mean(axis=2).T  # .shape (#eval,#alg)
        # Ys_std = new_curr.std(axis=2, ddof=1).T
        Ys_std = new_curr.std(axis=2, ddof=ddof).T
        multiple_hist_chart(Ys_avg, Ys_std, key, '', ens,
                            figname=fgn + mark, rotate=20)
        return


class GatherD_Measures(PlotD_Measures):
    def schedule_mspaint(self, res_data, res_all,
                         optional_data, figname="", verbose=False):
        # each res_data.shape (#iter,#att,#ens,#eval) =(5,1|2,7,113)
        # each new_data.shape (#att,#ens,#eval,#iter) =(1|2,7,113,5)
        new_data = [self.prepare_graph(np.array(
            res_data[i]))[:, :, 56:, :] for i in optional_data]

        alt_data = np.concatenate(new_data, axis=0)  # (9,7,57,5)
        num_s, num_e, num_v, _ = alt_data.shape
        alt_data = np.concatenate([
            alt_data[i] for i in range(num_s)], axis=2)  # (7,57,45)
        alt_data = np.concatenate([
            alt_data[i] for i in range(num_e)], axis=1)  # (57,315)

        X, Ys = alt_data[26], alt_data[[50, 51, 52, 54], :]
        annots = (r"$\Delta$(Accuracy)", "Fairness Measure")
        annotZs = ('DP', 'EO', 'PQP', 'DR')
        fgn = figname + "_correlation"
        if verbose:
            multiple_scatter_chart(X, Ys, annots, annotZs, fgn)
        multiple_scatter_chart(X, Ys, annots, annotZs, fgn,
                               ind_hv='v', identity=False)

        Mat = alt_data[[26, 27, 28, 29, 32, 33, 50, 51, 52, 54]]
        key = ["Acc", "P", "R", "f1", "Sen", "Spe"]
        key = [r"$\Delta$(%s)" % i for i in key
               ] + ["DP", "EO", "PQP", "DR"]
        fgn = figname + "_confusion"
        if verbose:
            analogous_confusion(Mat, key, fgn, normalize=False)
        analogous_confusion_extended(
            Mat[-4:], Mat[[0, 1, 2, 3, 5]], key[-4:],
            [key[i] for i in [0, 1, 2, 3, 5]], fgn,
            figsize='M-WS', cmap_name='Blues', rotate=0)
        # pdb.set_trace()
        return


class PlotE_Measures(PlotD_Measures):
    def schedule_mspaint(self, res_data, res_all, figname="",
                         jt=False):
        # res_data.shape: (#iter, #sen_att, #name_ens, #eval)
        # new_data.shape: (#sen_att, #name_ens, #eval, #iter)
        new_data = np.array(res_data)
        idx = list(range(56, 112)) + list(
            range(168, 224)) + list(range(280, 336)) + [337 - 1, ]
        # attr_A = list(range(56, 112))
        # attr_B = list(range(168, 224))
        # attr_J = list(range(280, 336))
        # idx = attr_A + attr_B + attr_J + [337 - 1, ]
        new_data = self.prepare_graph(new_data)[:, :, idx, :]

        data_name, binary, nb_cls, _, nb_iter, _ = res_all[0]
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

        # new_data.shape: (1|2,7,169,5)= (#sen_att,#name_ens,#eval',#iter)
        attr_A = list(range(56))
        attr_B = list(range(56, 112))
        attr_J = list(range(112, 168))
        test_A = new_data[:, :, attr_A, :].astype(DTY_FLT)
        if len(sensitive_attributes) > 1:
            test_B = new_data[:, :, attr_B, :].astype(DTY_FLT)
            test_J = new_data[:, :, attr_J, :].astype(DTY_FLT)
        # test_*.shape: (1|2,7,56,5)= (#sen_att,#name_ens,#eval',#iter)

        num_s, num_e, num_v, _ = new_data.shape
        test_A = np.concatenate([  # (7, 56, 5* 1|2)
            test_A[i] for i in range(num_s)], axis=2)
        test_A = np.concatenate([  # (56, 5* 1|2 *7)
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
        key = [
            r"$\Delta$(%s)" % i for i in key] + [
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
        num_s, num_e, num_v, _ = new_data.shape  # shape (1|2,7,169,5)
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
        # test_?.shape: (7,56,5* 1|2) = (#ens,#eval',?)
        # test_?.shape: (56,5* 1|2 *7)= (#eval,??)

        if len(sensitive_attributes) == 1:
            return test_A  # .shape (56, 5* 1|2 *7)
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
        key = [
            r"$\Delta$(%s)" % i for i in key] + ["DP", "EO", "PQP", "DR"]
        fgn_f = figname + "_confusion" + "_{}".format(str(jt)[0])

        X, Ys = alt_data[26], alt_data[[50, 51, 52, 54], :]
        multiple_scatter_chart(X, Ys, annots, annotZs,
                               fgn_r, ind_hv='v', identity=False)
        Mat = alt_data[[26, 27, 28, 29, 32, 33, 50, 51, 52, 54]]
        analogous_confusion(Mat, key, fgn_f, normalize=False)
        return


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


class PlotF_Prunings(PlotD_Measures):
    def prepare_graph(self, res_data):
        # res_data.shape (5,29,339)= (#iter,#ens/pru,#eval)
        new_data = np.zeros_like(res_data).transpose(1, 2, 0)  # (29,339,5)
        nb_iter, nb_ens, nb_eval = np.shape(res_data)          # (5,29,339)
        for i in range(nb_ens):
            for j in range(nb_eval):
                new_data[i, j] = res_data[:, i, j]
        del nb_iter, nb_ens, nb_eval
        return new_data

    def schedule_mspaint(self, res_data, res_all, figname="",
                         jt=False, logger=None):
        # res_data.shape: (#iter, #ens, #eval)
        # new_data.shape: (#ens, #eval, #iter)
        # res_data.shape= (5, 29, 339)
        # new_data.shape= (29, 339, 5) -> (29, 171, 5)
        new_data = np.array(res_data)  # .transpose(1, 2, 0)
        idx = list(range(56, 112)) + list(range(168, 224)) + list(
            range(280, 336)) + list(range(336, 339))  # three times
        new_data = self.prepare_graph(new_data)[:, idx, :]

        data_name, binary, nb_cls, nb_pru, nb_iter, _ = res_all[0]
        sensitive_attributes = res_all[1]
        name_ens_set, name_pru_set = res_all[3], res_all[4]  # -4/-3
        domestic_key = ['{} & {}'.format(
            i, j) for i in name_ens_set for j in name_pru_set]
        domestic_key.extend(res_all[-2])
        for sa in sensitive_attributes:
            domestic_key.extend([
                '{} via {}'.format(i, sa) for i in res_all[-1]])
        domestic_key = [_little_helper(
            i, sensitive_attributes) for i in domestic_key]

        e_i = [0, ]         # [0, 1, 2]
        p_i = [0, 1, 3, 5]  # [0, 1, 2, 3, 4, 5]
        f_i = [0, 3]  # [0,1,3]     # [0, 1, 2, 3]
        s_i_set = [0, 1] if data_name != 'ricci' else [0]
        j_i = [20]          # [18, 19, 20]
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
        # each.new_data .shape (25|29, 339->171, 5)
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

    def tabulate_output(self, new_data, res_all, optional_data, pt_i,
                        fgn="", ddof=0, logger=None):
        Ys_avg = np.zeros((9, 7))  # 7*(1+2*4) =63
        Ys_std = np.zeros((9, 7))
        Ys_i = 0
        rez = 2 if pt_i in [0, 13, 26] else 4
        reorder = [4, 5, 6, 0, 1, 2, 3]
        mode = "ascend" if pt_i >= 39 else "descend"

        for i in optional_data:
            ret_key, ret_r, ret_c, sens_attr = self.merge_sub_data(
                res_all[i], logger=logger)
            for si, sk in enumerate(sens_attr):
                alt_data = new_data[i][:, ret_c[sk]]  # .shape (25|29, 171,5) ->
                alt_data = alt_data[ret_r[sk], :][:, pt_i]  # (7,56+3,5) ->(7,5)
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
                # elegant_print(["\n", i], logger)
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
                    otmp_1 += " & {}".format(
                        cmp_paired_wtl(G_A, G_B, mk_mu, mk_s2, mode))
                    otmp_2 += " & {}".format(
                        cmp_paired_avg(G_A, G_B, mode))
                otmp_1 += " & ---"
                otmp_2 += " & ---"
                elegant_print([otmp_1, otmp_2], logger)
                del otmp_1, otmp_2, otmp, compared, proposed
                del sign_A, sign_B, G_A, G_B, mk_mu, mk_s2

                Ys_i += 1
        # mode = "ascend" if pt_i >= 39 else "descend"
        rank, idx_bar = Friedman_init(Ys_avg, mode=mode)
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
            kwargs["cmap_name"] = 'Greens'  # 'autumn'
        elif pt_i >= 54:
            kwargs["cmap_name"] = 'Greens'  # 'Blues'
        elif 0 <= pt_i < 13:
            kwargs["cmap_name"] = 'Blues'  # 'bone_r'
        if pt_i in annots.keys():
            kwargs["annots"] = annots[pt_i]
        stat_chart_stack(idx_bar, re_key, fgn + "_stack", **kwargs)
        elegant_print("\n\n", logger)


class PlotB_Measures(GraphSetup):
    def __init__(self):
        pass  # To verify the proposed measure

    def prepare_graph(self, res_data):
        # res_data.shape (5,12,1|3,8,63)= (#iter, #indv, #sen_att, 1+1+#pru, 63)
        nb_iter, nb_indv, nb_attr, nb_epru, nb_eval = np.shape(res_data)
        new_data = np.zeros([  # .transpose(2,1,3,4,0)
            nb_attr, nb_indv, nb_epru, nb_eval - 3, nb_iter])  # (1|3,12,8,63,5)
        for i in range(nb_attr):
            for j in range(nb_indv):
                for k in range(nb_epru):
                    for m in range(nb_eval - 3):
                        for n in range(nb_iter):
                            new_data[i, j, k, m, n] = res_data[
                                n][j][i][k][m]
        del nb_iter, nb_indv, nb_attr, nb_epru, nb_eval
        return new_data  # .shape (1|3,12,8,60,5)

    def schedule_mspaint(self, res_data, res_all, figname=""):
        # res_data.shape (#iter=5, #ens=12, #att=1/3, 1+1+#pru=8, 63)
        # new_data.shape (#att=1/3, #bag=12, 1+1+#pru=8, #iter=5, 60)
        new_data = self.prepare_graph(res_data)[:, :, :, 30:, :]

        data_name, binary, nb_cls, nb_pru, nb_iter, _ = res_all[0]
        sensitive_attributes = res_all[1]
        name_ens_set, abbr_cls_set = res_all[3], res_all[4]
        rank_pru_set = res_all[-1]  # name_pru based on ranking
        if len(sensitive_attributes) > 1:
            sensitive_attributes.append('joint')
        # idx_1 = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # list(range(12))
        idx_1 = list(range(11))  # after removing `LR' out
        name_ens_set = [name_ens_set[i] for i in idx_1]
        abbr_cls_set = [abbr_cls_set[i] for i in idx_1]
        idx_2 = [3, 4, 5, 7]  # [0, 3, 4, 5, 7]  # list(range(8))
        rank_pru_set = [rank_pru_set[i] for i in idx_2]
        rank_pru_set = [i.replace(
            'rank.', 'Rank by '  # 'Rank.', 'Rank via '
        ) if 'rank.' in i else i for i in rank_pru_set]

        for sa, sens_attr in enumerate(sensitive_attributes):
            fgn = "{}_{}_".format(figname, sens_attr)
            curr = new_data[sa][idx_1]  # shape: (#bag=11, 8, 30, 5)

            curr = curr[:, idx_2, :, :]  # shape (11,5,30,5)= (#bag#ind,#pru,30,#iter)
            self.plot_multiple_hist_chart(
                curr, abbr_cls_set, rank_pru_set, "acc",
                fgn, seperated=False)
            self.plot_multiple_hist_chart(
                curr, abbr_cls_set, rank_pru_set, "fair",
                fgn, seperated=False)
            del curr, fgn

    def plot_multiple_hist_chart(self, curr, ens, pru, mark="acc",
                                 fgn="", seperated=True):
        if mark == "acc":
            idx = [0, 1, 2, 3]
            key = ["Accuracy", "Precision", "Recall", "f1_score"]
        elif mark == "sen":
            idx = [6, 7, 8, 10, 11]
            key = ["Sensitivity", "Specificity", "G_mean",
                   "Matthew", "Cohen"]
        elif mark == "fair":
            idx = [24, 25, 26, 28]
            key = ["DP", "EO", "PQP", "DR"]
        mode = "ascend" if mark == "fair" else "descend"

        # new_curr = curr[:, :, idx]  # shape (8, 60, ?)
        # new_curr = curr[:, :, :, idx]# shape (#bag=12,#pru=8,5,#eval)
        new_curr = curr[:, :, idx, :]  # shape (11,5,4|5|4,5)= (#indv,#pru,#eval,#iter)

        alt_curr = np.concatenate(
            [new_curr[i] for i in range(11)], axis=2)
        # new_curr.shape (11,5,4|5|4,5)= (#indv,#pru,#eval,#iter)
        # alt_curr.shape   (5,4|5|4,55)= (#pru,#eval,#iter*#indv)
        # Ys_avg/std.shape ()= (#eval,#indv*#pru)
        Ys_avg = alt_curr.mean(axis=2).T  # shape (?,8)
        Ys_std = alt_curr.std(axis=2, ddof=1).T
        multiple_hist_chart(Ys_avg, Ys_std, key, '', pru,
                            figname=fgn + mark, rotate=20)
        if not seperated:
            return

        for k, val in enumerate(ens):
            values = new_curr[k]  # .transpose(0, 2, 1) shape(8,?,5)
            Ys_avg = values.mean(axis=2).T
            # Ys_std = values.std(axis=2, ddof=1).T
            Ys_std = values.std(axis=2).T
            multiple_hist_chart(
                Ys_avg, Ys_std, key, '', pru,
                figname=fgn + mark + "_bag_" + str(val),
                rotate=20)
        return


"""
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

        data_name, binary, nb_cls, _, nb_iter, _ = res_all[0]
        sensitive_attributes = res_all[1]
        ensemble_methods = res_all[-1]
        ensemble_methods = [
            i.replace('FPR', 'fpr').replace('FNR', 'fnr') 
            if 'FairGBM' in i else i for i in ensemble_methods]
        idx = [0, 1, 2, 3, 4, 6]
        ensemble_methods = [ensemble_methods[i] for i in idx]

        for sa, sens_attr in enumerate(sensitive_attributes):
            fgn = "{}_{}_".format(figname, sens_attr)
            curr = new_data[sa][idx]
            del fgn, curr

        num_s, num_e, num_v, _ = new_data.shape
        alt_data = np.concatenate([
            new_data[i] for i in range(num_s)], axis=2)
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
        key = [r"$\Delta$({})".format(i) for i in key
               ] + ["DP", "EO", "PQP", "DR"]
        fgn = figname + "_confusion"
        # analogous_confusion(Mat, key, fgn, normalize=False)

        idx_A = [0, 1, 2, 3, 5]
        Mat_A, Mat_B = Mat[idx_A], Mat[6:]
        key_A, key_B = [key[i] for i in idx_A], key[6:]
        analogous_confusion_extended(Mat_B, Mat_A, key_B, key_A,
                                     figname + '_confusion_alt',
                                     cmap_name='PuBu', rotate=0,
                                     figsize='M-WS')
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
            key = ["Sensitivity", "Specificity", "G_mean", "Matthew", "Cohen"]
        elif mark == "fair":
            idx = [50, 51, 52, 54]
            key = ["DP", "EO", "PQP", "DR"]  # or "FQ (ours)"

        if mark.endswith("norm"):
            pass
        elif mark.endswith("advr"):
            idx = [i + 13 for i in idx]
        elif mark.endswith("abs_"):
            idx = [i + 26 for i in idx]
        mode = "ascend" if mark == "fair" else "descend"

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
        num_s, num_e, num_v, _ = alt_data.shape
        alt_data = np.concatenate([
            alt_data[i] for i in range(num_s)], axis=2)
        alt_data = np.concatenate([
            alt_data[i] for i in range(num_e)], axis=1)

        X, Ys = alt_data[26], alt_data[[50, 51, 52, 54], :]
        # annots = (r"$\Delta$(Accuracy)", "Fairness Measure")
        annots = (r"$\Delta$(Accuracy)", "Fairness measure")
        annotZs = ('DP', 'EO', 'PQP', 'DR')
        fgn = figname + "_correlation"
        multiple_scatter_chart(X, Ys, annots, annotZs, fgn,
                               ind_hv='v', identity=False)

        Mat = alt_data[[26, 27, 28, 29, 32, 33, 50, 51, 52, 54]]
        key = ["Acc", "P", "R", "f1", "Sen", "Spe"]
        key = [r"$\Delta$(%s)" % i for i in key
               ] + ["DP", "EO", "PQP", "DR"]
        fgn = figname + "_confusion"
        # analogous_confusion(Mat, key, fgn, normalize=False)

        Mat_A, Mat_B = Mat[: 6], Mat[6:]
        key_A, key_B = key[: 6], key[6:]
        Mat_C = Mat_A[[0, 1, 2, 3, 5]]
        key_C = [key_A[i] for i in [0, 1, 2, 3, 5]]  # delta(Sen)
        analogous_confusion_extended(
            Mat_B, Mat_C, key_B, key_C, figname + '_confusion_alt',
            cmap_name='PuBu', rotate=0, figsize='M-WS')
        return


class PlotE_Measures(GraphSetup):
    def prepare_graph(self, res_data):
        new_data = np.zeros_like(res_data).transpose(1, 2, 3, 0)
        nb_iter, nb_attr, nb_ens, nb_eval = np.shape(res_data)
        for i in range(nb_attr):
            for j in range(nb_ens):
                for k in range(nb_eval):
                    new_data[i, j, k] = res_data[:, i, j, k]
        del nb_iter, nb_attr, nb_ens, nb_eval
        return new_data


class PlotF_Prunings(GraphSetup):
    def __init__(self):
        pass


# class Renew_PlotF_Prunings(PlotF_Prunings):
#     def renew_schedule_msgraph(self):
#         return
"""


class Renew_GatherF_Prunings(GatherF_Prunings):
    def renew_schedule_msgraph(self, res_data, res_all, optional_data,
                               figname='', jt=False, logger=None,
                               verbose=False):
        idx = list(range(56, 112)) + list(range(168, 224)) + list(
            range(280, 336)) + list(range(336, 339))
        new_data = {i: self.prepare_graph(np.array(
            res_data[i]))[:, idx, :] for i in optional_data}
        elegant_print("renew mCV_expt6\n", logger)

        _, Ys_model, Ys_acc = self.renew_retrieve_dat(
            new_data, res_all, optional_data, 0)
        _, _, Ys_f1s = self.renew_retrieve_dat(
            new_data, res_all, optional_data, 3)
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
            verbose)
        return

    def renew_retrieve_dat(self, new_data, res_all, optional_data,
                           pt_i, fgn='', ddof=0, logger=None):
        Ys_avg, Ys_std = np.zeros((9, 7)), np.zeros((9, 7))
        Ys_i, reorder = 0, [4, 5, 6, 0, 1, 2, 3]
        rez = 2 if pt_i in [0, 13, 26] else 4
        mode = 'ascend' if pt_i >= 39 else 'descend'
        Ys_entire = np.zeros((9, 7, 5))
        for i in optional_data:
            ret_key, ret_r, ret_c, sen_att = self.merge_sub_data(
                res_all[i], logger=logger)
            for si, sk in enumerate(sen_att):
                alt_data = new_data[i][:, ret_c[sk]]
                alt_data = alt_data[ret_r[sk], :][:, pt_i]
                alt_data = alt_data.astype(DTY_FLT)[reorder, :]
                Ys_avg[Ys_i] = alt_data.mean(axis=1)
                Ys_std[Ys_i] = alt_data.std(axis=1, ddof=ddof)
                Ys_entire[Ys_i] = alt_data
                Ys_i += 1
        Ys_entire_trans = Ys_entire.transpose(1, 0, 2).reshape(
            -1, 5 * 9)  # ->(7,9,5) ->(7,45)
        Ys_models = ret_key[sen_att[0]]
        Ys_models = [Ys_models[i] for i in reorder]
        return Ys_entire, Ys_models, Ys_entire_trans

    def renew_graph_scatter(self, X, Ys, Ys_model, Ys_annot,
                            verbose):
        label_x = 'Performance (Accuracy)'  # f1_score
        kw_balance = {'alpha_loc': 'b4', 'alpha_rev': True}
        n = len(Ys_model)
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
        return


# -------------------------------------
# To verify


# =====================================
# Benchmarks
# =====================================
