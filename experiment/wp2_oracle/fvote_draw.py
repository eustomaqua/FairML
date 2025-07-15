# coding: utf-8


from fairml.facilc.draw_graph import (
    multiple_line_chart, multiple_hist_chart,
    Friedman_chart, stat_chart_stack, twinx_hist_chart,
    twinx_bars_chart, histogram_chart, scatter_parl_chart,
    scatter_and_corr, sns_scatter_corr, scatter_id_chart,
    _setup_config, _setup_figsize, _setup_figshow)
from fairml.facilc.draw_hypos import (
    Friedman_init, _avg_and_stdev, _encode_sign,
    comp_t_init, comp_t_prep, comp_t_sing,
    cmp_paired_avg, cmp_paired_wtl)

from experiment.wp2_oracle.fetch_data import (
    GraphSetup, CURR_EXPT_DIR, pd_concat_divide_raw,
    pd_concat_divide_sht, pd_concat_sens_raw)
from fairml.widget.utils_const import (
    DTY_FLT, unique_column, _get_tmp_name_ens, _get_tmp_document)
from fairml.widget.utils_remark import (
    AVAILABLE_ABBR_ENSEM, AVAILABLE_ABBR_CLS)

import csv
import os
import numpy as np
import pandas as pd


# =====================================
# Properties
# =====================================


# ----------------------------------
# Section 3.1


class PlotC_TheoremsLemma(GraphSetup):
    def __init__(self, name_ens, nb_cls, nb_iter=5,
                 trial_type='mCV_expt3', figname='expt3_thm'):
        super().__init__(name_ens, nb_cls, figname=figname)
        self._trial_type = trial_type

    def prepare_graph(self):
        csv_row_1 = unique_column(8 + 26 + 7 * 4)
        return csv_row_1[8:]

    def schedule_mspaint(self, raw_dframe, partition=False):
        # nb_set, id_set, index = self.recap_sub_data(raw_dframe)
        _, _, index = self.recap_sub_data(raw_dframe)
        tag_col = self.prepare_graph()
        index = np.concatenate(index, axis=0)
        nvt = [v.loc[index][tag_col] for v in raw_dframe.values()]
        dframe = pd.concat(nvt)
        del nvt

        tmp = _get_tmp_name_ens(self._name_ens)  # 'all'
        self.verify_theorem31(dframe, tmp)
        self.verify_theorem33(dframe, tmp)
        self.verify_lemma32(dframe, tmp)
        self.verify_theorem34(dframe, tmp)

        if not partition:
            return

        for k, v in raw_dframe.items():
            # nvt = v.iloc[index][tag_col]
            nvt = v.loc[index][tag_col]

            self.verify_theorem31(nvt, k)  # keyword
            self.verify_theorem33(nvt, k)
            self.verify_lemma32(nvt, k)
            self.verify_theorem34(nvt, k)

    def verify_theorem31(self, df, kw):
        tag_trn = ['AI', 'AJ', 'AK']
        tag_tst = ['AW', 'AX', 'AY']
        annots = (
            r"$2\mathbb{E}_\rho[\mathcal{L}_{bias}(f)]$",
            r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$"
        )
        figname = self._figname + "31_" + kw
        self.plot_scatter_chart(
            df, 'AK', 'AW', figname + '', annots=annots)

    def verify_theorem33(self, df, kw):
        tag_trn = ['AI', 'AL', 'AM']
        tag_tst = ['AW', 'AZ', 'BA']
        annots = (
            r"$4\mathbb{E}_{\rho^2}[\mathcal{L}_{bias}(f,f^\prime)]$",
            r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$"
        )
        figname = self._figname + "33_" + kw
        self.plot_scatter_chart(
            df, 'AM', 'AW', figname + '', annots=annots)

    def verify_lemma32(self, df, kw):
        tag_trn = ['AN', 'AL']
        tag_tst = ['BB', 'AZ']
        annots = (
            r"$\mathbb{E}_{\rho^2}[\mathcal{L}_{bias}(f,f^\prime)]$",
            r"$\mathbb{E}_\mathcal{D}[\mathbb{E}_\rho[\ell_{bias}(f,\mathbf{x})]^2]$"
        )
        figname = self._figname.replace("thm", "lem") + '32_' + kw
        # LHS, RHS: Left/Right Hand Side
        self.plot_scatter_chart(
            df, 'AL', 'BB', figname + '_bak', annots=annots)

    def verify_theorem34(self, df, kw):
        tag_trn = ['AI', 'AJ', 'AL', 'AO']  # 'Y','X'
        tag_tst = ['AW', 'AX', 'AZ', 'BC']  # 'Y','X'
        annots = (
            r"$\mathrm{RHS}$ in Theorem 3.4",  # $\mathbf{RHS}$
            r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$"
        )
        figname = self._figname + "34_" + kw
        self.plot_scatter_chart(
            df, 'AO', 'AW', figname + '', annots=annots)  # _bak

    def plot_scatter_chart(self, df, tag_X, tag_Y,
                           figname, annots=('X', 'Y')):
        X = df[tag_X].values.astype(DTY_FLT)
        Y = df[tag_Y].values.astype(DTY_FLT)
        kwargs = {"annots": annots, "identity": True,
                  "figsize": "L-WS"}
        if "lem32" in figname:
            kwargs["diff"] = 0.05  # -0.05
            kwargs["base"] = 0.05
        kwargs["locate"] = "upper left"
        scatter_id_chart(X, Y, 'man_' + figname, **kwargs)


class PlotC_Revised_TheoremsLemma(GraphSetup):
    def __init__(self, name_ens, nb_cls, nb_iter=5,
                 trial_type='mCV_expt3', figname='expt3_thm'):
        super().__init__(name_ens, nb_cls, nb_iter,
                         trial_type, figname)

    def schedule_mspaint(self, raw_dframe, partition=False):
        nb_set, id_set, index = self.recap_sub_data(raw_dframe)
        tag_col = self.prepare_graph()
        ind = np.concatenate(index, axis=0)

        nvt = [v.loc[ind][tag_col] for v in raw_dframe.values()]
        dframe = pd.concat(nvt)  # TODO
        del nvt

    def verify_bounds(self, df, kw):
        import matplotlib.pyplot as plt
        tag_trn = ['AI', 'AK', 'AM', 'AO']
        tag_tst = ['AW', 'AY', 'BA', 'BC']

        annots = (
            r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$",
            r"$2\mathbb{E}_\rho[\mathcal{L}_{bias}(f)]$",
            r"$4\mathbb{E}_{\rho^2}[\mathcal{L}_{bias}(f,f^\prime)]$",
            r"$\mathrm{RHS}$ in Theorem 3.4"
        )

        X = df['AW'].values.astype('float')
        Ys = df[['AK', 'AM', 'AO']].values.astype('float')
        ind = np.arange(len(X))
        figname = self._figname + "_" + kw

        fig = plt.figure(figsize=_setup_config['L-NT'])
        plt.plot(ind, X, 'r.', label=annots[0])
        plt.plot(ind, Ys[:, 0], '.', label=annots[1])
        plt.plot(ind, Ys[:, 1], '.', label=annots[2])
        plt.plot(ind, Ys[:, 2], '.', label=annots[3])
        plt.legend(loc='best')
        _setup_figsize(fig, figsize=_setup_config['M-WS'])
        _setup_figshow(fig, figname=figname)
        return


# -------------------------------------
# Section 3.2, not Section 3.5


class PlotK_PACGeneralisation(GraphSetup):
    def __init__(self, name_ens, nb_cls, nb_iter=5,
                 trial_type='KFS_expt11', figname='expt11_thm'):
        super().__init__(name_ens, nb_cls, figname=figname)
        self._trial_type = trial_type

    def prepare_graph(self):
        length = self._nb_cls * 6 + (7 * 2 + 1)
        csv_row_1 = unique_column(8 + 2 + length)
        return csv_row_1[8:]

    def generate_column(self, tag_col):
        idv_XY = [[
            i * 6 + j for j in [1, 2]] for i in range(self._nb_cls)]
        idv_XY = [[j + 1 + (7 * 2 + 2) for j in i] for i in idv_XY]
        return [[tag_col[j] for j in i] for i in idv_XY]

    def schedule_mspaint(self, raw_dframe, partition=False):
        nb_set, id_set, index = self.recap_sub_data(raw_dframe)
        tag_col = self.prepare_graph()
        index = np.concatenate(index, axis=0)

        nvt = [v.loc[index][tag_col] for v in raw_dframe.values()]
        dframe = pd.concat(nvt)
        del nvt

        tmp = _get_tmp_name_ens(self._name_ens)
        self.verify_theorem36(dframe, tmp)
        self.verify_theorem35(dframe, tmp)

        if not partition:
            return

        for k, v in raw_dframe.items():
            nvt = v.loc[index][tag_col]
            self.verify_theorem36(nvt, k)
            self.verify_theorem35(nvt, k)

    # def verify_lam(self):
    #   pass

    def verify_theorem36(self, df, kw, verbose=False):
        lim_XY = ['M', 'N', 'P', 'Q']
        ens_XY = ['T', 'U', 'W', 'X']
        annots = (
            r"$\mathrm{RHS}$ in Theorem 3.6",
            r"$\mathcal{L}_{bias}(\mathbf{wv}_\rho)$",
        )
        figname = self._figname + "36_" + kw

        # self.plot_scatter_chart(
        #     df, 'N', 'M', figname + '_fair_lim', annots=annots)
        self.plot_scatter_chart(
            df, 'U', 'T', figname + '_fair_ens', annots=annots)

        if not verbose:
            return
        self.plot_scatter_chart(
            df, 'N', 'M', figname + '_fair_lim', annots=annots)

        self.plot_scatter_chart(
            df, 'Q', 'P', figname + '_loss_lim', annots=annots)
        self.plot_scatter_chart(
            df, 'X', 'W', figname + '_loss_ens', annots=annots)

    def verify_theorem35(self, df, kw):
        tag_col = self.prepare_graph()
        idv_XY = self.generate_column(tag_col)
        del tag_col

        annots = (
            r"$\mathrm{RHS}$ in Theorem 3.5",
            # r"$\mathcal{L}_{fair}(f)$",
            r"$\mathcal{L}_{bias}(f)$",
        )
        figname = self._figname + "35_" + kw

        ndf = [df[i] for i in idv_XY]
        key = [{i[j]: idv_XY[
            0][j] for j in [0, 1]} for i in idv_XY[1:]]
        ndf = [ndf[0]] + [
            i.rename(columns=j) for i, j in zip(ndf[1:], key)]
        ndf = pd.concat(ndf)

        self.plot_scatter_chart(
            ndf, 'AB', 'AA', figname + '_fair_idv', annots=annots)

    def plot_scatter_chart(self, df, tag_X, tag_Y,
                           figname, annots=('X', 'Y')):
        X = df[tag_X].values.astype(DTY_FLT)
        Y = df[tag_Y].values.astype(DTY_FLT)
        kwargs = {
            "annots": annots, "identity": True,
            "locate": "upper left"
        }
        scatter_id_chart(X, Y, 'man_' + figname, **kwargs)


# ----------------------------------
# Section 3.3


class PlotH_ImprovePruning(GraphSetup):
    def __init__(self, name_ens, nb_cls, nb_pru, nb_iter=5,
                 trial_type='mCV_expt8', figname='expt8',
                 logger=None):
        super().__init__(name_ens, nb_cls, nb_pru,
                         nb_iter=nb_iter, figname=figname)
        self._trial_type = trial_type
        self._logger = logger

        self._set_pru_name = [
            'ES', 'KL', 'KPk', 'KPz', 'RE', 'CM', 'OO',
            'GMA', 'LCS', 'DREP', 'SEP', 'OEP', 'PEP',
        ]
        self._set_pru_late = [
            'MRMC-MRMR', 'MRMC-MRMC', 'MRMREP',
            'Disc-EP', 'TSP-AP', 'TSP-DP',
        ]
        self._set_pru_prop = ['EPAF-C', 'EPAF-D', 'POEPAF', 'POPEP']

        self._length = sum(map(len, [
            self._set_pru_name, self._set_pru_late]))
        self._sens_dataset = {
            0: ["race"],
            1: ["sex", "age", "joint"],
            2: ["race", "sex", "joint"],
            3: ["sex", "race", "joint"],
            4: ["sex", "race", "joint"],
        }  # ricci, german, adult, ppr, ppvr

    @property
    def length(self):
        return self._length

    def prepare_graph(self):
        num_cmp = 1 + 5 + self._length  # Ensem,Propose,Compare
        csv_row_1 = unique_column(8 + 26 + 24 * num_cmp - 1)
        return csv_row_1[(8 + 26 - 2):]

    def pick_up_idx(self, rel_id=None):
        if rel_id is None:
            return []
        num_len, num_gap = self._length + 5 + 1, 10 * 2 + 4
        # rel_id:
        # -2,-1: ut,us
        #   0-4: La(MV), Lf(MV), 'Lo(MV)', E[La(f)], E[Lf(f,fp)]
        #   5-9: 'E[Lf(f)]', Acc, P, R, F1

        if rel_id == -2 or rel_id == -1:
            ind_trn = [i * num_gap + 2 * rel_id for i in range(num_len)]
            ind_tst = []
            ind_trn = [3 + 2 + i for i in ind_trn]

            if rel_id == -1:
                ind_trn[0] = ind_trn[0] + 1
            elif rel_id == -2:
                ind_trn[0] = ind_trn[0] - 1

        else:
            ind_trn = [i * num_gap + rel_id for i in range(num_len)]
            ind_tst = [i * num_gap + rel_id + 10 for i in range(num_len)]
            ind_trn = [3 + 2 + i for i in ind_trn]
            ind_tst = [3 + 2 + i for i in ind_tst]

        return ind_trn, ind_tst

    def pick_up_tag(self, csv_row_1, ind):
        return [csv_row_1[i] for i in ind]

    def pick_up_pru(self, tag_trn, tag_tst,
                    category=None, ind_ens=False):
        set_pru = self._set_pru_name + self._set_pru_late
        ind_pru = [1, 2, 4, 5, 6, 9, 10, 11, 12,
                   13, 14, 16, 17, 18]
        if category == 'ranking':
            ind_pru = [0, 1, 2, 4, 5, 6, 9, 11, 17, 18]
        elif category == 'optimis':
            ind_pru = [7, 8, 10, 12, 13, 14, 16]
        elif category == 'fairnes':
            ind_pru = []
            ind_pru = [7, 8, 12]
        elif category == 'spreads':
            ind_pru = [2, 6, 9, 10, 11, 12, 14, 16, 17]

        name_pru = [set_pru[i] for i in ind_pru]
        for k in ['KPk', 'KPz']:
            if k in name_pru:
                j = name_pru.index(k)
                name_pru[j] = 'KP'
        for k in ['MRMC-MRMR', 'MRMC-MRMC']:
            if k in name_pru:
                j = name_pru.index(k)
                name_pru[j] = k[-4:]
        for k in ['Disc-EP', 'TSP-AP', 'TSP-DP']:
            if k in name_pru:
                j = name_pru.index(k)
                name_pru[j] = k.replace('-', '.')

        # in fact, they are 'EPAF-D:2', 'POPEP'
        # name_pru.extend(['EPAF-C', 'EPAF-D', 'POEPAF'])
        name_pru.extend(['EPAF-C', 'EPAF-D', 'POAF'])  # POPEP
        ind_pru = [i + 6 for i in ind_pru]
        ind_pru = ind_pru + [1, 2, 5]
        if ind_ens:
            ind_pru = [0] + ind_pru
            name_pru = ['Ensem'] + name_pru

        tag_trn = [tag_trn[i] for i in ind_pru]
        tag_tst = [tag_tst[i] for i in ind_pru] if tag_tst else []
        return ind_pru, name_pru, tag_trn, tag_tst

    def pick_up_comparison(self, rel_id=None):
        csv_row_1 = self.prepare_graph()
        ind_trn, ind_tst = self.pick_up_idx(rel_id)
        tag_trn = self.pick_up_tag(csv_row_1, ind_trn)
        tag_tst = self.pick_up_tag(csv_row_1, ind_tst)
        return tag_trn, tag_tst

    def recap_sub_data(self, dframe, nb_row=4):
        nb_set = [len(v) for k, v in dframe.items()]
        assert len(set(nb_set)) == 1, "{}-{}".format(
            self._name_ens, self._nb_cls)
        nb_set = list(set(nb_set))[0]

        num_attr = 4 * self._nb_iter + 1
        nb_set = (nb_set - nb_row + 1) // num_attr

        id_set = [i * num_attr + nb_row - 1 for i in range(nb_set)]
        index = [[j * 4 + 1 + i for j in range(
            self._nb_iter)] for i in id_set]
        return nb_set, id_set, index

    def schedule_mspaint(self, raw_dframe, partition=False,
                         verbose=False):
        nb_set, id_set, index = self.recap_sub_data(raw_dframe)
        tag_col = self.prepare_graph()
        tag_col.remove('AI')
        for k in [
            'BI', 'CG', 'DE', 'EC', 'FA',
            'FY', 'GW', 'HU', 'IS', 'JQ', 'KO', 'LM', 'MK', 'NI', 'OG',
                'PE', 'QC', 'RA', 'RY', 'SW', 'TU', 'US', 'VQ', 'WO']:
            tag_col.remove(k)

        # CROPPED
        self.plot_for_sec33_prus(
            raw_dframe, partition, tag_col, nb_set, index)
        self.plot_for_sec33_fair(
            raw_dframe, partition, tag_col, nb_set, index,)
        self.plot_for_sec34_fair(
            raw_dframe, partition, tag_col, nb_set, index,
            verbose)
        # CROPPED
        if not verbose:
            return
        self.plot_for_sec35_prus(
            raw_dframe, partition, tag_col, nb_set, index)

    def plot_for_sec33_prus(self, raw_dframe, partition,
                            tag_col, nb_set, index):
        # aka. def plot_for_sec33()
        tmp = _get_tmp_name_ens(self._name_ens)
        fn = '_'.join([self._figname, tmp])
        new_avg, _, new_var, new_raw = pd_concat_divide_raw(
            raw_dframe, tag_col, nb_set, index)  # _:new_std
        self.verify_aggregated_rank(new_avg, fn + "_whole_avg")

        if not partition:
            return
        for k, v in raw_dframe.items():
            avg, _, _, raw = pd_concat_divide_sht(
                v, tag_col, nb_set, index)
            self.verify_aggregated_rank(raw, fn + "_split_raw_" + k)

    def verify_Friedman_chart(self, df, kw, rel_id,
                              category=None, ind_ens=False):
        tag_trn, tag_tst = self.pick_up_comparison(rel_id)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category=category, ind_ens=ind_ens)
        tag_col = tag_trn if rel_id < 0 else tag_tst

        U = df[tag_col].values.astype(DTY_FLT)
        if rel_id == -2 and ind_ens:
            U[:, 0] = U[:, 0] / 60
        mode = 'ascend'
        if rel_id in [6, 7, 8, 9]:
            mode = 'descend'

        # rank, idx_bar = Friedman_init(U, mode=mode)
        _, idx_bar = Friedman_init(U, mode=mode)
        figname = '_'.join([kw, 'rel{}'.format(rel_id)])
        if rel_id not in [4, -1, -2, 9, 8, ] + [2, ]:
            Friedman_chart(idx_bar, name_pru, figname + '_fried5',
                           alpha=0.05, logger=self._logger,
                           anotCD=True)

        annots = {
            6: r"aggr.rank.accuracy",
            7: r"aggr.rank.precision",
            8: r"aggr.rank.recall",
            9: r"aggr.rank.f1_score",

            0: r"aggr.rank.$G_1(\mathbf{wv}_\rho)$",
            1: r"aggr.rank.$G_2(\mathbf{wv}_\rho)$",
            2: r"aggr.rank.$\mathcal{L}(\mathbf{wv}_\rho)$",
            # 0: r"aggr.rank.$\mathcal{L}_{acc}(\mathbf{wv}_\rho)$",
            # 1: r"aggr.rank.$\mathcal{L}_{fair}(\mathbf{wv}_\rho)$",

            -2: r"Aggregated Time Cost (min)",
            -1: r"Aggregated Space Cost",
        }
        kwargs = {"cmap_name": 'GnBu', "rotation": 80}  # 85,55
        if rel_id in annots.keys():
            kwargs["annots"] = annots[rel_id]
        stat_chart_stack(idx_bar, name_pru, figname + '_stack',
                         **kwargs)

    def verify_aggregated_rank(self, df, kw):
        '''
        self.verify_Friedman_chart(df, kw, rel_id=6)
        self.verify_Friedman_chart(df, kw, rel_id=7)
        self.verify_Friedman_chart(df, kw, rel_id=8)
        self.verify_Friedman_chart(df, kw, rel_id=9)

        self.verify_Friedman_chart(df, kw, rel_id=0)
        self.verify_Friedman_chart(df, kw, rel_id=1)
        self.verify_Friedman_chart(df, kw, rel_id=2)

        self.verify_Friedman_chart(df, kw, rel_id=-2)
        self.verify_Friedman_chart(df, kw, rel_id=-1)
        '''

        # if not verbose:
        #     return
        self.verify_Friedman_chart(df, kw, rel_id=6)
        self.verify_Friedman_chart(df, kw, rel_id=7)
        self.verify_Friedman_chart(df, kw, rel_id=1)
        self.verify_Friedman_chart(df, kw, rel_id=8)
        self.verify_Friedman_chart(df, kw, rel_id=2)

    def plot_for_sec33_fair(self, raw_dframe, partition,
                            tag_col, nb_set, index):
        A1_data, A2_data, Jt_data = pd_concat_sens_raw(
            raw_dframe, tag_col, nb_set, index)
        new_avg = pd.concat([A1_data[0], A2_data[0], Jt_data[0]])
        new_raw = pd.concat([A1_data[3], A2_data[3], Jt_data[3]])
        tmp = _get_tmp_name_ens(self._name_ens)
        fn = '_'.join([self._figname, tmp])
        self.compare_aggregated_fair(new_avg, fn + "_whole_avg")
        if not partition:
            return

    def compare_fair_measure(self, df, kw, rel_id,
                             category=None, ind_ens=True):
        tag_trn, tag_tst = self.pick_up_comparison(rel_id * 2)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category=category, ind_ens=ind_ens)

        # rel_id:
        #   0,2,4,6,8: `g1`
        #   1,3,5,7,9: `g0`
        #         unaware,GF-one,GF-two,GF-thr,manual-accuracy

        U_g1 = df[tag_tst].values.astype(DTY_FLT)
        mode = 'descend'

        tag_trn, tag_tst = self.pick_up_comparison(rel_id * 2 + 1)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category=category, ind_ens=ind_ens)
        U_g0 = df[tag_tst].values.astype(DTY_FLT)

        # U = U_g1 - U_g0
        U = np.abs(U_g1 - U_g0)
        mode = 'ascend'
        rank, idx_var = Friedman_init(U, mode=mode)
        figname = '_'.join([kw, 'fair{}'.format(rel_id)])
        Friedman_chart(idx_var, name_pru, figname + '_fried5',
                       alpha=0.05, logger=self._logger,
                       anotCD=True)

        annots = {
            0: r"aggr.rank.$diff()$",
            1: r"aggr.rank.$diff(DP_F)$",
            2: r"aggr.rank.$diff(EO_F)$",
            3: r"aggr.rank.$diff(PQP_F)$",
            4: r"aggr.rank.$diff($accuracy$)$",
        }
        annots = {
            0: r"aggr.rank.$diff()$",
            1: r"aggr.rank.$DP_F$",
            2: r"aggr.rank.$EO_F$",
            3: r"aggr.rank.$PQP_F$",
            # 4: r"aggr.rank.$\Delta($accuracy$)$",
            4: r"aggr.rank.$diff($accuracy$)$",
        }
        kwargs = {"cmap_name": 'GnBu', "rotation": 80}  # 75
        if rel_id in annots.keys():
            kwargs["annots"] = annots[rel_id]
        stat_chart_stack(idx_var, name_pru, figname + '_stack',
                         **kwargs)

    def compare_aggregated_fair(self, df, kw):
        self.compare_fair_measure(df, kw, rel_id=1)
        self.compare_fair_measure(df, kw, rel_id=2)
        self.compare_fair_measure(df, kw, rel_id=3)
        self.compare_fair_measure(df, kw, rel_id=4)

    def plot_for_sec34_fair(self, raw_dframe, partition,
                            tag_col, nb_set, index,
                            verbose):
        fn = _get_tmp_name_ens(self._name_ens)
        fn = '_'.join([self._figname, fn])
        for rel_set in range(nb_set):
            if not verbose and rel_set != 2:
                continue
            kw = '_'.join([fn, 'set{}'.format(rel_set)])
            ind = index[rel_set]
            self.compare_hist_gather_diff(
                raw_dframe, ind, kw, 0, rel_set)
            self.compare_hist_gather_diff(
                raw_dframe, ind, kw, 1, rel_set)
            if rel_set == 0:
                continue
            self.compare_hist_gather_diff(
                raw_dframe, ind, kw, 2, rel_set)
            self.compare_hist_gather_diff(
                raw_dframe, ind, kw, 3, rel_set)
        if not partition:
            return

        for k, v in raw_dframe.items():
            for rel_set in range(nb_set):
                if not verbose and rel_set != 2:
                    continue
                kw = '_'.join([fn, k, 'set' + str(rel_set)])
                ind = index[rel_set]

                self.compare_hist_group_diff(v, ind, kw, 1, rel_set)
                if rel_set == 0:
                    continue
                self.compare_hist_group_diff(v, ind, kw, 2, rel_set)
                self.compare_hist_group_diff(v, ind, kw, 3, rel_set)

    def compare_hist_group_ones(self, dframe, ind, kw,
                                rel_jt=1, rel_set=0):
        ind_sens = np.add(ind, rel_jt).tolist()

        tag_trn, tag_tst = self.pick_up_comparison(0)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category='fairnes', ind_ens=True)
        Ys_avg = np.zeros((len(name_pru), 4))
        Ys_std = np.zeros((len(name_pru), 4))

        for k in range(1, 5):
            tag_trn, tag_tst = self.pick_up_comparison(2 * k)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)
            tdf = dframe.loc[ind_sens][tag_tst]
            Ys_avg[:, (k - 1)] = tdf.mean(
                axis=0).values.astype(DTY_FLT)
            Ys_std[:, (k - 1)] = tdf.std(
                ddof=1).values.astype(DTY_FLT)

        annots = [
            r"$DP_F$",
            r"$EO_F$",
            r"$PQP_F$",
            r"accuracy"
        ]
        annotX = self._sens_dataset[rel_set][rel_jt - 1]
        fn = '_'.join([kw, 'sens' + str(rel_jt), 'g1'])
        multiple_hist_chart(Ys_avg.T, Ys_std.T, annots,
                            annotX, name_pru, fn, rotate=25)

    def compare_hist_group_zero(self, dframe, ind, kw,
                                rel_jt=1, rel_set=0):
        ind_sens = np.add(ind, rel_jt).tolist()

        tag_trn, tag_tst = self.pick_up_comparison(0)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category='fairnes', ind_ens=True)
        Ys_avg = np.zeros((len(name_pru), 4))
        Ys_std = np.zeros((len(name_pru), 4))

        for k in range(1, 5):
            tag_trn, tag_tst = self.pick_up_comparison(2 * k + 1)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)
            tdf = dframe.loc[ind_sens][tag_tst]
            Ys_avg[:, (k - 1)] = tdf.mean(axis=0).values.astype(DTY_FLT)
            Ys_std[:, (k - 1)] = tdf.std(ddof=1).values.astype(DTY_FLT)

        annots = [
            r"$DP_F$",
            r"$EO_F$",
            r"$PQP_F$",
            r"accuracy"
        ]
        annotX = self._sens_dataset[rel_set][rel_jt - 1]
        fn = '_'.join([kw, 'sens' + str(rel_jt), 'g0'])
        multiple_hist_chart(Ys_avg.T, Ys_std.T, annots,
                            annotX, name_pru, fn, rotate=25)

    def compare_hist_group_diff(self, dframe, ind, kw,
                                rel_jt=1, rel_set=0):
        ind_sens = np.add(ind, rel_jt).tolist()
        tag_trn, tag_tst = self.pick_up_comparison(0)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category='fairnes', ind_ens=True)
        Ys_avg = np.zeros((len(name_pru), 4 + 3))
        Ys_std = np.zeros((len(name_pru), 4 + 3))

        for k in range(1, 5):
            tag_trn, tag_tst = self.pick_up_comparison(2 * k)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)
            U_g1 = dframe.loc[ind_sens][tag_tst]
            tag_trn, tag_tst = self.pick_up_comparison(2 * k + 1)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)
            U_g0 = dframe.loc[ind_sens][tag_tst]
            tdf = U_g1.values.astype(DTY_FLT) - U_g0.values.astype(DTY_FLT)
            Ys_avg[:, (k - 1)] = tdf.mean(axis=0)
            Ys_std[:, (k - 1)] = tdf.std(axis=0, ddof=1)

        for k in range(3):
            tag_trn, tag_tst = self.pick_up_comparison(k)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)
            # tdf = dframe.iloc[ind][tag_tst]
            tdf = dframe.loc[ind](tagt)
            Ys_avg[:, (k + 4)] = tdf.mean(axis=0).values.astype(DTY_FLT)
            Ys_std[:, (k + 4)] = tdf.std(ddof=1).values.astype(DTY_FLT)

        annots = [
            r"$diff(DP_F)$",
            r"$diff(EO_F)$",
            r"$diff(PQP_F)$",
            r"$diff($accuracy$)$",
            r"$G_1(\mathbf{wv}_\rho)$",
            r"$G_2(\mathbf{wv}_\rho)$",
            r"$\mathcal{L}(\mathbf{wv}_\rho)$",
        ]
        # fn = self._figname + '_' + kw + '_diff'
        fn = '_'.join([kw, 'attr{}'.format(rel_jt), 'diff'])

        annotX = ('Fairness Measure', 'Objective Function')
        multiple_hist_chart(Ys_avg.T[:4], Ys_std.T[:4],
                            annots[:4], annotX[0], name_pru,
                            figname=fn + 'air', rotate=15)
        multiple_hist_chart(Ys_avg.T[4:], Ys_std.T[4:],
                            annots[4:], annotX[1], name_pru,
                            figname=fn + 'obj', rotate=15)

    def compare_hist_gather_diff(self, raw_dframe, ind, kw,
                                 rel_jt=1, rel_set=0):
        tag_trn, tag_tst = self.pick_up_comparison(0)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category='fairnes', ind_ens=True)
        fn = '_'.join([kw, 'attr{}'.format(rel_jt)])  # , 'diff'
        if rel_jt == 0:
            Ys_avg = np.zeros((len(name_pru), 3))
            Ys_std = np.zeros((len(name_pru), 3))

            for k in range(3):
                tag_trn, tag_tst = self.pick_up_comparison(k)
                _, _, tag_trn, tag_tst = self.pick_up_pru(
                    tag_trn, tag_tst, category='fairnes', ind_ens=True)
                tdf = [v.loc[
                    ind][tag_tst] for v in raw_dframe.values()]
                tdf = pd.concat(tdf).values.astype(DTY_FLT)
                Ys_avg[:, k] = tdf.mean(axis=0)
                Ys_std[:, k] = tdf.std(axis=0, ddof=1)

            annots = [
                r"$G_1(\mathbf{wv}_\rho)$",
                r"$G_2(\mathbf{wv}_\rho)$",
                r"$\mathcal{L}(\mathbf{wv}_\rho)$",
            ]
            multiple_hist_chart(Ys_avg.T, Ys_std.T, annots,
                                annotX='Objective Function',
                                annotYs=name_pru, figname=fn,
                                rotate=15)
            return

        if rel_set == 0:
            assert rel_jt == 1
        else:
            assert 1 <= rel_jt <= 3
        ind_sens = np.add(ind, rel_jt).tolist()
        Ys_avg = np.zeros((len(name_pru), 4))
        Ys_std = np.zeros((len(name_pru), 4))

        for k in range(1, 5):
            tag_trn, tag_tst = self.pick_up_comparison(2 * k)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)
            U_g1 = [v.loc[
                ind_sens][tag_tst] for v in raw_dframe.values()]
            U_g1 = pd.concat(U_g1).values.astype(DTY_FLT)

            tag_trn, tag_tst = self.pick_up_comparison(2 * k + 1)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)
            U_g0 = [v.loc[
                ind_sens][tag_tst] for v in raw_dframe.values()]
            U_g0 = pd.concat(U_g0).values.astype(DTY_FLT)

            tdf = U_g1 - U_g0
            tdf = np.abs(tdf)
            if k == 4:
                tdf *= 100.  # accuracy
            Ys_avg[:, (k - 1)] = tdf.mean(axis=0)
            Ys_std[:, (k - 1)] = tdf.std(axis=0, ddof=1)

        annots = [
            r"$diff(DP_F)$",
            r"$diff(EO_F)$",
            r"$diff(PQP_F)$",
            r"$diff($accuracy$)$",
        ]
        annots = [
            r"$DP_F$",
            r"$EO_F$",
            r"$PQP_F$",
            r"$diff($accuracy$)$",
        ]

        multiple_hist_chart(Ys_avg.T[:3], Ys_std.T[:3], annots[:3],
                            annotX='Fairness Measure',
                            annotYs=name_pru, figname=fn,
                            rotate=15)

    def plot_for_sec35_prus(self, raw_dframe, partition,
                            tag_col, nb_set, index):
        fn = _get_tmp_name_ens(self._name_ens)
        fn = '_'.join([self._figname, fn])
        ind = np.concatenate(index, axis=0)
        dframe = {k: v.loc[
            ind][tag_col] for k, v in raw_dframe.items()}

        tmp_dframe = pd.concat(list(dframe.values()))
        self.compare_Chronos_cost(tmp_dframe, fn + '_us', -1)
        self.compare_Chronos_cost(
            tmp_dframe, fn + '_ut', -2, ind_ens=True)
        self.compare_Efficiency(tmp_dframe, fn + '_parl')
        if not partition:
            return

        for k, v in raw_dframe.items():
            # tdf = v.iloc[ind][tag_col]
            tdf = v.loc[ind][tag_col]

    def compare_Chronos_cost(self, df, kw, rel_id,
                             category=None, ind_ens=False):
        tag_trn, tag_tst = self.pick_up_comparison(rel_id)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category=category, ind_ens=ind_ens)
        U = df[tag_trn].values.astype(DTY_FLT)

        if ind_ens:
            X, Ys = U[:, 0], U[:, 1:]
            name_pru = name_pru[1:]
        else:
            X, Ys = np.ones(U.shape[0]) * self._nb_cls, U
        if rel_id == -1:
            X = np.c_[X, np.ones(U.shape[0]) * self._nb_pru]
        elif rel_id == -2:  # and (not ind_ens):
            X = None

        annots = 'Space Cost' if rel_id == -1 else 'Time Cost (min)'
        histogram_chart(X, Ys, figname=kw, annotX=annots,
                        annotY=name_pru, ind_hv='h')

    def compare_Efficiency(self, df, kw):
        tag_trn = ['BF', 'CD', 'DB', 'DZ', 'EX']
        centralised = df[tag_trn[0]].values.astype(DTY_FLT)
        distributed = df[tag_trn[1: 3]].values.astype(DTY_FLT)
        num_expt = distributed.shape[0]

        speed_up = np.zeros((num_expt, 2))
        efficiency = np.zeros((num_expt, 2))
        for k in range(2):
            speed_up[:, k] = np.divide(distributed[:, k], centralised)
            # ↑ BUG?? TODO
            efficiency[:, k] = speed_up[:, k] / (k + 2)

        picked_keys = [r"$n_m = {}$".format(i) for i in [2, 3]]
        '''
        scatter_parl_chart(speed_up, picked_keys,
                       'Speedup', kw + '_sp')
        scatter_parl_chart(efficiency, picked_keys,
                       'Efficiency', kw + '_ey')
        '''

        scatter_parl_chart(speed_up[:, :1], picked_keys[:1],
                           'Speedup', kw + '_sp2')
        scatter_parl_chart(speed_up[:, 1:], picked_keys[1:],
                           'Speedup', kw + '_sp3')
        scatter_parl_chart(efficiency[:, :1], picked_keys[:1],
                           'Efficiency', kw + '_ey2')
        scatter_parl_chart(efficiency[:, 1:], picked_keys[1:],
                           'Efficiency', kw + '_ey3')

    def update_column(self, tag_col):
        # aka. def update_tag_col()
        tag_col.remove('AI')
        for k in ['BI', 'CG', 'DE', 'EC', 'FA',
                  'FY', 'GW', 'HU', 'IS', 'JQ',
                  'KO', 'LM', 'MK', 'NI', 'OG',
                  'PE', 'QC', 'RA', 'RY', 'SW',
                  'TU', 'US', 'VQ', 'WO']:
            tag_col.remove(k)
        return tag_col


# Section 3.3

class PlotHGather_ImprovePruning(PlotH_ImprovePruning):
    def __init__(self, nb_cls, nb_pru, nb_iter=5,
                 trial_type='mCV_expt8', figname='expt8',
                 logger=None):
        super().__init__(None, nb_cls, nb_pru, nb_iter,
                         trial_type, figname, logger)

    def schedule_mspaint(self, raw_dframe, tag_col):
        # nb_set, id_set, index = self.recap_sub_data(raw_dframe[0])
        # nm_set = ['ricci', 'german', 'adult', 'ppr', 'ppvr']
        nb_set, _, index = self.recap_sub_data(raw_dframe[0])
        '''
        self.plot_for_sec33_prus(raw_dframe, tag_col, nb_set, index)
        self.plot_for_sec33_fair(raw_dframe, tag_col, nb_set, index)
        self.plot_for_sec34_fair(raw_dframe, tag_col, nb_set, index)
        '''
        # CROPPED
        self.plot_for_sec35_prus(raw_dframe, tag_col, nb_set, index)

    def plot_for_sec33_prus(self, raw_dframe, tag_col, nb_set, index):
        new_avg, new_std, new_raw = [], [], []
        for rdf in raw_dframe:
            tmp_avg, tmp_std, _, tmp_raw = pd_concat_divide_raw(
                rdf, tag_col, nb_set, index)

            new_avg.append(tmp_avg)
            new_std.append(tmp_std)
            new_raw.append(tmp_raw)
        new_avg = pd.concat(new_avg)
        new_std = pd.concat(new_std)
        new_raw = pd.concat(new_raw)

        # fn = '_'.join([self._figname, "whole"])
        fn = self._figname
        self.verify_aggregated_rank(new_avg, fn + "_avg")
        # self.verify_aggregated_rank(new_raw, fn + "_raw")

    def plot_for_sec33_fair(self, raw_dframe, tag_col, nb_set, index):
        new_avg, new_std, new_raw = [], [], []
        for rdf in raw_dframe:
            A1_data, A2_data, Jt_data = pd_concat_sens_raw(
                rdf, tag_col, nb_set, index)
            tmp_avg = pd.concat([A1_data[0], A2_data[0], Jt_data[0]])
            tmp_std = pd.concat([A1_data[1], A2_data[1], Jt_data[1]])
            tmp_raw = pd.concat([A1_data[3], A2_data[3], Jt_data[3]])

            new_avg.append(tmp_avg)
            new_std.append(tmp_std)
            new_raw.append(tmp_raw)
        new_avg = pd.concat(new_avg)
        new_std = pd.concat(new_std)
        new_raw = pd.concat(new_raw)

        # fn = '_'.join([self._figname, "whole"])
        fn = self._figname
        self.compare_aggregated_fair(new_avg, fn + "_avg")
        # self.compare_aggregated_fair(new_raw, fn + "_raw")

    def plot_for_sec35_prus(self, raw_dframe, tag_col, nb_set, index):
        ind = np.concatenate(index, axis=0)
        dframe = []
        for rdf in raw_dframe:
            # tdf = {k: v.iloc[ind][tag_col] for k, v in rdf.items()}
            tdf = {k: v.loc[ind][tag_col] for k, v in rdf.items()}
            tdf = pd.concat(list(tdf.values()))
            dframe.append(tdf)
        tdf = pd.concat(dframe)

        self.compare_Chronos_cost(tdf, self._figname + "_ut", -2,
                                  ind_ens=True)
        # CROPPED
        self.compare_Chronos_cost(tdf, self._figname + "_us", -1)
        self.compare_Efficiency(tdf, self._figname + "_parl")

    def plot_for_sec34_fair(self, raw_dframe, tag_col, nb_set, index):
        for rel_set in range(nb_set):
            kw = '_'.join([self._figname, "set{}".format(rel_set)])
            ind = index[rel_set]
            self.compare_hist_gather_diff(raw_dframe, ind, kw, 0, rel_set)
            self.compare_hist_gather_diff(raw_dframe, ind, kw, 1, rel_set)
            if rel_set == 0:
                continue
            self.compare_hist_gather_diff(raw_dframe, ind, kw, 2, rel_set)
            self.compare_hist_gather_diff(raw_dframe, ind, kw, 3, rel_set)
        return

    def compare_hist_gather_diff(self, raw_dframe, ind, kw,
                                 rel_jt=1, rel_set=0):
        ind_sens = np.add(ind, rel_jt).tolist()
        tag_trn, tag_tst = self.pick_up_comparison(0)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category='fairnes', ind_ens=True)
        Ys_avg = np.zeros((len(name_pru), 4 + 3))
        Ys_std = np.zeros((len(name_pru), 4 + 3))

        for k in range(1, 5):
            tag_trn, tag_tst = self.pick_up_comparison(2 * k)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)

            sht_dframe = [{
                k: v.loc[ind_sens][tag_tst] for k, v in rdf.items()
            } for rdf in raw_dframe]
            sht_dframe = [
                pd.concat(list(rdf.values())) for rdf in sht_dframe]
            sht_dframe = pd.concat(sht_dframe)
            U_g1 = sht_dframe.values.astype(DTY_FLT)

            tag_trn, tag_tst = self.pick_up_comparison(2 * k + 1)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)

            sht_dframe = [{
                k: v.loc[ind_sens][tag_tst] for k, v in rdf.items()
            } for rdf in raw_dframe]
            sht_dframe = [
                pd.concat(list(rdf.values())) for rdf in sht_dframe]
            sht_dframe = pd.concat(sht_dframe)
            U_g0 = sht_dframe.values.astype(DTY_FLT)

            del sht_dframe
            tdf = U_g1 - U_g0
            Ys_avg[:, (k - 1)] = tdf.mean(axis=0)
            Ys_std[:, (k - 1)] = tdf.std(axis=0, ddof=1)

        for k in range(3):
            tag_trn, tag_tst = self.pick_up_comparison(k)
            _, _, tag_trn, tag_tst = self.pick_up_pru(
                tag_trn, tag_tst, category='fairnes', ind_ens=True)
            # tdf = dframe.iloc[ind][tag_tst]

            sht_dframe = [{
                k: v.loc[ind][tag_tst] for k, v in rdf.items()
            } for rdf in raw_dframe]
            sht_dframe = [
                pd.concat(list(rdf.values())) for rdf in sht_dframe]
            tdf = pd.concat(sht_dframe).values.astype(DTY_FLT)

            Ys_avg[:, (k + 4)] = tdf.mean(axis=0)
            Ys_std[:, (k + 4)] = tdf.std(axis=0, ddof=1)

        annots = [
            r"$diff(DP_F)$",
            r"$diff(EO_F)$",
            r"$diff(PQP_F)$",
            r"$diff($accuracy$)$",
            r"$G_1(\mathbf{wv}_\rho)$",
            r"$G_2(\mathbf{wv}_\rho)$",
            r"$\mathcal{L}(\mathbf{wv}_\rho)$",
        ]
        fn = "_".join([kw, "attr{}".format(rel_jt), "diff"])
        annotX = ('Fairness Measure', 'Objective Function')
        multiple_hist_chart(Ys_avg.T[:4], Ys_std.T[:4],
                            annots[:4], annotX[0], name_pru,
                            figname=fn + 'air', rotate=15)
        multiple_hist_chart(Ys_avg.T[4:], Ys_std.T[4:],
                            annots[4:], annotX[1], name_pru,
                            figname=fn + 'obj', rotate=15)
        return


class TableHGather_ImprovePruning(PlotH_ImprovePruning):
    def __init__(self, nb_cls, nb_pru, nb_iter=5,
                 trial_type='mCV_expt8', tabname='expt8',
                 logger=None):
        super().__init__(None, nb_cls, nb_pru, nb_iter,
                         trial_type, tabname, logger)

    def schedule_spreadsheet(self, raw_dframe, tag_col):
        nb_set, id_set, index = self.recap_sub_data(raw_dframe[0])
        nm_set = ['ricci', 'german', 'adult', 'ppr', 'ppvr']

        for i, name_ens in enumerate(AVAILABLE_ABBR_ENSEM):
            tdf = raw_dframe[i]['DT']
            kw = '_'.join([_get_tmp_name_ens(name_ens), 'DT'])
            self.streamlined_mediate(tdf, kw, nb_set, index)

    def streamlined_mediate(self, df, kw, nb_set, index,
                            category='spreads'):
        self.verify_paired_ttest_prus(
            df, kw, 6, category=category, nb_set=nb_set, index=index)
        self.verify_paired_ttest_prus(
            df, kw, 1, category=category, nb_set=nb_set, index=index)
        for rel_id in range(1, 4):
            nkw = "_".join([kw, "fair", "rel{}".format(rel_id)])

            tex_A1, wtl_A1, cmp_A1, name_pru, \
                idx_A1 = self.verify_paired_ttest_fair(
                    df, rel_id, 1, category=category, nb_set=nb_set, index=index)
            tex_A2, wtl_A2, cmp_A2, _, idx_A2 = self.verify_paired_ttest_fair(
                df, rel_id, 2, category=category, nb_set=nb_set, index=index)
            tex_Jt, wtl_Jt, cmp_Jt, _, idx_Jt = self.verify_paired_ttest_fair(
                df, rel_id, 3, category=category, nb_set=nb_set, index=index)

            ans_tex, ans_wtl, ans_cmp, ans_idx = [], [], [], []
            ans_tex.append(tex_A1[0])
            ans_wtl.append(wtl_A1[0])
            ans_cmp.append(cmp_A1[0])
            ans_idx.append(idx_A1[0])
            for i in range(nb_set - 1):
                ans_tex.extend([tex_A1[i + 1], tex_A2[i], tex_Jt[i]])
                ans_wtl.extend([wtl_A1[i + 1], wtl_A2[i], wtl_Jt[i]])
                ans_cmp.extend([cmp_A1[i + 1], cmp_A2[i], cmp_Jt[i]])
                ans_idx.extend([idx_A1[i + 1], idx_A2[i], idx_Jt[i]])

            results = [[nkw], name_pru]
            results.extend(ans_tex)
            results.extend(ans_wtl)
            results.append(self.formulate_sign(ans_wtl))
            results.extend(ans_idx)
            tmp_idx = np.mean(ans_idx, axis=0).tolist()
            results.append(tmp_idx)
            results.extend(ans_cmp)
            results.append(self.formulate_sign(ans_cmp))
            self.tabulate_output(results, nkw)
        return

    def tabulate_output(self, results, kw):
        csv_t = open(kw + ".csv", "w", newline="")
        csv_w = csv.writer(csv_t)
        csv_w.writerows(results)
        csv_t.close()
        del csv_t, csv_w
        return

    def formulate_sign(self, wtl_or_cmp):
        wtl_or_cmp = np.array(wtl_or_cmp)
        pick_leng_pru = wtl_or_cmp.shape[1]
        form = []
        for j in range(pick_leng_pru):
            # w = sum(wtl_or_cmp[:, j] == 'W')
            # t = sum(wtl_or_cmp[:, j] == 'T')
            # l = sum(wtl_or_cmp[:, j] == 'L')
            bak = wtl_or_cmp[:, j]
            w = sum([1 if i == 'W' else 0 for i in bak])
            t = sum([1 if i == 'T' else 0 for i in bak])
            l = sum([1 if i == 'L' else 0 for i in bak])
            form.append(r"\'{}/{}/{}".format(w, t, l))
        return form

    def common_wtl_avg(self, U_raw, nb_set, index, tag_col,
                       mode='descend', rez=2, fair=False):
        ans_tex, ans_wtl, ans_cmp = [], [], []
        for i in range(nb_set):
            proposed = U_raw.loc[
                index[i]][tag_col[-1]].astype(DTY_FLT).values
            sign_A, G_A = comp_t_sing(proposed, self._nb_iter, rez)
            if fair:
                G_A = (abs(G_A[0]), G_A[1])

            tmp_wtl, tmp_cmp, tmp_avg = [], [], []
            for j in tag_col[: -1]:
                compared = U_raw.loc[index[i]][j].astype(DTY_FLT).values
                sign_B, G_B = comp_t_sing(compared, self._nb_iter, rez)
                mk_mu, mk_s2 = comp_t_prep(proposed, compared)
                if fair:
                    G_B = (abs(G_B[0]), G_B[1])

                tmp_wtl.append(cmp_paired_wtl(G_A, G_B, mk_mu, mk_s2, mode))
                tmp_cmp.append(cmp_paired_avg(G_A, G_B, mode))
                tmp_avg.append(sign_B)
            tmp_avg.append(sign_A)

            ans_tex.append(tmp_avg)
            ans_wtl.append(tmp_wtl)
            ans_cmp.append(tmp_cmp)
        return ans_tex, ans_wtl, ans_cmp

    def verify_paired_ttest_prus(self, df, kw, rel_id,
                                 category=None, ind_ens=True,
                                 nb_set=5, index=None):
        tag_trn, tag_tst = self.pick_up_comparison(rel_id)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category=category, ind_ens=ind_ens)
        tag_col = tag_trn if rel_id < 0 else tag_tst

        U_avg, U_std, _, U_raw = pd_concat_divide_sht(
            df, tag_col, nb_set, index)
        mode = 'ascend'
        if rel_id in [6, 7, 8, 9]:
            mode = 'descend'
        rez = 2 if rel_id in [6, 7, 8] else 4
        if rel_id in [6, 7, 8]:
            U_raw *= 100

        _, idx_bar = Friedman_init(
            U_avg.astype(DTY_FLT).values, mode=mode)
        idx_bar = np.mean(idx_bar, axis=0).tolist()

        results = [[kw], name_pru]
        ans_tex = []
        for i in range(nb_set):
            ans_tex.append([_encode_sign(
                U_avg.loc[i][j],
                U_std.loc[i][j], 4) for j in tag_col])
        results.extend(ans_tex)

        ans_tex, ans_wtl, ans_cmp = self.common_wtl_avg(
            U_raw, nb_set, index, tag_col, mode, rez)
        results.extend(ans_wtl)
        results.extend(ans_tex)
        results.append(self.formulate_sign(ans_wtl))
        results.append(idx_bar)
        results.extend(ans_cmp)
        results.append(self.formulate_sign(ans_cmp))

        kw = kw + "_prus_rel{}".format(rel_id)
        self.tabulate_output(results, kw)

    def verify_paired_ttest_fair(self, df, rel_id, rel_jt,
                                 category=None, ind_ens=True,
                                 nb_set=5, index=None):
        ind = [np.add(i, rel_jt).tolist() for i in index]
        if rel_jt > 1:
            ind = ind[1:]
            nb_set -= 1

        tag_trn, tag_tst = self.pick_up_comparison(rel_id * 2)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category=category, ind_ens=ind_ens)

        U_g1_avg, _, _, U_g1_raw = pd_concat_divide_sht(
            df, tag_tst, nb_set, ind)
        mode = 'descend'
        vals = tag_tst

        tag_trn, tag_tst = self.pick_up_comparison(rel_id * 2 + 1)
        _, name_pru, tag_trn, tag_tst = self.pick_up_pru(
            tag_trn, tag_tst, category=category, ind_ens=ind_ens)

        U_g0_avg, _, _, U_g0_raw = pd_concat_divide_sht(
            df, tag_tst, nb_set, ind)
        mode = 'ascend'
        keys = tag_tst

        tag_columns = {k: v for k, v in zip(keys, vals)}
        U_g0_raw = U_g0_raw.rename(columns=tag_columns)
        U_raw = U_g1_raw - U_g0_raw
        U_raw = U_raw.abs()

        U_avg = U_g1_avg.values.astype(
            DTY_FLT) - U_g0_avg.values.astype(DTY_FLT)
        U_avg = np.abs(U_avg)
        _, idx_bar = Friedman_init(U_avg, mode=mode)

        ans_tex, ans_wtl, ans_cmp = self.common_wtl_avg(
            U_raw, nb_set, ind, vals, mode, rez=4, fair=True)
        return ans_tex, ans_wtl, ans_cmp, name_pru, idx_bar

    def tabulate_sec33_prus(self, raw_dframe, nb_set, index):
        for name_ens, rdf in zip(AVAILABLE_ABBR_ENSEM, raw_dframe):
            for key, val in rdf.items():
                kws = "_".join([_get_tmp_name_ens(name_ens), key])
                self.verify_paired_ttest_prus(
                    val, kws, 6, nb_set=nb_set, index=index)
                self.verify_paired_ttest_prus(
                    val, kws, 1, nb_set=nb_set, index=index)

        return

    def tabulate_sec33_fair(self, raw_dframe, nb_set, index):
        for name_ens, rdf in zip(AVAILABLE_ABBR_ENSEM, raw_dframe):
            for key, val in rdf.items():
                for rel_id in range(1, 4):
                    kws = "_".join([
                        _get_tmp_name_ens(name_ens), key,
                        "fair", "rel{}".format(rel_id)
                    ])

                    tex_A1, wtl_A1, cmp_A1, name_pru, _ = \
                        self.verify_paired_ttest_fair(
                            val, rel_id, 1, nb_set=nb_set, index=index)
                    tex_A2, wtl_A2, cmp_A2, _, _ = self.verify_paired_ttest_fair(
                        val, rel_id, 2, nb_set=nb_set, index=index)
                    tex_Jt, wtl_Jt, cmp_Jt, _, _ = self.verify_paired_ttest_fair(
                        val, rel_id, 3, nb_set=nb_set, index=index)

                    ans_tex, ans_wtl, ans_cmp = [], [], []
                    ans_tex.append(tex_A1[0])
                    ans_wtl.append(wtl_A1[0])
                    ans_cmp.append(cmp_A1[0])
                    for i in range(nb_set - 1):
                        ans_tex.extend([
                            tex_A1[i + 1], tex_A2[i], tex_Jt[i]])
                        ans_wtl.extend([
                            wtl_A1[i + 1], wtl_A2[i], wtl_Jt[i]])
                        ans_cmp.extend([
                            cmp_A1[i + 1], cmp_A2[i], cmp_Jt[i]])

                    results = [[kws], name_pru]
                    results.extend(ans_wtl)
                    results.extend(ans_tex)
                    results.extend(ans_cmp)
                    results.append(self.formulate_sign(ans_wtl))
                    results.append(self.formulate_sign(ans_cmp))
                    print("kws", kws)
                    self.tabulate_output(results, kws)


# -------------------------------------
# Section 3.5


class PlotJ_LambdaEffect(GraphSetup):
    def __init__(self, name_ens, nb_cls, nb_pru, nb_iter=2,
                 nb_lam=9,
                 trial_type='KFS_expt10', figname='expt10'):
        super().__init__(name_ens, nb_cls, nb_pru,
                         nb_iter=nb_iter, figname=figname)
        self._trial_type = trial_type
        self._nb_lam = nb_lam

    def get_raw_filename(self, trial_type='KFS_expt10'):
        nmens_tmp = _get_tmp_document(self._name_ens, self._nb_cls)
        filename = "{}_iter{}_pms_lam{}.xlsx".format(
            nmens_tmp, self._nb_iter, self._nb_lam)
        if trial_type:
            trial_type += "_"
        return os.path.join(CURR_EXPT_DIR,  # RAW_EXPT_DIR,
                            "{}{}".format(trial_type, filename))

    def recap_sub_data(self, dframe, nb_row=3):
        nb_set = [len(v) for k, v in dframe.items()]
        assert len(set(nb_set)) == 1, "{}-{}".format(
            self._name_ens, self._nb_cls)
        nb_set = list(set(nb_set))[0]

        each = self._nb_iter * (self._nb_lam * 4 + 1) + 1
        nb_set = (nb_set - nb_row + 1) // each

        id_set = [i * each + nb_row - 1 for i in range(nb_set)]
        index = [[
            i + j * (self._nb_lam * 4 + 1) + 1 for j in range(
                self._nb_iter)] for i in id_set]
        return nb_set, id_set, index

    def pick_up_pru_idx(self, tag_col, rel_id=0):
        """
            [ut, ut.calc, us, seq]
        prus:
            La(MV), Lf(MV), Lo(MV), E[La(f)], E[Lf(f,f')], E[Lf(f)],
            Acc, P, R, F1
        fair:
            unaware(g1,g0),
            group_fair1(g1,g0), group_fair2(g1,g0), group_fair3(g1,g0),
            accuracy_manual(g1,g0)
        """
        if rel_id is None:
            return [], []
        num_len, num_gap = 4 + 1, 10 * 2 + 4
        # rel_id:  -2,-1: ut,us

        if rel_id in [-2, -1]:
            ind_trn = [i * num_gap + 2 * rel_id for i in range(num_len)]
            ind_tst = []
            ind_trn = [4 + 2 + i for i in ind_trn]

            if rel_id == -1:
                ind_trn[0] = ind_trn[0] + 1
            elif rel_id == -2:
                ind_trn[0] = ind_trn[0] - 2

        else:
            ind_trn = [i * num_gap + rel_id for i in range(num_len)]
            ind_tst = [i * num_gap + rel_id + 10 for i in range(num_len)]
            ind_trn = [4 + 2 + i for i in ind_trn]
            ind_tst = [4 + 2 + i for i in ind_tst]

        tag_trn = [tag_col[i] for i in ind_trn]
        tag_tst = [tag_col[i] for i in ind_tst]
        return tag_trn, tag_tst

    def pick_up_set_idx(self, index, rel_set=0, rel_jt=0):
        ind_set = index[rel_set]
        ind_jt = list(range(self._nb_lam))
        ind_jt = [i + rel_jt * self._nb_lam for i in ind_jt]
        ind_jt = [i + 1 for i in ind_jt]
        return [[i + j for j in ind_jt] for i in ind_set]

    def prepare_graph(self):
        csv_row_1 = unique_column(8 + 2 + 24 * 5)
        return csv_row_1[8:]

    def schedule_mspaint(self, raw_dframe, abbr_cls=None):
        nb_set, id_set, index = self.recap_sub_data(raw_dframe)
        tag_col = self.prepare_graph()

        if abbr_cls is not None:
            dframe = raw_dframe[abbr_cls]
            kw = "_".join([self._figname,
                           _get_tmp_name_ens(self._name_ens),
                           abbr_cls])
            self.pipeline_prus(dframe, kw, nb_set, index, tag_col)
            self.pipeline_fair(dframe, kw, nb_set, index, tag_col)
            return

        for abbr_cls in AVAILABLE_ABBR_CLS:
            if abbr_cls != 'MLP':
                continue
            dframe = raw_dframe[abbr_cls]
            kw = "_".join([_get_tmp_name_ens(self._name_ens),
                           abbr_cls])  # self._figname,
            self.pipeline_prus(dframe, kw, nb_set, index, tag_col)
            self.pipeline_fair(dframe, kw, nb_set, index, tag_col)

    def verify_lam_effect_prus(self, df, kw, tag_col, rel_id=None):
        tag_trn, tag_tst = self.pick_up_pru_idx(tag_col, rel_id)
        tag_col = tag_trn if rel_id < 0 else tag_tst
        df = df[tag_col].astype(DTY_FLT)

        X = np.linspace(0, 1, self._nb_lam).tolist()
        Ys = df.values
        annotY = ('Ensem', 'EPAF-C', 'EPAF-D', 'POEPAF', 'POPEP')
        Ys = Ys[:, [0, 1, 2, 4]]  # 'EPAF-D.2', 'POEPAF.1'
        annotY = ('Ensem', 'EPAF-C', 'EPAF-D', 'POAF')

        annotX = {
            0: r"$\mathcal{L}_{err}(\mathbf{wv}_\rho)$",
            # 0: r"$\mathcal{L}_{acc}(\mathbf{wv}_\rho)$",
            1: r"$\mathcal{L}_{fair}(\mathbf{wv}_\rho)$",
            2: r"$\mathcal{L}(\mathbf{wv}_\rho)$",

            3: r"$G_1(\mathbf{wv}_\rho)$",
            4: r"$G_2(\mathbf{wv}_\rho)$",
            5: r"$\mathbf{E}[\mathcal{L}_{fair}(f)]$",

            6: r"Test Accuracy (%)",
            7: r"Precision (%)",
            8: r"Recall (%)",
            9: r"f1_score",

            -1: r"Space Cost",
            -2: r"Time Cost (min)",
        }

        if rel_id in [6, 7, 8]:
            Ys *= 100
        annots = (r"$\lambda$", annotX[rel_id])

        if rel_id == -1:
            nkw = "us"
        elif rel_id == -2:
            nkw = "ut"
        else:
            nkw = "rel" + str(rel_id)
        nkw = "_".join([self._figname, "pru", nkw, kw])
        multiple_line_chart(X, Ys, annots, annotY, figname=nkw)

    def verify_lam_effect_fair(self, df, kw, tag_col, rel_id=None):
        tag_trn, tag_tst = self.pick_up_pru_idx(tag_col, 2 * rel_id)
        U_g1 = df[tag_tst].astype(DTY_FLT).values
        tag_trn, tag_tst = self.pick_up_pru_idx(tag_col, 2 * rel_id + 1)
        U_g0 = df[tag_tst].astype(DTY_FLT).values

        X = np.linspace(0, 1, self._nb_lam).tolist()
        Ys = (U_g1 - U_g0)[:, [0, 1, 2, 4]]
        Ys = np.abs(Ys)  # updated icml
        annotY = ('Ensem', 'EPAF-C', 'EPAF-D', 'POAF')

        annotX = {
            1: r"$diff(DP_F)$",
            2: r"$diff(EO_F)$",
            3: r"$diff(PQP_F)$",
            4: r"$diff($accuracy$)$",
            0: r"$diff(\cdot)$",
        }
        annotX = {
            1: r"$DP_F$",
            2: r"$EO_F$",
            3: r"$PQP_F$",
            4: r"$diff($accuracy$)$",
            0: r"$diff(\cdot)$",
        }

        annots = (r"$\lambda$", annotX[rel_id])
        nkw = "_".join([self._figname,
                        "fair_rel" + str(rel_id),
                        kw])
        multiple_line_chart(X, Ys, annots, annotY, figname=nkw)

    def pipeline_prus(self, dframe, kw, nb_set, index,
                      tag_col=None, verbose=False):
        if tag_col is None:
            tag_col = self.prepare_graph()
        for i in range(nb_set):
            if i > 0 and (not verbose):
                return  # continue

            ind_jt = self.pick_up_set_idx(index, i, 0)
            # if not verbose:
            #     ind_jt = ind_jt[: 1]
            for l, ind in enumerate(ind_jt):
                # if l > 0 and (not verbose):
                #     continue

                nkw = "_".join([
                    # kw, "itr{}".format(l), "set{}".format(i)])
                    "iter{}".format(l), "set{}".format(i), kw])
                tdf = dframe.loc[ind][tag_col]

                # CROPPED
                self.verify_lam_effect_prus(tdf, nkw, tag_col, 6)
                self.verify_lam_effect_prus(tdf, nkw, tag_col, 0)
                self.verify_lam_effect_prus(tdf, nkw, tag_col, 1)
                self.verify_lam_effect_prus(tdf, nkw, tag_col, 2)
                # CROPPED

    def pipeline_fair(self, dframe, kw, nb_set, index,
                      tag_col=None, verbose=False):
        if tag_col is None:
            tag_col = self.prepare_graph()
        for i in range(nb_set):
            if i > 0 and (not verbose):
                return  # continue

            ind_jt = self.pick_up_set_idx(index, i, 1)
            for l, ind in enumerate(ind_jt):
                nkw = "_".join([
                    # kw, "itr{}".format(l), "set{}".format(i)])
                    "iter{}".format(l), "set{}".format(i), kw])
                tdf = dframe.loc[ind][tag_col]

                self.verify_lam_effect_fair(tdf, nkw, tag_col, 1)
                self.verify_lam_effect_fair(tdf, nkw, tag_col, 2)
                self.verify_lam_effect_fair(tdf, nkw, tag_col, 3)
                self.verify_lam_effect_fair(tdf, nkw, tag_col, 4)

            if i == 0:  # not 1!
                continue

            ind_jt = self.pick_up_set_idx(index, i, 2)
            for l, ind in enumerate(ind_jt):
                nkw = "_".join([
                    # kw, "itr{}".format(l), "set{}".format(i)])
                    "iter{}".format(l), "set{}".format(i), kw])
                tdf = dframe.loc[ind][tag_col]

                self.verify_lam_effect_fair(tdf, nkw, tag_col, 1)
                self.verify_lam_effect_fair(tdf, nkw, tag_col, 2)
                self.verify_lam_effect_fair(tdf, nkw, tag_col, 3)
                self.verify_lam_effect_fair(tdf, nkw, tag_col, 4)

            ind_jt = self.pick_up_set_idx(index, i, 3)
            for l, ind in enumerate(ind_jt):
                nkw = "_".join([
                    # kw, "iter{}".format(l), "set{}".format(i)])
                    "iter{}".format(l), "set{}".format(i), kw])
                tdf = dframe.loc[ind][tag_col]

                self.verify_lam_effect_fair(tdf, nkw, tag_col, 1)
                self.verify_lam_effect_fair(tdf, nkw, tag_col, 2)
                self.verify_lam_effect_fair(tdf, nkw, tag_col, 3)
                self.verify_lam_effect_fair(tdf, nkw, tag_col, 4)


class PlotJGather_LambdaEffect(PlotJ_LambdaEffect):
    def __init__(self, name_ens, nb_cls, nb_pru,
                 nb_iter=2, nb_lam=9,
                 trial_type='KFS_expt10', figname='expt10'):
        super().__init__(name_ens, nb_cls, nb_pru, nb_iter,
                         nb_lam, trial_type, figname)

    def schedule_mspaint(self, raw_dframe):
        nb_set, id_set, index = self.recap_sub_data(raw_dframe)
        tag_col = self.prepare_graph()

        for i in range(nb_set):
            ind = self.pick_up_set_idx(index, i, 0)
            ind = np.array(ind).reshape(-1).tolist()
            # for rel_id in [6, 7, 8, 9, 0, 1, 2, -1, -2]:
            for rel_id in [6, 2, 0, 1]:  # 7, 8, 9,
                kw = "_".join([self._figname,
                               _get_tmp_name_ens(self._name_ens),
                               "pru_rel" + str(rel_id),
                               "set{}".format(i)])
                self.merge_plot_pru(
                    raw_dframe, tag_col, ind, kw, rel_id)

    def merge_plot_pru(self, rdf, tag_col, ind, kw, rel_id=None):
        tag_trn, tag_tst = self.pick_up_pru_idx(tag_col, rel_id)
        tag_col = tag_trn if rel_id < 0 else tag_tst
        ndf = pd.concat([v.loc[ind][tag_col] for v in rdf.values()])

        X = np.linspace(0, 1, self._nb_lam).tolist()
        X = X * self._nb_iter * len(rdf)
        Ys = ndf.astype(DTY_FLT).values[:, [0, 1, 2, 4]]
        annotY = ('Ensem', 'EPAF-C', 'EPAF-D', 'POAF')

        annotX = {
            0: r"$\mathcal{L}_{err}(\mathbf{wv}_\rho)$",
            # 0: r"$\mathcal{L}_{acc}(\mathbf{wv}_\rho)$",
            1: r"$\mathcal{L}_{fair}(\mathbf{wv}_\rho)$",
            2: r"$\mathcal{L}(\mathbf{wv}_\rho)$",

            3: r"$G_1(\mathbf{wv}_\rho)$",
            4: r"$G_2(\mathbf{wv}_\rho)$",
            5: r"$\mathbf{E}[\mathcal{L}_{fair}(f)]$",

            6: r"Test Accuracy (%)",
            7: r"Precision (%)",
            8: r"Recall (%)",
            9: r"f1_score",

            -1: r"Space Cost",
            -2: r"Time Cost (min)",
        }

        mkrs = ('.',) * 4
        if rel_id in [6, 7, 8]:
            Ys *= 100
        annots = (r"$\lambda$", annotX[rel_id])
        multiple_line_chart(X, Ys, annots, annotY, mkrs, kw)

    def merge_plot_fair(self, rdf, tag_col, ind, kw, rel_id=None):
        tag_trn, tag_tst = self.pick_up_pru_idx(tag_col, 2 * rel_id)
        U_g1 = pd.concat([v.loc[ind][tag_col] for v in rdf.values()])
        U_g1 = U_g1.astype(DTY_FLT).values

        tag_trn, tag_tst = self.pick_up_pru_idx(tag_col, 2 * rel_id + 1)
        U_g0 = pd.concat([v.loc[ind][tag_col] for v in rdf.values()])
        U_g0 = U_g0.astype(DTY_FLT).values

        X = np.linspace(0, 1, self._nb_lam).tolist()
        X = X * self._nb_iter * len(rdf)
        Ys = (U_g1 - U_g0)[:, [0, 1, 2, 4]]
        annotY = ('Ensem', 'EPAF-C', 'EPAF-D', 'POAF')

        annotX = {
            1: r"$diff(DP_F)$",
            2: r"$diff(EO_F)$",
            3: r"$diff(PQP_F)$",
            4: r"$diff($accuracy$)$",
            0: r"$diff(\cdot)$",
        }

        mkrs = ('.',) * 4
        annots = (r"$\lambda$", annotX[rel_id])
        multiple_line_chart(X, Ys, annots, annotY, mkrs, kw)
