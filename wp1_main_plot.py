# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority vote
#

import argparse
import logging
# import json
import os
import sys
import time
import numpy as np
import pandas as pd

from fairml.widget.utils_saver import (get_elogger, rm_ehandler,
                                       elegant_print)
from fairml.widget.utils_timer import elegant_dated, elegant_durat
from fairml.widget.utils_const import _get_tmp_document
from fairml.widget.utils_remark import AVAILABLE_ABBR_ENSEM

from experiment.wp2_oracle.fetch_data import (
    pd_concat_divide_raw)  # DataSetup, CURR_EXPT_DIR,
# from experiment.wp2_oracle.fvote_addtl import (
#     PlotD_Measures, GatherD_Measures, GatherF_Prunings,
#     PlotF_Prunings, GatherE_Measures, PlotE_Measures,
#     PlotA_Measures, PlotB_Measures, Renew_GatherF_Prunings)
from experiment.wp2_oracle.fvote_draw import (
    PlotJ_LambdaEffect, PlotJGather_LambdaEffect,
    PlotH_ImprovePruning, PlotHGather_ImprovePruning,
    TableHGather_ImprovePruning, PlotC_TheoremsLemma,
    PlotK_PACGeneralisation)
from experiment.wp2_oracle.fvote_draw import PlotD_ImprovePruning  # legacy
from experiment.wp2_oracle.fvote_addtl import FairVoteDrawing
from experiment.wp2_oracle.rev_fig_repr import FVre_Drawing


# ----------------------------------
# Plotting
# ----------------------------------


class OracleDrawing(object):
    def __init__(self, trial_type,
                 name_ens, nb_cls, nb_pru=None,
                 nb_iter=5, lam=.45, partition=False,
                 nb_lam=9):
        self._trial_type = trial_type
        self._nb_iter = nb_iter
        self._lam = lam
        self._pn = partition
        self._iterator = None

        if trial_type.endswith('expt3'):
            self._iterator = PlotC_TheoremsLemma(
                name_ens, nb_cls,
                nb_iter=nb_iter, trial_type=trial_type)
        elif trial_type.endswith('expt11'):
            self._iterator = PlotK_PACGeneralisation(
                name_ens, nb_cls,
                nb_iter=nb_iter, trial_type=trial_type)

        elif trial_type.endswith('expt10'):
            self._iterator = PlotJ_LambdaEffect(
                name_ens, nb_cls, nb_pru, nb_iter, nb_lam,
                trial_type=trial_type)

        elif trial_type.endswith('expt8'):
            self._iterator = PlotH_ImprovePruning(
                name_ens, nb_cls, nb_pru, nb_iter,
                trial_type=trial_type)
        elif trial_type.endswith('expt4'):
            self._iterator = PlotD_ImprovePruning(
                name_ens, nb_cls, nb_pru, nb_iter,
                trial_type=trial_type)

        elif trial_type.endswith('expt5'):
            self._iterator = None

        nmens_tmp = _get_tmp_document(name_ens, nb_cls)
        self._log_document = "_".join([trial_type, nmens_tmp, "paint"])

    @property
    def iterator(self):
        return self._iterator

    @property
    def log_document(self):
        return self._log_document

    def trial_one_process(self):
        since = time.time()
        logger, formatter, fileHandler = get_elogger(
            "oracle_fair", self._log_document + '.log')
        elegant_print([
            "[BEGAN AT {:s}]".format(elegant_dated(since, 'txt')),
            "EXPERIMENT",
            "\t   trial = {}".format(self._trial_type),
            "PARAMETERS",
            "\tname_ens = {}".format(self._iterator.name_ens),
            # "\tabbr_cls = {}".format(self._iterator.abbr_cls),
            "\t  nb_cls = {}".format(self._iterator.nb_cls),
            "\t  nb_pru = {}".format(self._iterator.nb_pru),
            "\t nb_iter = {}".format(self._iterator.nb_iter),
            "HYPER-PARAMS", ""], logger)

        # START
        if self._trial_type.endswith('expt4'):
            self._iterator._logger = logger
        elif self._trial_type.endswith('expt8'):
            self._iterator._logger = logger
        self.trial_one_iteration(logger)
        # END

        tim_elapsed = time.time() - since
        since = time.time()
        elegant_print([
            "",  # "Duration in total"
            " Time Cost: {:s}".format(elegant_durat(tim_elapsed)),
            "[ENDED AT {:s}]".format(elegant_dated(since, 'txt')),
        ], logger)
        del since, tim_elapsed
        rm_ehandler(logger, formatter, fileHandler)
        logging.shutdown()
        return

    def trial_one_iteration(self, logger):
        since = time.time()

        trial_type = self._iterator.trial_type
        filename = self._iterator.get_raw_filename(trial_type)
        assert os.path.exists(filename), filename
        raw_dframe = self._iterator.load_raw_dataset(filename)

        if self._trial_type.endswith('expt10'):
            self._iterator.schedule_mspaint(raw_dframe, None)
        else:
            self._iterator.schedule_mspaint(raw_dframe, self._pn)

        tim_elapsed = time.time() - since
        elegant_print("\tDrawing: time cost {:.6f} minutes".format(
            tim_elapsed / 60), logger)
        return


class OracleGatheredDrawing(object):
    def __init__(self, trial_type, nb_iter=5, lam=.5,
                 nb_lam=9, tab=False):
        self._trial_type = trial_type
        self._nb_iter = nb_iter
        self._lam = lam
        self._tab = tab  # fig
        self._nb_lam = nb_lam

    @property
    def trial_type(self):
        return self._trial_type

    @property
    def nb_iter(self):
        return self._nb_iter

    @property
    def lam(self):
        return self._lam

    def get_log_document(self, name_ens, nb_cls):
        nmens_tmp = _get_tmp_document(name_ens, nb_cls)
        log_document = "_".join([self._trial_type, nmens_tmp])
        return log_document

    def get_hyper_params(self, name_ens):
        nb_cls, nb_pru = None, None

        if self._trial_type.endswith('expt3'):
            nb_cls = 21 if name_ens == "Bagging" else 11
        elif self._trial_type.endswith('expt11'):
            nb_cls, nb_pru = 11, 5

        elif self._trial_type.endswith('expt10'):
            nb_cls, nb_pru = 11, 5

        elif self._trial_type.endswith('expt8'):
            nb_cls, nb_pru = 11, 5

        elif self._trial_type.endswith('expt4'):
            nb_cls, nb_pru = 11, 5
        elif self._trial_type.endswith('expt6'):
            pass

        # return name_ens, nb_cls, nb_pru
        return nb_cls, nb_pru

    def get_iterator(self, name_ens):
        iterator = None

        if self._trial_type.endswith('expt3'):
            nb_cls, _ = self.get_hyper_params(name_ens)
            iterator = PlotC_TheoremsLemma(
                name_ens=name_ens, nb_cls=nb_cls,
                nb_iter=self._nb_iter, trial_type=self._trial_type)
        elif self._trial_type.endswith('expt11'):
            nb_cls, _ = self.get_hyper_params(name_ens)
            iterator = PlotK_PACGeneralisation(
                name_ens=name_ens, nb_cls=nb_cls,
                nb_iter=self._nb_iter, trial_type=self._trial_type)

        elif self._trial_type.endswith('expt10'):
            nb_cls, nb_pru = self.get_hyper_params(name_ens)
            iterator = PlotJGather_LambdaEffect(
                name_ens, nb_cls, nb_pru, self._nb_iter,
                nb_lam=self._nb_lam, trial_type=self._trial_type)

        elif self._trial_type.endswith('expt8'):
            nb_cls, nb_pru = self.get_hyper_params(name_ens)
            iterator = PlotH_ImprovePruning(
                name_ens=name_ens, nb_cls=nb_cls, nb_pru=nb_pru,
                nb_iter=self._nb_iter, trial_type=self._trial_type)

        elif self._trial_type.endswith('expt4'):
            nb_cls, nb_pru = self.get_hyper_params(name_ens)
            iterator = PlotD_ImprovePruning(
                name_ens, nb_cls, nb_pru,
                nb_iter=self._nb_iter, trial_type=self._trial_type)
        elif self._trial_type.endswith('expt6'):
            pass

        return iterator

    def trial_one_process(self):
        since = time.time()
        # log_document = self.get_log_document('', '')
        log_document = "paint_" + self._trial_type

        logger, formatter, fileHandler = get_elogger(
            "fair_paint", log_document + '.log')
        elegant_print([
            "[BEGAN AT {:s}]".format(elegant_dated(since, 'txt')),
            "EXPERIMENT",
            "\t   trial = {}".format(self._trial_type),
            "\t nb_iter = {}".format(self._nb_iter),
            "HYPER-PARAMS",
            "\t  lambda = {}".format(self._lam),
            ""], logger)

        # START
        self.trial_one_iteration(logger)
        # END

        tim_elapsed = time.time() - since
        since = time.time()
        elegant_print([
            "",  # "Duration in total"
            " Time Cost: {:s}".format(elegant_durat(tim_elapsed)),
            "[ENDED AT {:s}]".format(elegant_dated(since, 'txt')),
        ], logger)
        del since, tim_elapsed
        rm_ehandler(logger, formatter, fileHandler)
        logging.shutdown()
        return

    def trial_one_iteration(self, logger):
        since = time.time()

        raw_dframe = []
        # for name_ens in ["Bagging", "AdaBoostM1", "SAMME"]:
        for name_ens in AVAILABLE_ABBR_ENSEM:
            used_tc = time.time()

            _, nb_pru = self.get_hyper_params(name_ens)  # nb_cls,
            iterator = self.get_iterator(name_ens)
            filename = iterator.get_raw_filename(
                trial_type=self._trial_type)
            assert os.path.exists(filename), filename
            tmp_df = iterator.load_raw_dataset(filename)

            nb_set, id_set, index = iterator.recap_sub_data(tmp_df)
            tag_col = iterator.prepare_graph()
            # START

            if self._trial_type.endswith(
                    'expt3') or self._trial_type.endswith('expt11'):
                dframe = self.merge_simplex(tmp_df, tag_col, index)

            elif self._trial_type.endswith('expt10'):
                dframe = {k: v[tag_col] for k, v in tmp_df.items()}

            elif self._trial_type.endswith('expt8'):
                tag_col = iterator.update_column(tag_col)
                dframe = {k: v[tag_col] for k, v in tmp_df.items()}

            elif self._trial_type.endswith('expt4'):
                dframe = self.merge_complex(tmp_df, tag_col, nb_set, index)
            else:
                raise ValueError("Wrong trial_type: {}".format(self._trial_type))

            # END
            raw_dframe.append(dframe)
            used_tc = time.time() - used_tc
            elegant_print(
                "\tReading {:10s}: time cost {:.6f} seconds"
                "".format(name_ens, used_tc), logger)

        if self._trial_type[-5:] in [
            'expt3', 'expt4'] or self._trial_type.endswith(
                'expt11'):
            raw_dframe = pd.concat(raw_dframe, ignore_index=True)
        else:
            pass

        tim_elapsed = time.time() - since
        since = time.time()
        elegant_print("\tReading time cost: {:.6f} minutes"
                      "".format(tim_elapsed / 60), logger)

        # START
        if (self._trial_type[-5:] in ['expt3', 'expt4']) or (
                self._trial_type[-6:] in ['expt11', 'expt10']):
            self.trial_one_decomposition(logger, raw_dframe)
        else:
            self.trial_one_decomposition(logger, raw_dframe, tag_col)
        # END

        tim_elapsed = time.time() - since
        elegant_print("\tDrawing time cost: {:.6f} minutes"
                      "".format(tim_elapsed / 60), logger)
        return

    def merge_simplex(self, dframe, tag_col, index):
        index = np.concatenate(index, axis=0)
        # nvt = [v.iloc[index][tag_col] for v in dframe.values()]
        nvt = [v.loc[index][tag_col] for v in dframe.values()]
        dframe = pd.concat(nvt, ignore_index=True)
        return dframe

    def merge_complex(self, dframe, tag_col, nb_set, index):
        # avg, _, _, raw = pd_concat_divide_raw(
        _, _, _, raw = pd_concat_divide_raw(
            dframe, tag_col, nb_set, index)
        return raw  # or `avg`/`raw`

    def trial_one_decomposition(self, logger, raw_dframe,
                                tag_col=None):
        if self._trial_type.endswith('expt3'):
            iterator = PlotC_TheoremsLemma(None, None)

            iterator.verify_theorem31(raw_dframe, "all")
            iterator.verify_theorem33(raw_dframe, "all")
            iterator.verify_lemma32(raw_dframe, "all")
            iterator.verify_theorem34(raw_dframe, "all")

        elif self._trial_type.endswith('expt11'):
            iterator = PlotK_PACGeneralisation(None, 11)

            iterator.verify_theorem36(raw_dframe, "all")
            iterator.verify_theorem35(raw_dframe, "all")

        elif self._trial_type.endswith('expt10'):
            for i, name_ens in enumerate(AVAILABLE_ABBR_ENSEM):
                nb_cls, nb_pru = self.get_hyper_params(name_ens)
                iterator = self.get_iterator(name_ens)
                iterator.schedule_mspaint(raw_dframe[i])

        elif self._trial_type.endswith('expt8'):
            if not self._tab:
                iterator = PlotHGather_ImprovePruning(
                    # self._name_ens, self._nb_cls, self._nb_pru,
                    11, 5,
                    self._nb_iter, self._trial_type, logger=logger)
                iterator.schedule_mspaint(raw_dframe, tag_col)
            else:
                iterator = TableHGather_ImprovePruning(
                    11, 5, self._nb_iter, self._trial_type, None)
                iterator.schedule_spreadsheet(raw_dframe, tag_col)

        elif self._trial_type.endswith('expt4'):
            iterator = PlotD_ImprovePruning(None, None, None)
            iterator.verify_aggregated_rank(raw_dframe, "all")

        elif self._trial_type.endswith('expt6'):
            pass
        return


# ----------------------------------
# Executing
# ----------------------------------


def default_parameters():  # default
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-exp", "--expt-id", type=str, default="mCV_expt2",
        help="Experiment ID")
    parser.add_argument(
        "-dat", "--dataset", type=str, default="ricci",
        choices=["ricci", "german", "adult", "ppr", "ppvr"])

    parser.add_argument(
        "--name-ens", type=str, default="Bagging",
        choices=["Bagging", "AdaBoostM1", "SAMME"],
        help="Construct ensemble classifiers")
    parser.add_argument(
        "--abbr-cls", type=str, default="DT", choices=[
            "DT", "NB", "SVM", "linSVM", "LR", "kNNu", "kNNd",
            "MLP", "lmSGD", "NN", "LM", "LR1", "LR2", "LM1", "LM2"
        ], help="Individual /weak classifiers")
    parser.add_argument("--nb-cls", type=int, default=21,
                        help="Size of ensemble")
    parser.add_argument("--nb-pru", type=int, default=11,
                        help="Size of pruned sub-ensemble")

    parser.add_argument(
        "--nb-iter", type=int, default=5, help="Cross validation")
    parser.add_argument(
        "--lam", type=float, default=.25, help="Regularization factor")
    parser.add_argument(  # default=.5
        "--ratio", type=float, default=.95, help="Disturbing ratio")

    parser.add_argument(
        "--screen", action="store_true", help="Where to output")
    parser.add_argument(
        "--logged", action="store_true", help="Where to output")
    parser.add_argument(
        "--draw", action="store_true", help="Plotting pictures")
    parser.add_argument(
        "--gather", action="store_true", help="Plot them together")

    parser.add_argument('--tab', action='store_true')
    parser.add_argument('--nb-lam', type=int, default=9)
    parser.add_argument('-pn', '--partition', action='store_true')
    return parser


parser = default_parameters()
args = parser.parse_args()

trial_type = args.expt_id
nb_cls = args.nb_cls
nb_pru = args.nb_pru
nb_iter = args.nb_iter
screen = args.screen
logged = args.logged


kwargs = {}
if trial_type[-5:] in ('expt4', 'expt6',
                       'expt5', 'expt1', 'expt2'):
    # if trial_type.endswith('expt3'):
    #     kwargs["name_ens"] = args.name_ens
    #     kwargs["abbr_cls"] = args.abbr_cls

    kwargs["gather"] = args.gather
    if (not args.draw) and ('gather' in kwargs):
        kwargs.pop('gather')
    if not args.gather:
        kwargs["data_type"] = args.dataset

    case = FairVoteDrawing(
        trial_type, nb_cls, nb_pru, nb_iter,
        args.ratio, args.lam,
        screen=screen, logged=logged, **kwargs)
    if not args.gather:
        case.trial_one_process()
    else:
        case.trial_gather_process()
    if not trial_type.endswith('expt6'):
        sys.exit()

    del case
    case = FVre_Drawing(
        trial_type, nb_cls, nb_pru, nb_iter, args.ratio,
        args.lam, screen=screen, logged=logged, **kwargs)
    prefix = 'findings/wp2_oracle'
    if not args.gather:
        case.trial_one_process(prefix)
    else:
        case.trial_gather_process(prefix)
    del case, prefix
    sys.exit()  # 'expt5'


name_ens = args.name_ens
gather = args.gather
if trial_type.endswith('expt10'):
    kwargs['nb_lam'] = args.nb_lam
else:
    kwargs['lam'] = args.lam
    if not gather:
        kwargs['partition'] = args.partition
    else:
        kwargs['tab'] = args.tab
if not gather:
    case = OracleDrawing(trial_type, name_ens, nb_cls, nb_pru,
                         nb_iter, **kwargs)
elif trial_type[-5:] in ('expt3', 'xpt11', 'expt8'):
    case = OracleGatheredDrawing(trial_type, nb_iter, **kwargs)
case.trial_one_process()


del screen, logged, trial_type, gather
del name_ens, nb_cls, nb_pru, nb_iter
del case, kwargs, args, parser


# ----------------------------------
# Empirical results
# ----------------------------------

# Empirical results in manuscript
#   see `wp1_case_plot.py`

"""
python wp1_case_plot.py

python wp1_main_plot.py --draw -exp mCV_expt4 --gather
python wp1_main_plot.py --draw -exp mCV_expt6 --gather --nb-pru 7
python wp1_main_plot.py -exp mCV_expt8 --gather
python wp1_main_plot.py -exp mCV_expt8 --gather --tab
python wp1_main_plot.py -exp mCV_expt3 --gather
python wp1_main_plot.py -exp mCV_expt11 --gather
python wp1_main_plot.py -exp mCV_expt8 --name-ens Bagging
python wp1_main_plot.py -exp mCV_expt8 --name-ens AdaBoostM1 --nb-cls 11 --nb-pru 5
python wp1_main_plot.py -exp mCV_expt8 --name-ens SAMME --nb-cls 11 --nb-pru 5
python wp1_main_plot.py -exp mCV_expt10 --name-ens Bagging --nb-iter 2 --nb-cls 11
"""
