# coding: utf-8

import argparse
import logging
import json
import os
import pdb
import time
import numpy as np

from fairml.datasets import RAW_EXPT_DIR
from experiment.wp2_oracle.fetch_data import DataSetup
from experiment.wp2_oracle.fvote_addtl import (
    PlotD_Measures, GatherD_Measures)
CURR_EXPT_DIR = os.path.join(RAW_EXPT_DIR, 'wp2_oracle')


class FairVoteDrawing(DataSetup):
    def __init__(self, trial_type, data_type,
                 nb_cls, nb_pru, nb_iter=5, ratio=.5, lam=.5,
                 name_ens="AdaBoostM1", abbr_cls="DT",
                 gather=False, screen=True, logged=False):
        super().__init__(data_type)
        self._trial_type = trial_type

        self._nb_cls = nb_cls
        self._nb_pru = nb_pru
        self._nb_iter = nb_iter
        self._ratio = ratio
        self._lam = lam

        self._log_document = "_".join([
            trial_type, "{}vs{}".format(nb_cls, nb_pru),
            "iter{}".format(nb_iter), self._log_document,
            "ratio{}".format(int(ratio * 100)), "pms"])

        self._name_ens = name_ens
        self._abbr_cls = abbr_cls
        if trial_type.endswith('expt3'):
            nmens_tmp = _get_tmp_name_ens(name_ens)
            self._log_document = self._log_document.replace(
                data_type, '')
            self._log_document += "_{}_{}_{}".format(
                nmens_tmp, abbr_cls, data_type)
        if trial_type.endswith('expt6'):
            self._log_document += "_lam{}".format(int(lam * 100))

        self._screen = screen
        self._logged = logged

        if self._trial_type.endswith('expt1'):
            self._iterator = PlotA_Measures()

        elif self._trial_type.endswith('expt4') and (not gather):
            self._iterator = PlotD_Measures()
        elif self._trial_type.endswith('expt5') and (not gather):
            self._iterator = PlotE_Measures()
        elif self._trial_type.endswith('expt6') and (not gather):
            self._iterator = PlotF_Prunings()

        elif self._trial_type.endswith('expt4') and gather:
            self._iterator = GatherD_Measures()
        elif self._trial_type.endswith('expt6') and gather:
            self._iterator = GatherF_Prunings()
        elif self._trial_type.endswith('expt5') and gather:
            self._iterator = GatherE_Measures()

        elif self._trial_type.endswith('expt2'):
            self._iterator = PlotB_Measures()
        self._figname = "{}_{}".format(trial_type, data_type)
        if gather:
            self._figname = "{}_entire".format(trial_type)

    def trial_one_process(self):
        since = time.time()
        # START

        json_rf = open(os.path.join(
            CURR_EXPT_DIR, self._log_document + ".json"), 'r')
        # json_rf = open(self._log_document + ".json", 'r')
        content = json_rf.read()
        json_reader = json.loads(content)
        res_all = json_reader['res_all']
        res_data = json_reader['res_data']

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
        else:
            self._iterator.schedule_mspaint(res_data, res_all,
                                            self._figname)
        json_rf.close()  # file to read
        del json_rf, content, json_reader
        del res_all, res_data
        if self._trial_type.endswith('expt6'):
            rm_ehandler(logger, formatter, fileHandler)

        # END
        tim_elapsed = time.time() - since
        return

    def trial_gather_process(self):
        since = time.time()
        # START

        # optional_data = ["ricci", "german", "adult", "ppc", "ppvc"]
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

            json_rf = open(os.path.join(
                CURR_EXPT_DIR, log_document + ".json"), 'r')
            # json_rf = open(log_document + ".json", 'r')
            content = json_rf.read()
            json_reader = json.loads(content)
            res_all[data_type] = json_reader['res_all']
            res_data[data_type] = json_reader['res_data']
            json_rf.close()
            del json_rf, content, json_reader

        if self._trial_type.endswith('expt5'):
            self._iterator.schedule_mspaint(
                res_data, res_all, optional_data,
                self._figname, jt=True)
        elif self._trial_type.endswith('expt6'):
            if os.path.exists(self._figname + ".txt"):
                os.remove(self._figname + ".txt")
            logger, formatter, fileHandler = get_elogger(
                "fvote", self._figname + ".txt")
            self._iterator.schedule_mspaint(
                res_data, res_all, optional_data,
                self._figname, jt=False, logger=logger)
        else:
            self._iterator.schedule_mspaint(
                res_data, res_all, optional_data, self._figname)
        del res_data, res_all, optional_data
        if self._trial_type.endswith('expt6'):
            rm_ehandler(logger, formatter, fileHandler)

        # END
        tim_elapsed = time.time() - since
        return


def parameters():  # default
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
        help="Construct ensemble")
    parser.add_argument(
        "--abbr-cls", type=str, default="DT",
        choices=[
            "DT", "NB", "SVM", "linSVM", "LR", "kNNu", "kNNd", "MLP",
            "lmSGD", "NN", "LM", "LR1", "LR2", "LM1", "LM2"
        ], help="Individual classifiers")
    parser.add_argument(
        "--nb-cls", type=int, default=21, help="Size of ensemble")
    parser.add_argument(
        "--nb-pru", type=int, default=11,
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
    return parser


parser = parameters()
args = parser.parse_args()

trial_type = args.expt_id
data_type = args.dataset
nb_cls = args.nb_cls
nb_pru = args.nb_pru
nb_iter = args.nb_iter
ratio = args.ratio
lam = args.lam
screen = args.screen
logged = args.logged


kwargs = {}
if trial_type.endswith('expt3'):
    kwargs["name_ens"] = args.name_ens
    kwargs["abbr_cls"] = args.abbr_cls
elif trial_type[-5:] in ('expt4', 'expt5', 'expt6'):
    kwargs["gather"] = args.gather

if (not args.draw) and ('gather' in kwargs):
    kwargs.pop('gather')


case = FairVoteDrawing(
    trial_type, data_type, nb_cls, nb_pru, nb_iter, ratio, lam,
    screen=screen, logged=logged, **kwargs)
if not args.gather:
    case.trial_one_process()
else:
    case.trial_gather_process()


# Empirical results in manuscript
"""
python wp1_main_plot.py --draw -exp mCV_expt4 --gather
"""

# Empirical results not in use
"""
python wp1_main_plot.py --draw -exp mCV_expt4 -dat german
python wp1_main_plot.py --draw -exp mCV_expt4 -dat adult
python wp1_main_plot.py --draw -exp mCV_expt4 -dat ppr
python wp1_main_plot.py --draw -exp mCV_expt4 -dat ppvr
"""
