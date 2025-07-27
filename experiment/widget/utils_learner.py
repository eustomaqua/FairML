# coding: utf-8
# utils_wpclf(s).py
#
# TARGET:
#   Oracle bounds regarding fairness for majority vote
#


# sklearn
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import neural_network  # as nn

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier,
    # HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier)

import lightgbm
import fairgbm
# from lightgbm import LGBMClassifier
# from fairgbm import FairGBMClassifier
from experiment.widget.pkgs_AdaFair_py36 import AdaFair


# =====================================
# degree/core
# =====================================


# ALG_NAMES = [
#     "DT", "NB", "LR", "SVM", "linSVM", "MLP",
#     "LR1", "LR2", "LM1", "LM2", "kNNu", "kNNd",
# ]


CONCISE_ALG_NAMES = [
    'DT', 'NB', 'LR', 'SVM', 'linSVM',
    'kNNu', 'kNNd', 'MLP', 'lmSGD',  # 'NN','LM',
]

CONCISE_INDIVIDUALS = {
    'DT': tree.DecisionTreeClassifier(),
    'NB': naive_bayes.GaussianNB(),
    'LR': linear_model.LogisticRegression(),
    'SVM': svm.SVC(),
    'linSVM': svm.LinearSVC(),
    'kNNu': neighbors.KNeighborsClassifier(
        weights='uniform'),  # default
    'kNNd': neighbors.KNeighborsClassifier(weights='distance'),
    'MLP': neural_network.MLPClassifier(),
    'lmSGD': linear_model.SGDClassifier(),
}


# utils_remark.py
# -------------------------------------


FAIR_TEMPORARY = {
    'lightgbm': lightgbm.LGBMClassifier(),
    'fairgbm': fairgbm.FairGBMClassifier(),
    'AdaFair': AdaFair,
}

LGBMClassifier = lightgbm.LGBMClassifier
FairGBMClassifier = fairgbm.FairGBMClassifier


FAIR_INDIVIDUALS = {
    'DT': tree.DecisionTreeClassifier(),
    'NB': naive_bayes.GaussianNB(),
    'LR': linear_model.LogisticRegression(max_iter=500),
    'LR1': linear_model.LogisticRegression(
        penalty='none', max_iter=500),
    'LR2': linear_model.LogisticRegression(
        penalty='l2', max_iter=500),  # default

    'SVM': svm.SVC(),
    'linSVM': svm.LinearSVC(max_iter=5000),
    'kNNu': neighbors.KNeighborsClassifier(
        weights='uniform'),  # default
    'kNNd': neighbors.KNeighborsClassifier(weights='distance'),

    # 'NN': neural_network.MLPClassifier(
    #     solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)),
    'MLP': neural_network.MLPClassifier(max_iter=1000),
    'lmSGD': linear_model.SGDClassifier(),
    'LM1': linear_model.SGDClassifier(penalty='l1'),
    'LM2': linear_model.SGDClassifier(penalty='l2'),  # default
}

TREE_ENSEMBLES = {
    'RF': RandomForestClassifier(),
    'ET': ExtraTreesClassifier(),
    'GradBoost': GradientBoostingClassifier(),
}

HOMO_ENSEMBLES = {
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
}  # homogeneous

HETERO_ENSEMBLES = {
    'VotingC': VotingClassifier,
    'StackingC': StackingClassifier,
}  # heterogeneous

ENS_NAMES = [
    'RF', 'ET', 'Bagging', 'AdaBoost',
    'GradBoost',  # 'GradientBoost',
    'VotingC', 'StackingC',
]


# refs:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
# https://scikit-learn.org/0.24/modules/ensemble.html

# https://scikit-learn.org/0.24/modules/ensemble.html#bagging
# https://scikit-learn.org/0.24/modules/ensemble.html#adaboost
# https://scikit-learn.org/0.24/modules/ensemble.html#gradient-boosting
# https://scikit-learn.org/0.24/modules/ensemble.html#voting-classifier
# https://scikit-learn.org/0.24/modules/ensemble.html#stacked-generalization
# https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.VotingClassifier.html
# https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.StackingClassifier.html


# =====================================
# algorithms
# =====================================


'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
'''


# Algorithm.py
# Generic.py
# -------------------------------------


# -------------------------------------
# DecisionTree.py
# GaussianNB.py
# LogisticRegression.py
# SVM.py
# -------------------------------------


# -------------------------------------
# Ensemble
# -------------------------------------


# list.py
# -------------------------------------


# -------------------------------------
