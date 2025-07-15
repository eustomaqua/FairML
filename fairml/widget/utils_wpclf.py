# coding: utf-8
# utils_clfs.py
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


# =====================================
# degree/core
# =====================================


ALG_NAMES = [
    "DT", "NB", "LR", "SVM", "linSVM", "MLP",
    "LR1", "LR2", "LM1", "LM2", "kNNu", "kNNd",
]


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


INDIVIDUALS = {
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
    'RF': RandomForestClassifier,
    'ET': ExtraTreesClassifier,
    'GradBoost': GradientBoostingClassifier,
}

HOMO_ENSEMBLES = {
    'Bagging': BaggingClassifier,
    'AdaBoost': AdaBoostClassifier,
}  # homogeneous

HETERO_ENSEMBLES = {
    'VotingC': VotingClassifier,
    'StackingC': StackingClassifier,
}  # heterogeneous

ENS_NAMES = [
    'RF', 'ET',
    'Bagging', 'AdaBoost',
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

'''
class GenericAlgorithm:
  def __init__(self):
    pass

  @property
  def classifier(self):
    return self._classifier

  @classifier.setter
  def classifier(self, value):
    self._classifier = value

  @property
  def alg_clf_name(self):
    return self._alg_clf_name

  @alg_clf_name.setter
  def alg_clf_name(self, value):
    self._alg_clf_name = value

  def get_param_info(self):
    return {}

  def get_default_params(self):
    return {}

  def get_supported_data_types(self):
    # raise NotImplementedError(
    #   "get_supported_data_types() in Algorithm is not implemented")
    return set(["numerical", "numerical-binsensitive"])

  def running(self, df_trn, df_tst,
              label, pos_val, sens_attrs, priv_vals,
              single_sensitive, params):
    # remove sensitive attributes from the training set
    trn_df_nosensitive = df_trn.drop(columns=sens_attrs)
    tst_df_nosensitive = df_tst.drop(columns=sens_attrs)

    # create and train the classifier
    clf = self.classifier
    y_trn = trn_df_nosensitive[label]
    X_trn = trn_df_nosensitive.drop(columns=label)
    clf.fit(X_trn, y_trn)

    # get the predictions on the test set
    y_tst = tst_df_nosensitive[label]
    X_tst = tst_df_nosensitive.drop(label, axis=1)

    tst_pred = clf.predict(X_tst)
    trn_pred = clf.predict(X_trn)
    return y_trn, y_tst, trn_pred, tst_pred, []
'''


# -------------------------------------
# DecisionTree.py
# GaussianNB.py
# LogisticRegression.py
# SVM.py
# -------------------------------------


'''
class DecisionTreeALG(GenericAlgorithm):
  def __init__(self):
    super().__init__()
    self.classifier = DecisionTreeClassifier()
    self.alg_clf_name = "DecisionTree"


class GaussianNBALG(GenericAlgorithm):
  def __init__(self):
    super().__init__()
    self.classifier = GaussianNB()
    self.alg_clf_name = "GaussianNB"


class LogicRegreALG(GenericAlgorithm):
  def __init__(self):
    super().__init__()
    self.classifier = LogisticRegression()
    self.alg_clf_name = "LR"


class SvmALG(GenericAlgorithm):
  def __init__(self):
    super().__init__()
    self.classifier = SVC()
    self.alg_clf_name = "SVM"


class LsvmALG(GenericAlgorithm):
  def __init__(self):
    super().__init__()
    self.classifier = LinearSVC()
    self.alg_clf_name = "LinearSVM"


class KnnALG(GenericAlgorithm):
  def __init__(self, name='kNNu'):
    super().__init__()
    wt = 'uniform' if name.endswith('u') else 'distance'
    self.classifier = KNeighborsClassifier(weights=wt)
    self.alg_clf_name = name


class LmodelALG(GenericAlgorithm):
  def __init__(self, name='LM1'):
    super().__init__()
    pt = 'l1' if name.endswith('1') else 'l2'
    self.classifier = SGDClassifier(penalty=pt)
    self.alg_clf_name = name


class MlpNetALG(GenericAlgorithm):
  def __init__(self):
    super().__init__()
    self.classifier = MLPClassifier()
    self.alg_clf_name = "NN"
'''


# -------------------------------------
# Ensemble
# -------------------------------------


'''
class EnsemAlgorithm(GenericAlgorithm):
  def __init__(self, nb_cls):
    # super().__init__()
    self._nb_cls = nb_cls

  def running(self, df_trn, df_tst,
              label, pos_val, sens_attrs, priv_vals,
              single_sensitive, params):
    trn_df_nosens = df_trn.drop(columns=sens_attrs)
    tst_df_nosens = df_tst.drop(columns=sens_attrs)

    ens = self.classifier
    y_trn = trn_df_nosens[label]
    X_trn = trn_df_nosens.drop(columns=label)
    ens.fit(X_trn, y_trn)

    y_tst = tst_df_nosens[label]
    X_tst = tst_df_nosens.drop(label, axis=1)

    tst_pred = ens.predict(X_tst)
    trn_pred = ens.predict(X_trn)

    return


class RandForestENS(EnsemAlgorithm):
  def __init__(self, nb_cls):
    super().__init__(nb_cls)
    self.classifier = RandomForestClassifier(n_estimators=nb_cls)
    self.alg_clf_name = 'RF_ens'
'''


# list.py
# -------------------------------------
'''
ALGORITHMS = {
    "DT": DecisionTreeALG(),
    "NB": GaussianNBALG(),
    "LR": LogicRegreALG(),
    "svm": SvmALG(),
    "lsvm": LsvmALG(),
    "kNNu": KnnALG(name='kNNu'),
    "kNNd": KnnALG(name='kNNd'),
    "LM1": LmodelALG(name='LM1'),
    "LM2": LmodelALG(name='LM2'),
    "NN": MlpNetALG(),  # 'nn'
    # individual baselines
}
'''


# -------------------------------------
