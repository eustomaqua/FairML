# Discriminative risk (DR)


We propose a fairness quality measure named *discriminative risk (DR)* to reflect both individual and group fairness aspects. We also investigate its properties and establish the first and second-order oracle bounds concerning fairness to show that fairness can be boosted via ensemble combination with theoretical learning guarantees. The analysis is suitable for both binary and multi-class classification. Furthermore, an ensemble pruning method named *POAF (Pareto optimal ensemble pruning via improving
accuracy and fairness concurrently)* is also proposed to utilise DR. Comprehensive experiments are conducted to evaluate the effectiveness of the proposed methods.

The full paper entitled *Increasing Fairness via Combination with Learning Guarantees* can be found on [arXiv](https://arxiv.org/pdf/2301.10813). There are also a couple of short versions for dissemination purposes only, see a non-archival [document](https://openreview.net/pdf?id=QHILhNkVUX) and its [poster](https://eustomadew.github.io/posters/2024_m3l_bounds.pdf), as well as [slides'23](https://eustomadew.github.io/slides/pre23_letall.pdf) and [slides'24](https://eustomadew.github.io/slides/pre24_melanie.pdf).


## Methodology

**Discriminative risk (DR)**

Following the principle of individual fairness, *the treatment/evaluation of one instance should not change solely due to minor changes in its sensitive attributes* (sen-att-s, aka. the protected attributes). If it happens, this indicates the existence of underlying *discriminative risks*.

Naturally, the *fairness quality* of one hypothesis $f(\cdot)$ can be evaluated by

$$ \ell_\text{bias}(f,\mathbf{x})= \mathbb{I}(\overbrace{ f(\breve{\mathbf{x}}, \mathbf{a})\neq f(\breve{\mathbf{x}}, \underbrace{ \tilde{\mathbf{a}} }_{\hbox{ slightly perturbed version of sen-att-s }} ) }^{ \hbox{$f$ makes a discriminative decision} }) \,.$$

This equation is defined on one instance, evaluating the risk from an individual aspect, and the empirical DR over one dataset describes this from a group aspect, as an unbiassed estimation of the true DR over one data distribution. There are no restrictions applying to the type of $f(\cdot)$.


**Oracle bounds and PAC bounds regarding fairness for the weighted voting**

If the weighted vote makes a discriminative decision, then *at least a $\rho$-weighted half* of the individual classifiers *have made a discriminative decision* and, therefore, the DR of an ensemble can be bounded by a constant times the DR of the individual classifiers. In other words, there exists a cancellation-of-biases effect in combination, similar to its well-known cancellation-of-error effect. We also provided two PAC bounds regarding fairness to bound the discrepancy between one hypothesis (either an individual classifier or an ensemble)'s empirical DR and its true DR.


**POAF for improving accuracy and fairness at the same time**

We use the domination concept with DR and the 0/1 loss function as two sub-objectives to be minimised, aiming to construct stronger ensemble classifiers with less accuracy damage. We also came up with two
extra pruning methods that could be easily implemented,
named as *EPAF (ensemble pruning via improving accuracy and
fairness concurrently),* presented in centralised and distributed versions (that is, EPAF-C and EPAF-D).



## Usage examples
<!-- Examples of how to use them -->

(1) Set parameters (changeable)
```python
import numpy as np
# np.random.seed(None)

k = 2        # k-th in Cross Validation
nb_cls = 21  # number of individual classifiers
nb_pru = 11  # size of the pruned sub-ensemble
lam = .5     # regularisation factor in bi-objective
n_m = 2      # number of machines in EPAF-D
```

(2) load one dataset
```python
from fairml.datasets import German, preprocess  # Ricci
from fairml.preprocessing import (
    adversarial, transform_X_and_y, transform_unpriv_tag)

dt = German()  # Ricci()
df = dt.load_raw_dataset()
processed_dat = preprocess(dt, df)
disturbed_dat = adversarial(dt, df, ratio=.97)
processed_dat = processed_dat['numerical-binsensitive']
disturbed_dat = disturbed_dat['numerical-binsensitive']

pos_label = dt.get_positive_class_val('')
non_sa, _ = transform_unpriv_tag(dt, processed_dat)
X, y = transform_X_and_y(dt, processed_dat)
Xp, _ = transform_X_and_y(dt, disturbed_dat)
y[y == 2] = 0  # only for German()

del processed_dat, disturbed_dat, df, dt
```

(3) divide the training set and test set
```python
from sklearn import model_selection

kf = model_selection.KFold(n_splits=5)
split_idx = []
for trn, tst in kf.split(y):
    split_idx.append((trn.tolist(), tst.tolist()))
del kf

i_trn, i_tst = split_idx[k]
X_trn, y_trn = X.iloc[i_trn], y.iloc[i_trn]
X_tst, y_tst = X.iloc[i_tst], y.iloc[i_tst]

Xp_trn = Xp.iloc[i_trn].to_numpy()
Xp_tst = Xp.iloc[i_tst].to_numpy()
X_trn, y_trn = X_trn.to_numpy(), y_trn.to_numpy()
X_tst, y_tst = X_tst.to_numpy(), y_tst.to_numpy()

nsa_idx_trn = [idx[i_trn] for idx in non_sa]
nsa_idx_tst = [idx[i_tst] for idx in non_sa]
```

(4) train one ensemble classifier
```python
from sklearn import ensemble
from sklearn import metrics
from fairml.facils.ensem_voting import weighted_voting

clf = ensemble.BaggingClassifier(n_estimators=nb_cls)
clf.fit(X_trn, y_trn)
yhat_trn = [clf.estimators_[i].predict(X_trn).tolist() for i in range(nb_cls)]
yhat_tst = [clf.estimators_[i].predict(X_tst).tolist() for i in range(nb_cls)]
yp_qtb_trn = [clf.estimators_[i].predict(Xp_trn).tolist() for i in range(nb_cls)]
yp_qtb_tst = [clf.estimators_[i].predict(Xp_tst).tolist() for i in range(nb_cls)]

coef = [1. / nb_cls] * nb_cls
y_trn, y_tst = y_trn.tolist(), y_tst.tolist()
hens_trn = weighted_voting(yhat_trn, coef)
hens_tst = weighted_voting(yhat_tst, coef)
hp_qtb_trn = weighted_voting(yp_qtb_trn, coef)
hp_qtb_tst = weighted_voting(yp_qtb_tst, coef)

def get_accuracy(y, y_hat):
    # return np.mean(np.equal(y, y_hat))
    return metrics.accuracy_score(y, y_hat)
# print accuracy e.g.
acc = get_accuracy(y_tst, hens_tst)
acc_qtb = get_accuracy(y_tst, hp_qtb_tst)
```

(5) compute the discriminative risk and three group fairness measures
```python
from fairml.discriminative_risk import hat_L_fair  # ,hat_L_loss
from fairml.facils.metric_fair import (
    unpriv_group_one, unpriv_group_two, unpriv_group_thr,
    marginalised_np_mat, calc_fair_group,)

def get_grp_fairness(y, y_hat, pos_lbl, non_sa):
    g1_Cm, g0_Cm = marginalised_np_mat(y, y_hat, pos_lbl, non_sa)
    bias_grp1 = unpriv_group_one(g1_Cm, g0_Cm)
    bias_grp2 = unpriv_group_two(g1_Cm, g0_Cm)
    bias_grp3 = unpriv_group_thr(g1_Cm, g0_Cm)
    bias_grp1 = calc_fair_group(*bias_grp1)
    bias_grp2 = calc_fair_group(*bias_grp2)
    bias_grp3 = calc_fair_group(*bias_grp3)
    return bias_grp1, bias_grp2, bias_grp3

dr = hat_L_fair(hens_tst, hp_qtb_tst)
grp_sa1 = get_grp_fairness(y_tst, hens_tst, pos_label, nsa_idx_tst[0])
grp_sa2 = get_grp_fairness(y_tst, hens_tst, pos_label, nsa_idx_tst[1])
```

(6) get sub-ensembles using different pruning methods
```python
from fairml.dr_pareto_optimal import POAF_PEP as POAF
from fairml.dr_pareto_optimal import Centralised_EPAF_Pruning as EPAF_C
from fairml.dr_pareto_optimal import Distributed_EPAF_Pruning as EPAF_D

def get_subensemble(y, y_hat, y_hat_qtb, coef, H, non_sa):
    pru_y_hat = np.array(y_hat)[H].tolist()
    pru_yp_qtb = np.array(y_hat_qtb)[H].tolist()
    pruned_coef = np.array(coef)[H].tolist()
    h_ens = weighted_voting(pru_y_hat, pruned_coef)
    hp_qtb = weighted_voting(pru_yp_qtb, pruned_coef)

    acc = get_accuracy(y, h_ens)
    acc_qtb = get_accuracy(y, hp_qtb)
    dr = hat_L_fair(h_ens, hp_qtb)
    grp_sa = [get_grp_fairness(
      y, h_ens, pos_label, i) for i in non_sa]
    return

seq = POAF(y_trn, yhat_trn, yp_qtb_trn, coef, lam, nb_pru)
# H = np.zeros(nb_cls, dtype='bool')
# H[seq] = 1
get_subensemble(y_tst, yhat_tst, yp_qtb_tst, coef, seq, nsa_idx_tst)

seq = EPAF_C(y_trn, yhat_trn, yp_qtb_trn, coef, nb_pru, lam)
seq = EPAF_D(y_trn, yhat_trn, yp_qtb_trn, coef, nb_pru, lam, n_m)
```

```python
from fairml.dr_pareto_optimal import Ranking_based_fairness_Pruning

H, _ = Ranking_based_fairness_Pruning(
    y_trn, yhat_trn, yp_qtb_trn, nb_pru, lam, 'DR', pos_label)
get_subensemble(yhat_trn, yhat_tst, yp_qtb_trn, yp_qtb_tst,
                coef, H, nsa_idx_tst)

for criterion in ['DP', 'EO', 'PQP']:
    for i, _ in enumerate(non_sa):
        H, _ = Ranking_based_fairness_Pruning(
            y_trn, yhat_trn, yp_qtb_trn, nb_pru, lam,
            criterion, idx_priv=nsa_idx_trn[i])
```


## Empirical result reproduction