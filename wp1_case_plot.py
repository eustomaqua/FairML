# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority voting


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fairml.facilc.draw_graph import (  # .graph.utils_param
    PLT_LOCATION, DTY_PLT, PLT_AX_STYLE,
    _setup_config, _setup_figsize, _setup_figshow, _barh_kwargs)


# -----------------------
# Case study (results)


model_string = ["Full", "Unaware", "Fair K", "Fair add"]

mse = {'org': [0.7755605, 0.80419207, 0.86384302, 0.86296964],
       'qtb': [0.90229589, 0.80419207, 0.86393648, 0.859056]}
rmse = {'org': [0.88065913, 0.89676757, 0.92943156, 0.92896159],
        'qtb': [0.94989257, 0.89676757, 0.92948184, 0.92685274]}

att1 = {
    'DP': {'org': [0.004948, 0.052723, 0, 0.01735],
           'qtb': [0.051301, 0.052723, 0, 0.014767]},
    'EO': {'org': [0.011466, 0.017947, 0, 0.018456],
           'qtb': [0.031385, 0.017947, 0, 0.013989]},
    'PQP': {'org': [0.034763, 0.016614, 0.041549, 0.040802],
            'qtb': [0.016532, 0.016614, 0.041549, 0.04199]},
    'DR': [0.4217655301, 0, 0, 0.01261092947],
    'loss': [0.3825315273, 0.4093881364, 0.4607659972, 0.466137319]} 

att2 = {
    'DP': {'org': [0.775993, 0.347226, 0, 0.010956],
           'qtb': [0.247111, 0.347226, 0, 0.004904]},
    'EO': {'org': [0.738663, 0.321692, 0, 0.002868],
           'qtb': [0.26862, 0.321692, 0, 0.01135]},
    'PQP': {'org': [0.233629, 0.313737, 0.323618, 0.327041],
            'qtb': [0.348565, 0.313737, 0.323618, 0.325454]},
    'DR': [0.4217655301, 0, 0, 0.01261092947],
    'loss': [0.3825315273, 0.4093881364, 0.4607659972, 0.466137319]}


# -----------------------
# Case study plots


fair_string = ["DP", "EO", "PQP", "DR", "loss"]
bias_risk = ['undisturbed', 'disturbed']


def _subplot_bar_pl(ax, names, annot, val_1, val_2):
    num = len(names)
    ind = np.arange(num)  # x locations for group
    wid = .4  # width of the bars: can also be len(x) sequences
    h1_facecolor = tuple([0, 0.5019, 0.5020])
    h2_facecolor = tuple([i / 255 for i in [64, 224, 208]])

    # p1, p2 =
    ax.bar(ind - wid / 2, val_1, wid, color=h1_facecolor,
           label=bias_risk[0], **_barh_kwargs)
    ax.bar(ind + wid / 2, val_2, wid, color=h2_facecolor,
           label=bias_risk[1], linestyle='dashed', **_barh_kwargs)
    ax.legend(loc='lower right', labelspacing=.05)
    ax.set_ylabel(annot)
    ax.set_xticks(ind)
    ax.set_xticklabels(names, rotation=0)
    # ax.autoscale_view()
    return


def _subplot_scatter_pl(ax, ap, names, score, att, comparison=True):
    delta_score = np.subtract(score['org'], score['qtb'])
    delta_score = np.abs(delta_score)
    values = [np.subtract(att[i]['org'], att[i]['qtb']
                          ) for i in ['DP', 'EO', 'PQP']]
    val_alt = np.array(att['DR'])

    num = 2 + (3 if comparison else 0)
    ind = np.arange(4)  # x locations for group
    wid = .4  # width of the bars: can also be len(x) sequences
    colors = sns.color_palette(palette='muted', n_colors=5)

    if not comparison:
        p0 = ax.bar(ind - wid / 2, delta_score, wid,
                    color=colors[0], label=ap,  # alpha=.5,
                    linestyle='dashed', **_barh_kwargs)
        p4 = ax.bar(ind + wid / 2, val_alt, wid,
                    color=colors[4], label='DR', **_barh_kwargs)
    else:
        wid = .8 / num
        p0 = ax.bar(ind - 2 * wid, delta_score, wid,
                    color=colors[0], label=ap,  # alpha=.5,
                    linestyle='dashed', **_barh_kwargs)
        p1 = ax.bar(ind - wid, values[0], wid,
                    color=colors[1], label='DP', **_barh_kwargs)
        p2 = ax.bar(ind, values[1], wid,
                    color=colors[2], label='EO', **_barh_kwargs)
        p3 = ax.bar(ind + wid, values[2], wid,
                    color=colors[3], label='PQP', **_barh_kwargs)
        p4 = ax.bar(ind + 2 * wid, val_alt, wid,
                    color=colors[4], label='DR', **_barh_kwargs)

    ax.legend(loc='best', labelspacing=.05, frameon=False)
    # ax.set_ylabel('Fairness')
    ax.set_xticks(ind)
    ax.set_xticklabels(names, rotation=0)
    return


def counterfactual_fairness_case(score, att1, att2, annot, ap,
                                 figname='fair_cf', figsize='L-WS'):
    fig, ax = plt.subplots(figsize=_setup_config['L-NT'])

    _subplot_bar_pl(
        ax, model_string, annot, score['org'], score['qtb'])
    fig = _setup_figsize(fig, figsize, invt=False)
    _setup_figshow(fig, figname + '_model_score' + DTY_PLT)
    plt.clf()  # clear(fig)

    ax = fig.add_subplot(111)
    _subplot_scatter_pl(ax, ap, model_string, score, att1,
                        comparison=False)
    _setup_figsize(fig, figsize, invt=False)
    _setup_figshow(fig, figname + '_fair_nocmp' + DTY_PLT)
    # plt.show()
    return


if __name__ == '__main__':
    score = mse
    annot = 'MSE'
    ap = r'$\Delta(MSE)$'  # annot_prime
    counterfactual_fairness_case(score, att1, att2, annot, ap)
    '''
    score = rmse
    annot, ap = 'RMSE', r'$\Delta(RMSE)$'
    counterfactual_fairness_case(score, att1, att2, annot, ap)
    '''


# Empirical results in manuscript
"""
python wp1_case_plot.py
python wp1_main_plot.py --draw -exp mCV_expt4 --gather
python wp1_main_plot.py --draw -exp mCV_expt6 --gather --nb-pru 7
python wp1_main_plot.py -exp mCV_expt3 --gather
python wp1_main_plot.py -exp mCV_expt11 --gather
python wp1_main_plot.py -exp mCV_expt8 --gather
python wp1_main_plot.py -exp mCV_expt8 --gather --tab
python wp1_main_plot.py -exp mCV_expt8 --name-ens Bagging
python wp1_main_plot.py -exp mCV_expt8 --name-ens AdaBoostM1 --nb-cls 11 --nb-pru 5
python wp1_main_plot.py -exp mCV_expt8 --name-ens SAMME --nb-cls 11 --nb-pru 5
python wp1_main_plot.py -exp mCV_expt10 --name-ens Bagging --nb-iter 2 --nb-cls 11
"""
