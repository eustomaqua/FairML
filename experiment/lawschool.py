# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import time

import torch
from torch import nn
from tqdm import tqdm
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from fairml.widget.utils_saver import elegant_print
from fairml.widget.utils_timer import fantasy_durat


plt.rcParams['font.family'] = 'Times New Roman'
DTY_PLT = '.pdf'
CURR_PATH = 'findings/'
CURR_SAVE = True

pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)
assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)


# -----------------------
# Dataset


class LawSchoolData:
    def load_raw_dataset(self):
        curr_path = 'fairml/data/bar_pass_prediction.csv'
        return pd.read_csv(curr_path)  # data_frame

    def data_specific_processing(self, data_frame):
        cols = ["sex", "race", "ugpa", "lsat", "zfygpa"]
        cols.extend(['gpa'])
        data_df = data_frame[cols].dropna()
        return data_df

    # def handle_missing_data(self, data_frame):
    #   return data_frame

    def data_partitioning(self, data_frame, random_seed=None,
                          figname=None, verbose=False):
        train_set, test_set = train_test_split(
            data_frame, test_size=.2,
            stratify=data_frame[['sex', 'race']],
            random_state=random_seed)

        if figname is None:
            return train_set, test_set

        for attr in ["race", "sex"]:
            f, axes = plt.subplots(1, 3)
            for ax, dataset, name in zip(
                    axes, [data_frame, train_set, test_set],
                    ["whole", "train", "test"]):
                ax.hist(dataset[attr])
                ax.set_title(f"Dist. of {attr} in {name} data")
                f.tight_layout()
                f.set_size_inches(9, 3)
            if verbose:
                f.savefig("{}/{}_{}{}".format(
                    CURR_PATH, figname, attr, DTY_PLT), dpi=300)

        f, axes = plt.subplots(2, 3)
        for i, attr in enumerate(["race", "sex"]):
            for ax, dataset, name in zip(
                    axes[i], [data_frame, train_set, test_set],
                    ["whole", "train", "test"]):
                ax.hist(dataset[attr])
                ax.set_title(f"Dist. of {attr} in {name} data")
        f.tight_layout()
        f.set_size_inches(7.8, 4.5)
        # f.savefig("{}{}".format(figname, DTY_PLT), dpi=300)
        figpath = "{}/{}".format(CURR_PATH, figname)
        f.savefig(figpath + DTY_PLT, dpi=300)
        # f.savefig(figname + DTY_PLT, dpi=300)
        return train_set, test_set

    def adversarial(self, data_frame, ratio=.7,
                    sen_att=['sex', 'race']):
        unpriv_dict = [data_frame[sa].unique() for sa in sen_att]

        disturbed_data = data_frame.copy()
        dim = len(sen_att)  # num = len(disturbed_data)
        for i, ti in enumerate(data_frame.index):
            prng = np.random.rand(dim)
            prng = prng <= ratio
            curr_prev = data_frame.iloc[i][sen_att].values

            for j, (sa, up_dic) in enumerate(zip(
                    sen_att, unpriv_dict)):
                if not prng[j]:
                    continue

                # curr_up = np.random.random.choice(up_dic)
                # while curr_up == disturbed_data.loc[ti][sa]:
                #   curr_up = np.random.random.choice(up_dic)
                curr_post = up_dic.tolist()
                curr_post.remove(curr_prev[j])
                curr_post = np.random.choice(curr_post)
                # i.e., disturbed_data.iloc[i][sa]
                disturbed_data.loc[ti][sa] = curr_post
        return disturbed_data

    def visualization(self, data_set, model, curr_data,
                      curr_label=None, figname='visualization'):
        fit = data_set.copy()
        if curr_label is None:
            fit["FYA"] = model(curr_data).detach().numpy()
        else:
            fit["FYA"] = curr_label.detach().numpy()

        f, (ax1, ax2) = plt.subplots(2)
        f.set_size_inches(9, 6)
        sns.kdeplot(x="FYA", data=fit, hue="sex", ax=ax1,
                    legend=True, common_norm=False,
                    palette=sns.color_palette(n_colors=2))
        sns.kdeplot(x="FYA", data=fit, hue="race", ax=ax2,
                    legend=True, common_norm=False,
                    palette=sns.color_palette(n_colors=8))

        f.tight_layout()
        f.savefig("{}{}".format(figname, DTY_PLT), dpi=300)
        return


# =======================
# Models


# -----------------------
# helpers


def train_LinearModel(X_trn, y_trn, lr=.05,
                      num_iterations=500,
                      logger=None):

    lin_model = PyroModule[nn.Linear](X_trn.shape[1], 1)
    loss_fn = nn.MSELoss()  # torch.nn.MSELoss()
    # optim = torch.optim.Adam(lin_model.parameters(), lr=.05)
    # num_iterations = 500
    optim = torch.optim.Adam(lin_model.parameters(), lr=lr)

    def train():
        y_pred = lin_model(X_trn).squeeze(-1)
        loss = loss_fn(y_pred, y_trn)
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss

    for j in range(num_iterations):
        loss = train()
        if (j + 1) % 100 == 0:
            elegant_print("[Iteration %04d] RMSE loss: %.4f" % (
                j + 1, np.squeeze(loss.item())))
    elegant_print("Learned parameters:")
    for name, param in lin_model.named_parameters():
        elegant_print(
            "\t{:6s} {}".format(name, param.data.numpy()),
            logger)
    return lin_model, loss_fn


def visualization(data_set, model, curr_data, curr_label=None,
                  figname='visualization'):
    fit = data_set.copy()
    if curr_label is None:
        fit["FYA"] = model(curr_data).detach().numpy()
    else:
        fit["FYA"] = curr_label.detach().numpy()

    f, (ax1, ax2) = plt.subplots(2)
    f.set_size_inches(6, 5)
    sns.kdeplot(x="FYA", data=fit, hue="sex", ax=ax1,
                legend=True, common_norm=False,
                palette=sns.color_palette(n_colors=2))
    sns.kdeplot(x="FYA", data=fit, hue="race", ax=ax2,
                legend=True, common_norm=False,
                palette=sns.color_palette(n_colors=8))

    f.tight_layout()
    # f.savefig("{}{}".format(figname, DTY_PLT), dpi=300)
    figpath = "{}/{}".format(CURR_PATH, figname)
    f.savefig(figpath + DTY_PLT, dpi=300)
    return


def visualization_parallel(train_set, test_set, model,
                           curr_X_trn, curr_X_tst,
                           curr_y_trn=None, curr_y_tst=None,
                           sen_att=['sex', 'race'],
                           sen_num=[2, 8],
                           figname='visualization'):

    fit_trn = train_set.copy()
    fit_tst = test_set.copy()
    # if (curr_y_trn is None) or (curr_y_tst is None):
    #   fit_trn["FYA"] = model(curr_X_trn).detach().numpy()
    #   fit_tst["FYA"] = model(curr_X_tst).detach().numpy()
    # else:
    #   fit_trn["FYA"] = curr_y_trn.detach().numpy()
    #   fit_tst["FYA"] = curr_y_tst.detach().numpy()

    if curr_y_trn is None:
        fit_trn["FYA"] = model(curr_X_trn).detach().numpy()
    else:
        fit_trn["FYA"] = curr_y_trn.detach().numpy()
    if curr_y_tst is None:
        fit_tst["FYA"] = model(curr_X_tst).detach().numpy()
    else:
        fit_tst["FYA"] = curr_y_tst.detach().numpy()

    n_a = len(sen_att)
    _curr_font = 'Times New Roman'
    _curr_leg = {'family': _curr_font, 'size': 8}

    f, axes = plt.subplots(n_a, 2)
    f.set_size_inches(9, 5)
    for i, attr in enumerate(sen_att):
        ls1 = sns.kdeplot(
            x="FYA", data=fit_trn, hue=attr, ax=axes[i][0],
            legend=True, common_norm=False,
            palette=sns.color_palette(n_colors=sen_num[i]))
        ls2 = sns.kdeplot(
            x="FYA", data=fit_tst, hue=attr, ax=axes[i][1],
            legend=True, common_norm=False,
            palette=sns.color_palette(n_colors=sen_num[i]))

    f.tight_layout()
    # f.savefig("{}{}".format(figname, DTY_PLT), dpi=300)
    figpath = "{}/{}".format(CURR_PATH, figname)
    f.savefig(figpath + DTY_PLT, dpi=300)
    return


# -----------------------
# Full model or through unawareness


def _unaw_ful_no1_partition(train_set, test_set, picked_cols):
    sample_train = torch.tensor(
        train_set[picked_cols].values, dtype=torch.float32)
    sample_test = torch.tensor(
        test_set[picked_cols].values, dtype=torch.float32)
    train_X, train_y = sample_train[:, :-1], sample_train[:, -1]
    test_X, test_y = sample_test[:, :-1], sample_test[:, -1]
    return train_X, train_y, test_X, test_y


def _unaw_ful_no2_evaluate(test_X, test_y, curr_model,
                           loss_fn, picked, logger):
    y_pred = curr_model(test_X).squeeze(-1)
    loss = loss_fn(y_pred, test_y).item()
    elegant_print([
        # f"{picked} Model: RMSE on test samples = {np.square(loss):.4f}",
        f"{picked} Model: RMSE on test samples = {np.sqrt(loss):.4f}",
        ''], logger)
    return y_pred, loss


def _unaw_ful_no3_adversarial(X_trn, y_trn, X_qtb, curr_model, loss_fn,
                              picked, pick_trntst, logger):
    y_insp = curr_model(X_trn).squeeze(-1)
    loss = loss_fn(y_insp, y_trn).item()
    hx_qtb = curr_model(X_qtb).squeeze(-1)
    loss_q = loss_fn(hx_qtb, y_trn).item()

    elegant_print([
        f"{picked} Model: RMSE on {pick_trntst} samples",
        # f"\tnormally    RMSE   = {np.sqrt(loss):.4f}",
        # f"\t            loss   = {loss:.4f}",
        # f"\t            square = {np.square(loss):.4f}",
        # f"\tadversarial RMSE= {np.sqrt(loss_q):.4f}",

        f"\tsquare=  normal {np.square(loss):.8f}  adv {np.square(loss_q):.8f}",
        f"\tloss  =  normal {loss:.8f}  adv {loss_q:.8f}",
        f"\tRMSE  =  normal {np.sqrt(loss):.8f}  adv {np.sqrt(loss_q):.8f}",
    ], logger)
    return


def model_Unaware_or_Full(train_set, test_set, picked_cols,
                          picked, logger=None, figname='test_model',
                          train_adv=None, test_adv=None):
    train_X, train_y, test_X, test_y = _unaw_ful_no1_partition(
        train_set, test_set, picked_cols)
    curr_model, loss_fn = train_LinearModel(train_X, train_y)
    y_pred, loss = _unaw_ful_no2_evaluate(test_X, test_y, curr_model,
                                          loss_fn, picked, logger)
    visualization_parallel(train_set, test_set, curr_model,
                           train_X, test_X,
                           figname='{}_{}'.format(figname, picked))

    if (train_adv is None) or (test_adv is None):
        return
    X_trn_qtb, y_trn_q, X_tst_qtb, y_tst_q = _unaw_ful_no1_partition(
        train_adv, test_adv, picked_cols)
    assert (train_y == y_trn_q).all()
    assert (test_y == y_tst_q).all()
    _unaw_ful_no3_adversarial(train_X, train_y, X_trn_qtb, curr_model,
                              loss_fn, picked, 'training', logger)
    _unaw_ful_no3_adversarial(test_X, test_y, X_tst_qtb, curr_model,
                              loss_fn, picked, 'test', logger)
    elegant_print('\n', logger)
    visualization_parallel(train_adv, test_adv, curr_model,
                           X_trn_qtb, X_tst_qtb,
                           figname='{}_adv_{}'.format(figname, picked))

    y_insp = curr_model(train_X).squeeze(-1)
    y_pred = curr_model(test_X).squeeze(-1)
    hx_qtb_trn = curr_model(X_trn_qtb).squeeze(-1)
    hx_qtb_tst = curr_model(X_tst_qtb).squeeze(-1)
    return curr_model, loss_fn, (
        train_y, y_insp, hx_qtb_trn, test_y, y_pred, hx_qtb_tst)


# -----------------------
# Infer K


def LawSchoolModel(race, sex, gpa=None, lsat=None, fya=None):
    distributions = {
        'Inverse Gamma': dist.InverseGamma(
            torch.tensor(1.), torch.tensor(1.)),
        'Standard Normal': dist.Normal(
            torch.tensor(0.), torch.tensor(1.)),
    }

    k = pyro.sample("k", distributions['Standard Normal'])
    gpa0 = pyro.sample("gpa0", distributions['Standard Normal'])
    w_k_gpa = pyro.sample("w_k_gpa", distributions['Standard Normal'])
    w_r_gpa = pyro.sample("w_r_gpa", distributions['Standard Normal'])
    w_s_gpa = pyro.sample("w_s_gpa", distributions['Standard Normal'])

    lsat0 = pyro.sample("lsat0", distributions['Standard Normal'])
    w_k_lsat = pyro.sample("w_k_lsat", distributions['Standard Normal'])
    w_r_lsat = pyro.sample("w_r_lsat", distributions['Standard Normal'])
    w_s_lsat = pyro.sample("w_s_lsat", distributions['Standard Normal'])

    w_k_fya = pyro.sample("w_k_fya", distributions['Standard Normal'])
    w_r_fya = pyro.sample("w_r_fya", distributions['Standard Normal'])
    w_s_fya = pyro.sample("w_s_fya", distributions['Standard Normal'])

    sigma_gpa_square = pyro.sample(
        "sigma_gpa_sq", distributions['Inverse Gamma'])

    mean_gpa = gpa0 + k * w_k_gpa + race * w_r_gpa + sex * w_s_gpa
    param_lsat = lsat0 + k * w_k_lsat + race * w_r_lsat + sex * w_s_lsat
    mean_fya = k * w_k_fya + race * w_r_fya + sex * w_s_fya

    with pyro.plate("data", len(race)):
        gpa = pyro.sample("gpa", dist.Normal(
            mean_gpa, torch.square(sigma_gpa_square)), obs=gpa)
        lsat = pyro.sample("lsat", dist.Poisson(param_lsat.exp()), obs=lsat)
        fya = pyro.sample("fya", dist.Normal(mean_fya, 1), obs=fya)
        return gpa, lsat, fya


def model_Infer_K(data_tensor, picked):
    K_list = []
    for i in tqdm(range(data_tensor.shape[0])):
        conditioned = pyro.condition(LawSchoolModel, data={
            "gpa": data_tensor[i, 2],
            "lsat": data_tensor[i, 3].type(torch.int32),
            "fya": data_tensor[i, 4]})

        posterior = pyro.infer.Importance(conditioned,
                                          num_samples=10)
        marginal = pyro.infer.EmpiricalMarginal(
            posterior.run(race=data_tensor[:, 0],
                          sex=data_tensor[:, 1]),
            sites="k")
        K_list.append(marginal.mean)

    # with open(f"inferred_K_{picked}.pkl", 'wb') as f:
    with open(f"{CURR_PATH}/inferred_K_{picked}.pkl", 'wb') as f:
        pickle.dump(K_list, f)
    return


def _fair_k_infer_no1(train_set, test_set, CURR_FYA_LOC,
                      CURR_PATH, figname, logger):
    # Fair K model
    data_train_tensor = torch.tensor(train_set.values,
                                     dtype=torch.float32)
    data_test_tensor = torch.tensor(test_set.values,
                                    dtype=torch.float32)
    model_graph = pyro.render_model(LawSchoolModel, model_args=(
        data_train_tensor[:, 0], data_train_tensor[:, 1],
        data_train_tensor[:, 2], data_train_tensor[:, 3],
        data_train_tensor[:, 4]),
        render_distributions=True, render_params=True,
        filename=CURR_PATH + 'test.pdf')
    # print(model_graph)
    elegant_print(model_graph, logger)

    # Infer K
    if CURR_PATH == 'findings/' and (not CURR_SAVE):
        since = time.time()
        model_Infer_K(data_train_tensor, "train")
        model_Infer_K(data_test_tensor, "test")
        tim_elapsed = time.time() - since
        elegant_print("Infer K consumed: {}".format(
            fantasy_durat(tim_elapsed, abbreviation=True)), logger)
    # return

    with open(CURR_PATH + 'inferred_K_train.pkl', 'rb') as f:
        K_list = pickle.load(f)
    with open(CURR_PATH + 'inferred_K_test.pkl', 'rb') as f:
        K_list_test = pickle.load(f)

    # plt.hist(K_list)
    plt.figure(figsize=(7, 5))
    sns.kdeplot(np.array(K_list), label="Inferred K density")
    sns.kdeplot(np.random.randn(10000),
                label="Standard Gaussian density")
    plt.xlabel("K")
    plt.legend()
    # plt.show()
    # plt.savefig("{}_FairK_InferK{}".format(figname, DTY_PLT), dpi=300)
    figname = "{}{}_FairK_InferK".format(CURR_PATH, figname)
    plt.savefig(figname + DTY_PLT, dpi=300)

    # Predict FYA
    X_data, y_data = torch.tensor(
        K_list, dtype=torch.float32).reshape(
        -1, 1), data_train_tensor[:, CURR_FYA_LOC]
    model_fairK, loss_fn = train_LinearModel(X_data, y_data)

    y_pred_test = model_fairK(torch.tensor(
        K_list_test, dtype=torch.float32).reshape(-1, 1)).squeeze(-1)
    loss = loss_fn(y_pred_test,
                   data_test_tensor[:, CURR_FYA_LOC]).item()
    elegant_print([
        # f"Fair K Model: RMSE on test samples = {np.square(loss):.4f}",
        f"Fair K Model: RMSE on test samples = {np.sqrt(loss):.4f}",
        "\n"], logger)

    # y_insp_test = model_fairK(X_data).squeeze(-1)
    return model_fairK, loss_fn, X_data, y_pred_test


def _fair_k_infer_no3q(train_set, test_set, train_adv, test_adv,
                       CURR_FYA_LOC, CURR_PATH, logger):
    data_train_tensor = torch.tensor(train_adv.values,
                                     dtype=torch.float32)
    data_test_tensor = torch.tensor(test_adv.values,
                                    dtype=torch.float32)
    if CURR_PATH == 'findings/' and (not CURR_SAVE):
        since = time.time()
        model_Infer_K(data_train_tensor, "adv_trn")
        model_Infer_K(data_test_tensor, "adv_tst")
        tim_lapsed = time.time() - since
        elegant_print("Infer K adversarially consumed: {}".format(
            fantasy_durat(tim_elapsed, abbreviation=True)), logger)

    y_data_trn = torch.tensor(
        train_set.values, dtype=torch.float32)[:, CURR_FYA_LOC]
    y_data_tst = torch.tensor(
        test_set.values, dtype=torch.float32)[:, CURR_FYA_LOC]
    return y_data_trn, y_data_tst


def _fair_k_infer_no4q(y_data_trn, y_data_tst, curr_model, loss_fn,
                       CURR_PATH, logger):
    with open(CURR_PATH + 'inferred_K_train.pkl', 'rb') as f:
        K_list = pickle.load(f)
    with open(CURR_PATH + 'inferred_K_test.pkl', 'rb') as f:
        K_list_test = pickle.load(f)
    with open(CURR_PATH + 'inferred_K_adv_trn.pkl', 'rb') as f:
        K_list_adv_trn = pickle.load(f)
    with open(CURR_PATH + 'inferred_K_adv_tst.pkl', 'rb') as f:
        K_list_adv_tst = pickle.load(f)

    X_data_trn = torch.tensor(
        K_list, dtype=torch.float32).reshape(-1, 1)
    X_data_tst = torch.tensor(
        K_list_test, dtype=torch.float32).reshape(-1, 1)
    X_qtb_trn = torch.tensor(
        K_list_adv_trn, dtype=torch.float32).reshape(-1, 1)
    X_qtb_tst = torch.tensor(
        K_list_adv_tst, dtype=torch.float32).reshape(-1, 1)

    # y_pred_trn = curr_model(X_data_trn).squeeze(-1)
    # y_pred_tst = curr_model(X_data_tst).squeeze(-1)
    # y_qtb_trn = curr_model(X_qtb_trn).squeeze(-1)
    # y_qtb_tst = curr_model(X_qtb_tst).squeeze(-1)
    # loss_trn = loss_fn(y_pred_trn, y_data_trn).item()
    # loss_tst = loss_fn(y_pred_tst, y_data_tst).item()
    # loss_qtb_trn = loss_fn(y_qtb_trn, y_data_trn).item()
    # loss_qtb_tst = loss_fn(y_qtb_tst, y_data_tst).item()

    def _super_print(X_data, y_data, X_qtb, picked):
        y_pred = curr_model(X_data).squeeze(-1)
        y_qtb = curr_model(X_qtb).squeeze(-1)
        loss = loss_fn(y_pred, y_data).item()
        loss_qtb = loss_fn(y_qtb, y_data).item()

        elegant_print([
            f"Fair K Model: RMSE on {picked} samples",
            f"\tsquare=  normal {np.square(loss):.8f}  adv {np.square(loss_qtb):.8f}",
            f"\tloss  =  normal {loss:.8f}  adv {loss_qtb:.8f}",
            f"\tRMSE  =  normal {np.sqrt(loss):.8f}  adv {np.sqrt(loss_qtb):.8f}",
        ], logger)
        return y_pred, y_qtb

    y_insp, hx_trn = _super_print(
        X_data_trn, y_data_trn, X_qtb_trn, 'training')
    y_pred, hx_tst = _super_print(
        X_data_tst, y_data_tst, X_qtb_tst, 'test')
    return (y_data_trn, y_insp, hx_trn,
            y_data_tst, y_pred, hx_tst)


# Fair K model

def main_model_infer_K(train_set, test_set, logger=None,
                       figname='visualize_model',
                       train_adv=None, test_adv=None,
                       CURR_FYA_LOC=-2,
                       CURR_PATH='./'):  # None):
    # for CI testing
    # smoke_test = ('CI' in os.environ)
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)
    pyro.enable_validation(True)
    # setup
    assert issubclass(PyroModule[nn.Linear], nn.Linear)
    assert issubclass(PyroModule[nn.Linear], PyroModule)

    model_fairK, loss_fn, X_data, y_pred_test = _fair_k_infer_no1(
        train_set, test_set, CURR_FYA_LOC, CURR_PATH, figname, logger)
    visualization_parallel(train_set, test_set, model_fairK,
                           X_data, None, None, y_pred_test,
                           figname=figname + "_fairK")

    y_data_trn, y_data_tst = _fair_k_infer_no3q(
        train_set, test_set, train_adv, test_adv,
        CURR_FYA_LOC, CURR_PATH, logger)
    curr_dat_s = _fair_k_infer_no4q(y_data_trn, y_data_tst, model_fairK,
                                    loss_fn, CURR_PATH, logger)
    return model_fairK, loss_fn, curr_dat_s


# -----------------------
# Fair Add model


def _fairly_add_step1_dat_apart(data_set, sen_att=["race", "sex"],
                                logger=None):  # dat_partition
    race_sex_tensor = torch.tensor(
        data_set[sen_att].values, dtype=torch.float32)
    gpa_tensor = torch.tensor(
        data_set[["ugpa"]].values, dtype=torch.float32)[:, 0]
    lsat_tensor = torch.tensor(
        data_set[["lsat"]].values, dtype=torch.float32)[:, 0]
    return race_sex_tensor, gpa_tensor, lsat_tensor


def _fairly_add_step2_training(race_sex_tensor_train,
                               gpa_tensor_train, lsat_tensor_train,
                               logger=None):
    elegant_print(["-" * 20, "Predict GPA based on race and sex:"], logger)
    model_gpa_rs, _ = train_LinearModel(race_sex_tensor_train, gpa_tensor_train)
    elegant_print(["-" * 20, "Predict LSAT based on race and sex:"], logger)
    model_lsat_rs, _ = train_LinearModel(race_sex_tensor_train,
                                         lsat_tensor_train)
    return model_gpa_rs, model_lsat_rs


def _fairly_add_step3_residual(race_sex_tensor, gpa_tensor, lsat_tensor,
                               model_gpa_rs, model_lsat_rs, data_set):
    gpa_tensor_resid = gpa_tensor - model_gpa_rs(
        race_sex_tensor).detach().squeeze(0).numpy()[:, 0]
    lsat_tensor_resid = lsat_tensor - model_lsat_rs(
        race_sex_tensor).detach().squeeze(0).numpy()[:, 0]
    X_fair_add = torch.stack((gpa_tensor_resid, lsat_tensor_resid), dim=1)
    y_fair_add = torch.tensor(
        data_set[["zfygpa"]].values, dtype=torch.float32)[:, 0]
    return X_fair_add, y_fair_add


def main_model_Fair_Add(train_set, test_set, logger=None,
                        figname='visualize_model',
                        train_adv=None, test_adv=None):
    race_sex_tensor_train, gpa_tensor_train, lsat_tensor_train = \
        _fairly_add_step1_dat_apart(train_set, logger=logger)
    model_gpa_rs, model_lsat_rs = _fairly_add_step2_training(
        race_sex_tensor_train, gpa_tensor_train, lsat_tensor_train, logger)

    X_fair_add, y_fair_add = _fairly_add_step3_residual(
        race_sex_tensor_train, gpa_tensor_train, lsat_tensor_train,
        model_gpa_rs, model_lsat_rs, train_set)

    elegant_print("=" * 20, logger)
    elegant_print("Predict FYA based on residuals of GPA and LSAT:", logger)
    model_fair_add, loss_fn = train_LinearModel(X_fair_add, y_fair_add)

    race_sex_tensor_test, gpa_tensor_test, lsat_tensor_test = \
        _fairly_add_step1_dat_apart(test_set, logger=logger)
    X_fair_add_test, y_fair_add_test = _fairly_add_step3_residual(
        race_sex_tensor_test, gpa_tensor_test, lsat_tensor_test,
        model_gpa_rs, model_lsat_rs, test_set)

    y_pred_test = model_fair_add(X_fair_add_test).squeeze(-1)
    loss = loss_fn(y_pred_test, y_fair_add_test).item()
    elegant_print([
        # f"Fair add model: RMSE on test samples = {np.square(loss):.4f}",
        f"Fair add model: RMSE on test samples = {np.sqrt(loss):.4f}",
        ''], logger)
    # visualization(train_set, model_fair_add, X_fair_add)
    # visualization(test_set, model_fair_add, X_fair_add_test)
    visualization_parallel(train_set, test_set, model_fair_add,
                           X_fair_add, X_fair_add_test,
                           figname=figname + "_fairAdd")

    if (train_adv is None) or (test_adv is None):
        return
    rs_adv_trn, gpa_adv_trn, lsat_adv_trn = _fairly_add_step1_dat_apart(
        train_adv, logger=logger)
    rs_adv_tst, gpa_adv_tst, lsat_adv_tst = _fairly_add_step1_dat_apart(
        test_adv, logger=logger)
    assert (gpa_adv_trn == gpa_tensor_train).all()
    assert (lsat_adv_trn == lsat_tensor_train).all()
    assert (gpa_adv_tst == gpa_tensor_test).all()
    assert (lsat_adv_tst == lsat_tensor_test).all()

    X_qtb_add_trn, y_qtb_trn = _fairly_add_step3_residual(
        rs_adv_trn, gpa_adv_trn, lsat_adv_trn,
        model_gpa_rs, model_lsat_rs, train_adv)
    X_qtb_add_tst, y_qtb_tst = _fairly_add_step3_residual(
        rs_adv_tst, gpa_adv_tst, lsat_adv_tst,
        model_gpa_rs, model_lsat_rs, test_adv)
    assert (y_qtb_trn == y_fair_add).all()
    assert (y_qtb_tst == y_fair_add_test).all()

    _unaw_ful_no3_adversarial(
        X_fair_add, y_fair_add, X_qtb_add_trn, model_fair_add, loss_fn,
        "Fair add", 'training', logger)
    _unaw_ful_no3_adversarial(
        X_fair_add_test, y_fair_add_test, X_qtb_add_tst, model_fair_add,
        loss_fn, "Fair add", 'test', logger)

    y_insp = model_fair_add(X_fair_add).squeeze(-1)
    y_pred = model_fair_add(X_fair_add_test).squeeze(-1)
    hx_qtb_trn = model_fair_add(X_qtb_add_trn).squeeze(-1)
    hx_qtb_tst = model_fair_add(X_qtb_add_tst).squeeze(-1)
    return model_fair_add, loss_fn, (
        y_fair_add, y_insp, hx_qtb_trn,
        y_fair_add_test, y_pred, hx_qtb_tst)


# -----------------------


# -----------------------
