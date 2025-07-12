# coding: utf-8


from fairml.datasets import DATASETS, DATASET_NAMES, RAW_EXPT_DIR


# ==================================
# Experiments
# ==================================


class DataSetup:
    def __init__(self, data_type):
        self._data_type = data_type
        self._log_document = data_type

        # ['ricci', 'german', 'adult', 'ppc', 'ppvc']
        if data_type == 'ppr':
            self._data_type = DATASET_NAMES[-2]
        elif data_type == 'ppvr':
            self._data_type = DATASET_NAMES[-1]
        elif data_type not in ['ricci', 'german', 'adult']:
            raise ValueError("Wrong dataset `{}`".format(data_type))
        idx = DATASET_NAMES.index(self._data_type)

        self._dataset = DATASETS[idx]
        self._data_frame = self._dataset.load_raw_dataset()

        if data_type == "ricci":
            self.saIndex = [2]  # 'Race' -2
        elif data_type == "german":
            self.saIndex = [3, 5]  # ['sex', 'age'] [, 12]
        elif data_type == "adult":
            self.saIndex = [2, 3]  # ['race', 'sex'] [7, 8]
        elif data_type == "ppc":
            self.saIndex = [0, 2]  # ['sex', 'race'] [0, 3]
        elif data_type == "ppvc":
            self.saIndex = [0, 2]  # ['sex', 'race'] [0, 3]
        self.saValue = self._dataset.get_privileged_group('numerical-binsensitive')
        # self.saValue = 0  # 1 means the privileged group
        self.saValue = [0 for sa in self.saValue if sa == 1]

    @property
    def data_type(self):
        return self._data_type

    # no use?
    @property
    def trial_type(self):
        return self._trial_type

    @property
    def log_document(self):
        return self._log_document

    # # ----------- mu -----------
    # def prepare_mu_datasets(self, ratio=.5, logger=None):
    #   pass
    # # ----------- tr -----------
    # # ----------- bi -----------
    # def prepare_bi_datasets(self, ratio=.5, logger=None):
    #   pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_frame(self):
        return self._data_frame


class GraphSetup:
    def __init__(self,
                 name_ens,
                 nb_cls,
                 nb_pru=None,
                 nb_iter=5,
                 figname=''):
        self._name_ens = name_ens
        self._nb_cls = nb_cls
        if nb_pru is None:
            nb_pru = nb_cls
        self._nb_pru = nb_pru
        self._nb_iter = nb_iter  # 5
        self._figname = figname

    @property
    def trial_type(self):
        return self._trial_type

    @property
    def name_ens(self):
        return self._name_ens

    @property
    def nb_cls(self):
        return self._nb_cls

    @property
    def nb_pru(self):
        return self._nb_pru

    @property
    def nb_iter(self):
        return self._nb_iter

    @property
    def figname(self):
        return self._figname

    @property
    def raw_dframe(self):
        return self._raw_dframe

    @raw_dframe.setter
    def raw_dframe(self, value):
        self._raw_dframe = value

    def get_raw_filename(self, trial_type = 'mCV_expt3'):
        nmens_tmp = _get_tmp_document(self._name_ens, self._nb_cls)
        filename = "{}_iter{}_pms.xlsx".format(nmens_tmp, self._nb_iter)
        if trial_type:
            trial_type += "_"
        # return osp.join(RAW_EXPT_DIR,
        return os.path.join(RAW_EXPT_DIR,
                            "{}{}".format(trial_type, filename))

    def load_raw_dataset(self, filename):
        dframe = {}
        for k, v in enumerate(AVAILABLE_ABBR_CLS):
            dframe[v] = pd.read_excel(filename, v)
        self._raw_dframe = dframe
        return dframe

    def recap_sub_data(self, dframe, nb_row=3):
        nb_set = [len(v) for k, v in dframe.items()]
        assert len(set(nb_set)) == 1, "{}-{}".format(
            self._name_ens, self._nb_cls)
        nb_set = list(set(nb_set))[0]
        nb_set = (nb_set - nb_row + 1) // (self._nb_iter + 1)

        id_set = [i * (
            self._nb_iter + 1) + nb_row - 1 for i in range(nb_set)]
        index = [[i + j for j in range(
            1, self._nb_iter + 1)] for i in id_set]
        return nb_set, id_set, index

    def fetch_sub_data(self, df, index, tag_col, tag_ats='ua'):
        # index = np.concatenate(index).tolist()
        data = df.iloc[index][tag_col]
        # data = dframe.iloc[index, tag_col]

        if tag_ats == 'us':
            data.fillna(self._nb_cls, inplace=True)
        elif tag_ats == 'ua':
            data *= 100.
        data = data.values.astype('float')  # .to_numpy()
        return data  # np.ndarray

    def stats_sub_calc(self, data, nb_set, nb_col):
        avg = np.zeros((nb_set, nb_col))
        std = np.zeros((nb_set, nb_col))
        var = np.zeros((nb_set, nb_col))

        for i in range(nb_set):
            loc_idx = np.arange(self._nb_iter) + i * self._nb_iter
            now_set = data[loc_idx]

            avg[i] = now_set.mean(axis=0)
            std[i] = now_set.std(axis=0, ddof=1)
            var[i] = now_set.var(axis=0, ddof=1)
        return avg, std, var

    def merge_sub_data(self, dframe, index, tag_col, tag_ats='us'):
        nb_set = np.shape(index)[0]
        index = np.concatenate(index).tolist()
        nb_col = len(tag_col)

        raw_data = {}
        avg, std, var = {}, {}, {}

        for k, v in enumerate(AVAILABLE_ABBR_CLS):
            # dframe[v].iloc[index][tag_col]
            raw_data[v] = self.fetch_sub_data(
                dframe[v], index, tag_col, tag_ats)

            avg[v], std[v], var[v] = self.stats_sub_calc(
                raw_data[v], nb_set, nb_col)

        raw = [v for k, v in raw_data.items()]
        raw = np.concatenate(raw, axis=0)
        avg = np.concatenate([v for k, v in avg.items()], axis=0)
        std = np.concatenate([v for k, v in std.items()], axis=0)
        var = np.concatenate([v for k, v in var.items()], axis=0)
        return raw, avg, std, var

    # Pandas
    # def schedule_content(self):
    #   raise NotImplementedError

    def schedule_mspaint(self, raw_dframe, tag_col):
        raise NotImplementedError

    def subdraw_spliting(self):
        raise NotImplementedError

    def subdraw_asawhole(self):
        raise NotImplementedError

    def prepare_graph(self):
        raise NotImplementedError


# ==================================
# Experiments
# ==================================
