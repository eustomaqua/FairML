# coding: utf-8
import pdb


def test_datasets():
    from fairml.datasets import (
        Ricci, German, Adult, PropublicaRecidivism,
        PropublicaViolentRecidivism, preprocess)

    dt = Ricci()
    dt = German()
    dt = Adult()
    dt = PropublicaRecidivism()
    dt = PropublicaViolentRecidivism()

    df = dt.load_raw_dataset()
    ans = preprocess(dt, df)

    # pdb.set_trace()
    return
