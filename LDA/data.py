import pandas as pd

import numpy as np
from collections import namedtuple

Data = namedtuple('Data', field_names=['X', 'Y'])


def get_train_dataset(p = './data/train.csv'):

    df = pd.read_csv(p)

    df = np.array(df)

    Y = df[:, 0]
    X = df[:, 1:]

    print(Y.shape, X.shape)

    train_data = Data(X, Y)
    print('Total number of training samples: {}'.format(Y.shape[0]))
    return train_data


def get_test_dataset(p = './data/test.csv'):

    df = pd.read_csv(p)

    df = np.array(df)

    Y = df[:, 0]
    X = df[:, 1:]

    print('Total number of testing samples: {}'.format(Y.shape[0]))
    test_data = Data(X, Y)
    return test_data
