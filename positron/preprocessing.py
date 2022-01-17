from posixpath import split
import time
import datetime

import pandas as pd
import numpy as np


"""
:dates: np.ndarray containing dates in the given format
:format: str format string like "%d-%m-%Y"
"""
def dates_to_stamps(dates, format="%d-%m-%Y"):
    X = []
    for date in dates:
        X.append(time.mktime(datetime.datetime.strptime(date, format).timetuple()))
    return np.array(X)


"""
:stamps: np.ndarray containing unix timestamps
:format: str format string like "%d-%m-%Y"
"""
def stamps_to_dates(stamps, format="%d-%m-%Y"):
    X = []
    for stamp in stamps:
        X.append(datetime.datetime.utcfromtimestamp(int(stamp)).strftime(format))
    return np.array(X)


"""
Split data to train and test set according to the 
given test ratio
:data: pandas.DataFrame object
:test_ratio: float type containing the precentage, like: 0.2 for 20%
:returns: train, test data
"""
def split_train_test(data: pd.DataFrame, test_ratio=0.2):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def to_np(x: pd.Series, y: pd.Series, single=False):
    if single:
        return np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
    return np.array(x), np.array(y)


if __name__ == "__main__":
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = np.array([0, 2, 4, 8, 16, 32, 64, 128])
    df = pd.DataFrame()
    df["X"] = X
    df["y"] = y
    trainset, testset = split_train_test(df, test_ratio=0.4)

    print("X =", X, "( len =", len(y), ")")
    print("y =", y, "( len =", len(y), ")")
    print()
    print("xtest size =", len(testset["X"]))
    print("ytest size =", len(testset["y"]))
    print("xtrain size =", len(trainset["X"]))
    print("ytrain size =", len(trainset["y"]))

    date = ["17-01-2022"]
    print()
    print("Example date:", *date)
    print("Date as stamp:", *dates_to_stamps(date))
    print("Stamp as date:", *stamps_to_dates(dates_to_stamps(date)))
