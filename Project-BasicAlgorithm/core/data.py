import os

import pandas as pd
from sklearn.model_selection import train_test_split

PATH_ERROR_MESSAGE = (
    "data_path only support csv data or directory contained train.csv and test.csv"
)


def load_data(data_path, label_column, test_size=0.25, random_state=1):
    if os.path.isdir(data_path):
        train_path = os.path.join(data_path, "train.csv")
        test_path = os.path.join(data_path, "test.csv")
        assert os.path.exists(train_path) and os.path.exists(
            test_path
        ), PATH_ERROR_MESSAGE

        print(f"load train data from {train_path}")
        print(f"load test data from {test_path}")
        train_x, train_y = load_csv_data(train_path, label_column)
        test_x, test_y = load_csv_data(test_path, label_column)

    elif data_path.endswith(".csv"):
        print(f"load data from {data_path}")
        print("split data to train set and test set")
        train_x, train_y, test_x, test_y = load_split_csv_data(
            data_path, label_column, test_size=test_size, random_state=random_state
        )

    else:
        raise Exception(PATH_ERROR_MESSAGE)

    return train_x, train_y, test_x, test_y


def load_split_csv_data(data_path, label_column, test_size=0.25, random_state=1):

    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    train_x = train.drop([label_column], axis=1)
    test_x = test.drop([label_column], axis=1)
    train_y = train[[label_column]]
    test_y = test[[label_column]]
    return train_x, train_y, test_x, test_y


def load_csv_data(data_path, label_column):

    data = pd.read_csv(data_path)
    x = data.drop([label_column], axis=1)
    y = data[[label_column]]
    return x, y
