from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from dataclasses import dataclass
from simple_parsing import ArgumentParser
import pandas as pd
import numpy as np
import pickle
import yaml


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())


yaml.add_representer(np.ndarray, ndarray_representer)


@dataclass
class TrainParams:
    max_depth: int = 5
    n_jobs: int = 2


@dataclass
class DataParams:
    labels_path: str
    data_path: str
    model_path: str
    metrics_path: str


def load_data(path):
    return pd.read_csv(path, index_col=0)


def load_labels(path):
    return pd.read_csv(path, index_col=0)


def train_score_model(data_params, train_params):
    data = load_data(data_params.data_path)
    labels = load_labels(data_params.labels_path)
    y_train = labels.loc[labels.validation, "target"].values
    X_train = data.loc[labels.validation, :].values
    y_test = labels.loc[~labels.validation, "target"].values
    X_test = data.loc[~labels.validation, :].values
    print(data.shape, labels.shape)
    model = DecisionTreeClassifier(
        max_depth=train_params.max_depth,
    )
    model.fit(X_train, y_train)
    print(y_test)
    print(model.predict(X_test))
    precision, recall, f1, support = map(
        lambda a: float(a[0]),
        precision_recall_fscore_support(
            y_test, model.predict(X_test), labels=[1], pos_label=1
        ),
    )

    return model, {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def save_metrics(metrics, path):
    with open(path, "w") as f:
        yaml.dump(metrics, f)


def main(data_params, train_params):
    model, metrics = train_score_model(data_params, train_params)
    save_model(model, data_params.model_path)
    save_metrics(metrics, data_params.metrics_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(DataParams, dest="data_params")
    parser.add_arguments(TrainParams, dest="train_params")
    args = parser.parse_args()
    main(args.data_params, args.train_params)
