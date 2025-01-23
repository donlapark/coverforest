"""Benchmarking run time of MapieClassifier + RandomForestClassifier"""

from mapie.classification import MapieClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
import os
from scipy.stats import mode
import time
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

result_folder = "results_classification_time"

if not os.path.exists(result_folder):
    os.makedirs(result_folder)


rng = np.random.default_rng(1)
randints = rng.integers(0, high=1e6, size=30)
alpha = 0.05


################################
# Training & Evaluation workflow
################################


def experiment(data_name, X_train, y_train, X_test, y_test, method_list, params):
    for method, method_name, k_, lambda_ in method_list:
        echo = []
        train_times = []
        test_times = []
        for k in range(30):
            rf = RandomForestClassifier(n_estimators=params[method]["n_estimators"])
            rfcv = MapieClassifier(
                estimator=rf,
                method="aps",
                cv=method,
                test_size=0.3,
                random_state=randints[k],
                n_jobs=-1,
            )
            curr1 = time.time()
            rfcv.fit(X_train, y_train)
            curr2 = time.time()
            curr3 = time.time()
            _, y_pred = rfcv.predict(
                X_test,
                alpha=alpha,
                include_last_label="randomized",
                agg_scores="crossval",
            )
            curr4 = time.time()
            train_times.append(curr2 - curr1)
            test_times.append(curr4 - curr3)
            print("time", curr2 - curr1, curr4 - curr3)
            echo.append(k + 1)
            print("echo", k, ", alpha =", alpha)

        filename = f"{data_name}_{method_name}_MAPIE.csv"
        with open(filename, "w", newline="") as f:
            write = csv.writer(os.path.join(result_folder, f))
            write.writerows([echo, echo, echo, train_times, test_times])


# Set up methods and parameters.
method_list = [(10, "cv", 0, 0), ("split", "split", 0, 0)]

param_dict = {10: {"n_estimators": 100}, "split": {"n_estimators": int(1000 * 0.9)}}


##############
# Mice dataset
##############

MiceData = pd.read_csv("data/MiceClean.csv")

X = MiceData.drop(columns=["class", "MouseID"])
y = MiceData["class"]
y = y.values.reshape(-1, 1)
y = y[:, 0]

seed = 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=seed
)

experiment("MiceData", X_train, y_train, X_test, y_test, method_list, param_dict)


#####################
# WineQuality dataset
#####################
wine_quality = pd.read_csv("data/winequality-white.csv", sep=";")

seed = 123

sample_wine = wine_quality

X = sample_wine.drop(columns=["quality"])
y = sample_wine["quality"]
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=seed
)

# Normalize data to prevent numerical overflows.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

experiment("WineData", X_train, y_train, X_test, y_test, method_list, param_dict)


####################
# Myocardial dataset
####################

myocardial = fetch_ucirepo(id=579)

X = myocardial.data.features
y = myocardial.data.targets

y = y["LET_IS"]
y = y.to_numpy()

seed = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=seed
)

experiment("MyocData", X_train, y_train, X_test, y_test, method_list, param_dict)


###############
# MNIST dataset
###############

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train[0:5000]
y_train = y_train[0:5000]

X_test = X_test[0:1250]
y_test = y_test[0:1250]

experiment("MNISTData", X_train, y_train, X_test, y_test, method_list, param_dict)
