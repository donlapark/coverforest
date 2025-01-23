"""Benchmarking run time of CoverForestClassifier"""

from coverforest import CoverForestClassifier

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

rng = np.random.default_rng(2)
randints = rng.integers(0, high=1e6, size=30)
alpha = 0.5


################################
# Training & Evaluation workflow
################################


def experiment(data_name, X_train, y_train, X_test, y_test, method_list, params):
    for method, k_, lambda_ in method_list:
        n_preds = []
        coverage_probs = []
        echo = []
        train_times = []
        test_times = []
        for k in range(30):
            rfclf = CoverForestClassifier(
                n_estimators=params[method]["n_estimators"],
                method=method,
                allow_empty_sets=True,
                cv=10,
                k_init=k_,
                lambda_init=lambda_,
                n_jobs=-1,
                random_state=randints[k],
            )
            curr1 = time.time()
            rfclf.fit(X_train, y_train, alpha=alpha)
            curr2 = time.time()
            curr3 = time.time()
            _, y_pred = rfclf.predict(X_test, alpha=alpha, binary_output=False)
            curr4 = time.time()
            train_times.append(curr2 - curr1)
            test_times.append(curr4 - curr3)
            print("Training time:", curr2 - curr1, curr4 - curr3)
            avg_size = np.mean([y_pred[i].shape[0] for i in range(len(y_pred))])
            cvg_prob = np.mean([y_test[i] in y_pred[i] for i in range(len(y_pred))])
            n_preds.append(avg_size)
            coverage_probs.append(cvg_prob)
            echo.append(k + 1)
            print("echo", k, ", alpha =", alpha)
            print(f"average size = {avg_size}, coverage = {cvg_prob}.")

        method_ = method.replace("_", "")
        filename = f"{data_name}_{method_}_Ours.csv"
        with open(filename, "w", newline="") as f:
            write = csv.writer(os.path.join(result_folder, f))
            write.writerows([echo, n_preds, coverage_probs, train_times, test_times])


# Set up methods and parameters.
method_list = [("bootstrap", 0, 0), ("cv", 0, 0), ("split", 0, 0)]

param_dict = {
    "bootstrap": {"n_estimators": int(1000 * 0.9)},
    "cv": {"n_estimators": 100},
    "split": {"n_estimators": int(1000 * 0.9)},
}


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
